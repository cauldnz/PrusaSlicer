// pyslic3r read-only object model (M1) — PrusaSlicer port.
//
// Exposes Application / Document / Model tree / Config / Plates as an
// inspection-only surface. Invariants:
//   - Read-only: every member is a getter; nothing here mutates the document,
//     config, or presets.
//   - UI-parity: everything routes through Plater and the objects it owns
//     (Model, DynamicPrintConfig, PresetBundle) — never a parallel data path.
//   - Main-thread-guarded: every accessor asserts the wx main thread, so
//     off-thread callers must come through run_on_main_blocking().
//   - Handles are indices, re-resolved per call and bounds-checked, so a stale
//     Python handle raises rather than dereferencing a freed pointer if the
//     model changed between calls.
//
// PrusaSlicer note: PrusaSlicer is single-bed (no PartPlate). The Plate/PlateList
// surface is kept for cross-platform API parity but is backed by the single bed:
// count == 1, per-plate config maps to the global config, per-plate custom g-code
// maps to Model::custom_gcode_per_print_z(), and slice validity/stats come from
// Plater::active_fff_print() + get_gcode_results().

#include "PyBindings.hpp"

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/stl.h>

#include <wx/app.h>
#include <wx/thread.h>

#include "libslic3r/libslic3r.h"
#include "libslic3r/Model.hpp"
#include "libslic3r/TriangleSelector.hpp"   // facet paint (MMU)
#include "libslic3r/QuadricEdgeCollapse.hpp"   // object.simplify
#include "libslic3r/TriangleMesh.hpp"   // its_volume (object.measure)
#include "libslic3r/MeshBoolean.hpp"     // object.boolean
#include "libslic3r/BuildVolume.hpp"     // object.center (bed_center)
#include "libslic3r/NSVGUtils.hpp"      // object.emboss_svg (SVG -> shapes)
#include "libslic3r/Emboss.hpp"         // object.emboss_svg (shapes -> mesh)
#include "libslic3r/ClipperUtils.hpp"   // union_ex (object.emboss_svg)
#include <limits>
#include "libslic3r/Format/STL.hpp"   // store_stl (export_stl)
#include "libslic3r/Emboss.hpp"          // text2shapes / polygons2model (FreeType emboss)
#include "libslic3r/EmbossShape.hpp"     // EmbossShape / HealedExPolygons
#include "slic3r/Utils/WxFontUtils.hpp"  // font -> FontFile (Prusa has no load_text_shape)
#include "libslic3r/BoundingBox.hpp"
#include "libslic3r/Config.hpp"
#include "libslic3r/Preset.hpp"
#include "libslic3r/PresetBundle.hpp"
#include "libslic3r/Print.hpp"                 // Print, PrintStatistics
#include "libslic3r/Slicing.hpp"               // variable layer height
#include "libslic3r/GCode/GCodeProcessor.hpp"  // GCodeProcessorResult
#include "libslic3r/CustomGCode.hpp"      // colour-change-by-height
#include "slic3r/GUI/GUI_App.hpp"
#include "slic3r/GUI/GUI_ObjectList.hpp"   // obj_list()->delete_all_objects_from_list()
#include "slic3r/GUI/Plater.hpp"
#include "libslic3r/CutUtils.hpp"   // Cut class + ModelObjectCutAttribute (reworked cut)
#include "slic3r/GUI/GLCanvas3D.hpp"   // canvas3D()->get_selection()
#include "slic3r/GUI/Selection.hpp"     // Selection::add_object (headless select)
#include "slic3r/GUI/Tab.hpp"
#include "slic3r/GUI/BackgroundSlicingProcess.hpp"   // job.cancel() -> stop()

#include <boost/filesystem/path.hpp>
#include <chrono>
#include <cmath>
#include <algorithm>   // std::sort
#include <map>

#include <wx/utils.h>   // wxMilliSleep

namespace py = pybind11;
using namespace Slic3r;

namespace pyslic3r {
namespace {

// ---- guards + resolvers ---------------------------------------------------

void main_thread(const char *what)
{
    if (!wxThread::IsMain())
        throw std::runtime_error(std::string(what) +
            " must be accessed on the wx main thread; use the marshalling primitive");
}

GUI::Plater *plater_or_throw(const char *what)
{
    main_thread(what);
    auto *p = GUI::wxGetApp().plater();
    if (p == nullptr)
        throw std::runtime_error("no active document");
    return p;
}

Model &model_or_throw(const char *what) { return plater_or_throw(what)->model(); }

ModelObject *object_at(size_t idx, const char *what)
{
    Model &m = model_or_throw(what);
    if (idx >= m.objects.size())
        throw std::runtime_error("object index out of range (model changed?)");
    return m.objects[idx];
}

// ---- small value converters ----------------------------------------------

py::tuple vec3(const Vec3d &v) { return py::make_tuple(v.x(), v.y(), v.z()); }

const char *volume_type_str(ModelVolumeType t)
{
    switch (t) {
    case ModelVolumeType::MODEL_PART:         return "model_part";
    case ModelVolumeType::NEGATIVE_VOLUME:    return "negative_volume";
    case ModelVolumeType::PARAMETER_MODIFIER: return "modifier";
    case ModelVolumeType::SUPPORT_BLOCKER:    return "support_blocker";
    case ModelVolumeType::SUPPORT_ENFORCER:   return "support_enforcer";
    default:                                  return "invalid";
    }
}

ModelVolumeType volume_type_from_str(const std::string &s)
{
    if (s == "part"    || s == "model_part")         return ModelVolumeType::MODEL_PART;
    if (s == "modifier"|| s == "parameter_modifier") return ModelVolumeType::PARAMETER_MODIFIER;
    if (s == "negative"|| s == "negative_volume")    return ModelVolumeType::NEGATIVE_VOLUME;
    if (s == "support_blocker"  || s == "blocker")   return ModelVolumeType::SUPPORT_BLOCKER;
    if (s == "support_enforcer" || s == "enforcer")  return ModelVolumeType::SUPPORT_ENFORCER;
    throw std::runtime_error("unknown volume type '" + s +
        "' (part/modifier/negative/support_blocker/support_enforcer)");
}

// ---- handle types ---------------------------------------------------------
// Each holds indices only; the referenced C++ object is re-resolved per call.

struct PyApp {};
struct PyDocument {};
struct PyModel {};
struct PyObject   { size_t idx; };
struct PyVolume   { size_t obj_idx; size_t vol_idx; };
struct PySettings {};
struct PyFilament { size_t slot; };
struct PyText     { size_t obj_idx; size_t vol_idx; };
struct PyPlateList {};
struct PyPlate    { int idx; };

// Which config a PyConfig fronts.
enum class ConfigSource { Global, Print, Filament, Printer, Plate, Object, Volume };
struct PyConfig { ConfigSource source; int plate_idx = 0; int vol_idx = 0; };

// M3 slicing handles.
struct PySliceJob { int plate_idx; };
struct PySliceResult
{
    bool                    success = false;
    long                    print_time_s = 0;
    long                    layer_count = 0;
    double                  total_weight = 0;               // grams (sum of filament_g)
    double                  total_cost = 0;                 // currency units
    double                  total_wipe_tower_filament = 0;  // mm
    int                     total_toolchanges = 0;
    std::map<int, double>   filament_g;    // per extruder/slot
    std::map<int, double>   filament_mm;   // per extruder/slot (from volume)
    std::string             gcode_path;
    std::string             error;         // reason when !success
};

// The PresetCollection backing a preset source, or nullptr for Global/Plate.
PresetCollection *preset_collection(ConfigSource s)
{
    auto *pb = GUI::wxGetApp().preset_bundle;
    switch (s) {
    case ConfigSource::Print:    return &pb->prints;
    case ConfigSource::Filament: return &pb->filaments;
    case ConfigSource::Printer:  return &pb->printers;
    default:                     return nullptr;
    }
}

Preset::Type preset_type(ConfigSource s)
{
    switch (s) {
    case ConfigSource::Print:    return Preset::TYPE_PRINT;
    case ConfigSource::Filament: return Preset::TYPE_FILAMENT;
    case ConfigSource::Printer:  return Preset::TYPE_PRINTER;
    default:                     return Preset::TYPE_INVALID;
    }
}

// Read view. For preset sources this is the *edited* preset — the working copy
// the GUI reads and writes — so a value set via Config.set reads straight back.
const ConfigBase *resolve_config(const PyConfig &c, const char *what)
{
    switch (c.source) {
    case ConfigSource::Global:
    case ConfigSource::Plate: {
        // Single-bed platform: the "plate" config is the global (derived) config.
        const DynamicPrintConfig *cfg = plater_or_throw(what)->config();
        if (cfg == nullptr) throw std::runtime_error("no global config");
        return cfg;
    }
    case ConfigSource::Volume: {
        // per-volume overrides (ModelVolume::config); plate_idx=object, vol_idx=volume.
        ModelObject *obj = object_at(c.plate_idx, what);
        if (size_t(c.vol_idx) >= obj->volumes.size())
            throw std::runtime_error("volume index out of range");
        return &obj->volumes[c.vol_idx]->config.get();
    }
    case ConfigSource::Object: {
        // per-object overrides (ModelObject::config); plate_idx = object index.
        return &object_at(c.plate_idx, what)->config.get();
    }
    case ConfigSource::Print:
    case ConfigSource::Filament:
    case ConfigSource::Printer: {
        main_thread(what);
        return &preset_collection(c.source)->get_edited_preset().config;
    }
    }
    throw std::runtime_error("unknown config source");
}

ModelVolume *volume_at(const PyVolume &v, const char *what)
{
    ModelObject *obj = object_at(v.obj_idx, what);
    if (v.vol_idx >= obj->volumes.size())
        throw std::runtime_error("volume index out of range (model changed?)");
    return obj->volumes[v.vol_idx];
}

// MMU facet paint helper: paint original facets whose world-Z centroid is in
// (z_lo, z_hi] with the given 1-based extruder. Composes with existing paint.
static int mmu_paint_band(const PyVolume &v, double z_lo, double z_hi, int extruder)
{
    if (extruder < 1 || extruder > 16)
        throw std::runtime_error("extruder must be in 1..16");
    ModelObject *obj = object_at(v.obj_idx, "Volume.paint_mmu");
    ModelVolume *vol = volume_at(v, "Volume.paint_mmu");
    const TriangleMesh &mesh = vol->mesh();
    const indexed_triangle_set &its = mesh.its;
    if (its.indices.empty())
        throw std::runtime_error("paint_mmu: volume has no mesh");
    Transform3d tr = vol->get_matrix();
    if (!obj->instances.empty())
        tr = obj->instances[0]->get_matrix() * tr;
    TriangleSelector selector(mesh);
    if (!vol->mm_segmentation_facets.get_data().triangles_to_split.empty())
        selector.deserialize(vol->mm_segmentation_facets.get_data(), false);
    const TriangleStateType state = static_cast<TriangleStateType>(extruder);
    int painted = 0;
    for (size_t i = 0; i < its.indices.size(); ++i) {
        const auto &t = its.indices[i];
        const Vec3d c = (its.vertices[t[0]].cast<double>() + its.vertices[t[1]].cast<double>() +
                         its.vertices[t[2]].cast<double>()) / 3.0;
        const double wz = (tr * c).z();
        if (wz > z_lo && wz <= z_hi) { selector.set_facet(int(i), state); ++painted; }
    }
    vol->mm_segmentation_facets.set(selector);
    return painted;
}

// Generic facet-annotation paint: paint original facets whose world-Z centroid is
// in (z_lo, z_hi] with `state`, into the given FacetsAnnotation member. Composes
// with existing paint. Shared by support + seam (MMU has its own extruder wrapper).
static int paint_ann_band(const PyVolume &v, double z_lo, double z_hi,
                          TriangleStateType state, FacetsAnnotation ModelVolume::*member)
{
    ModelObject *obj = object_at(v.obj_idx, "Volume.paint");
    ModelVolume *vol = volume_at(v, "Volume.paint");
    const TriangleMesh &mesh = vol->mesh();
    const indexed_triangle_set &its = mesh.its;
    if (its.indices.empty())
        throw std::runtime_error("paint: volume has no mesh");
    Transform3d tr = vol->get_matrix();
    if (!obj->instances.empty())
        tr = obj->instances[0]->get_matrix() * tr;
    FacetsAnnotation &facets = vol->*member;
    TriangleSelector selector(mesh);
    if (!facets.get_data().triangles_to_split.empty())
        selector.deserialize(facets.get_data(), false);
    int painted = 0;
    for (size_t i = 0; i < its.indices.size(); ++i) {
        const auto &t = its.indices[i];
        const Vec3d c = (its.vertices[t[0]].cast<double>() + its.vertices[t[1]].cast<double>() +
                         its.vertices[t[2]].cast<double>()) / 3.0;
        const double wz = (tr * c).z();
        if (wz > z_lo && wz <= z_hi) { selector.set_facet(int(i), state); ++painted; }
    }
    facets.set(selector);
    return painted;
}

// "enforce"/"block" -> ENFORCER/BLOCKER
static TriangleStateType eb_mode(const std::string &mode)
{
    if (mode == "enforce" || mode == "enforcer") return TriangleStateType::ENFORCER;
    if (mode == "block"   || mode == "blocker")  return TriangleStateType::BLOCKER;
    throw std::runtime_error("mode must be 'enforce' or 'block'");
}



// Resolve a text handle -> ModelVolume, asserting it is editable emboss text.
ModelVolume *text_at(const PyText &t, const char *what)
{
    ModelVolume *v = volume_at(PyVolume{t.obj_idx, t.vol_idx}, what);
    if (!v->is_text() || !v->emboss_shape.has_value())
        throw std::runtime_error(std::string(what) + ": volume is not editable text");
    return v;
}

// Regenerate a text volume's mesh for `text` at emboss `scale`, via libslic3r
// Emboss (Prusa has no load_text_shape). NOTE: in Prusa the rendered SIZE is
// carried by emboss_shape.scale, not FontProp.size_in_mm — text2shapes emits
// glyphs at a fixed em-size and polygons2model applies `scale`. So resizing
// (fit) scales `scale`, not size_in_mm. Depth stays in mm (projection.depth).
// Generated at local Z origin; the caller re-aligns Z to the original mesh so
// surface placement is exact without recovering the original is_outside flag.
static TriangleMesh _prusa_text_mesh(const ModelVolume *v, const std::string &text,
                                     const FontProp &fp, double scale)
{
    wxFont wx_font = GUI::WxFontUtils::create_wxFont(v->text_configuration->style);
    if (!wx_font.IsOk() && fp.face_name.has_value())
        wx_font.SetFaceName(wxString::FromUTF8(fp.face_name->c_str()));
    std::unique_ptr<Emboss::FontFile> ffp = GUI::WxFontUtils::create_font_file(wx_font);
    if (!ffp)
        throw std::runtime_error("Text.set_text: could not load font (face '" +
            (fp.face_name.has_value() ? *fp.face_name : std::string("?")) + "')");
    Emboss::FontFileWithCache font(std::move(ffp));
    HealedExPolygons shapes = Emboss::text2shapes(font, text.c_str(), fp);
    if (shapes.expolygons.empty())
        throw std::runtime_error("Text.set_text: empty glyph shapes (font/text?)");
    double depth  = v->emboss_shape->projection.depth / scale;
    auto projectZ = std::make_unique<Emboss::ProjectZ>(depth);
    Transform3d tr = Transform3d::Identity();
    tr.scale(scale);
    Emboss::ProjectTransform project(std::move(projectZ), tr);
    return TriangleMesh(Emboss::polygons2model(shapes.expolygons, project));
}

// mm width of `text` at emboss `scale` (bbox X of the regenerated mesh).
static double _prusa_text_width(const ModelVolume *v, const std::string &text,
                                const FontProp &fp, double scale)
{
    return _prusa_text_mesh(v, text, fp, scale).bounding_box().size().x();
}

// Read slice stats off the sliced print into a PySliceResult. Main thread.
// Single-bed: the current g-code result is get_gcode_results().front().
void fill_slice_result(int /*plate_idx*/, PySliceResult &res)
{
    auto *plater = plater_or_throw("SliceResult");
    const auto &results = plater->get_gcode_results();
    if (results.empty()) return;
    const GCodeProcessorResult &gr = results.front();

    res.gcode_path = gr.filename;

    const auto &st   = gr.print_statistics;
    const auto &mode = st.modes[0];   // 0 = Normal
    res.print_time_s = long(mode.time + 0.5f);
    { long _lc = 0; for (const auto &mv : gr.moves) if (long(mv.layer_id) > _lc) _lc = long(mv.layer_id);
      res.layer_count = gr.moves.empty() ? 0L : _lc + 1; }   // Mode has no layers_times

    // volumes_per_extruder is mm^3 of extruded filament per extruder.
    // length mm = volume / filament cross-section (Ø1.75 mm default).
    // weight g  = volume(cm^3) * density(g/cm^3), density per extruder.
    const double area = M_PI * (1.75 / 2.0) * (1.75 / 2.0);   // mm^2
    for (const auto &kv : st.volumes_per_extruder) {
        const int    e       = int(kv.first);
        const double vol_mm3 = kv.second;
        res.filament_mm[e] = area > 0 ? vol_mm3 / area : 0.0;
        const double density = (e >= 0 && e < int(gr.filament_densities.size()))
                                   ? double(gr.filament_densities[e]) : 1.24;  // PLA fallback
        res.filament_g[e]  = vol_mm3 / 1000.0 * density;      // mm^3 -> cm^3 * g/cm^3
    }
    {
        const PrintStatistics &_ps = plater->active_fff_print().print_statistics();
        res.total_weight              = _ps.total_weight;
        res.total_cost                = _ps.total_cost;
        res.total_toolchanges         = _ps.total_toolchanges;
        res.total_wipe_tower_filament = _ps.total_wipe_tower_filament;
    }
    if (res.total_weight <= 0.0)   // fallback: sum per-extruder filament_g
        for (const auto &_kv : res.filament_g) res.total_weight += _kv.second;
}

// Point the GUI selection at a single object (headless) — the seam the
// selection-based Plater ops (fill_bed, split) bind to. Mirrors a click.
static void _select_only(size_t obj_idx, const char *what)
{
    (void) object_at(obj_idx, what);   // bounds-check
    GUI::Selection &sel = plater_or_throw(what)->canvas3D()->get_selection();
    sel.clear();
    sel.add_object((unsigned int) obj_idx, true);
}


// ---- curated settings registry ----------------------------------------------
struct SettingDef {
    std::string name;
    std::string key;
    char        kind;        // f i b p e o
    bool        is_vector;
    std::vector<std::pair<std::string,std::string>> emap;  // canonical -> fork string
};
static const std::vector<SettingDef> &settings_defs()
{
    static const std::vector<SettingDef> defs = {
        {"layer_height","layer_height",'f',false,{}},
        {"first_layer_height","first_layer_height",'o',false,{}},
        {"wall_count","perimeters",'i',false,{}},
        {"infill_density","fill_density",'p',false,{}},
        {"infill_pattern","fill_pattern",'e',false,{{"grid","grid"},{"gyroid","gyroid"},{"honeycomb","honeycomb"},{"cubic","cubic"},{"concentric","concentric"},{"triangles","triangles"},{"lightning","lightning"}}},
        {"top_layers","top_solid_layers",'i',false,{}},
        {"bottom_layers","bottom_solid_layers",'i',false,{}},
        {"top_pattern","top_fill_pattern",'e',false,{{"monotonic","monotonic"},{"concentric","concentric"},{"aligned","alignedrectilinear"}}},
        {"seam_position","seam_position",'e',false,{{"nearest","nearest"},{"aligned","aligned"},{"rear","rear"},{"random","random"}}},
        {"supports_enable","support_material",'b',false,{}},
        {"supports_threshold","support_material_threshold",'i',false,{}},
        {"supports_on_plate_only","support_material_buildplate_only",'b',false,{}},
        {"supports_style","support_material_style",'e',false,{{"grid","grid"},{"snug","snug"},{"organic","organic"}}},
        {"brim_type","brim_type",'e',false,{{"none","no_brim"},{"outer","outer_only"},{"inner","inner_only"},{"both","outer_and_inner"}}},
        {"brim_width","brim_width",'f',false,{}},
        {"skirt_loops","skirts",'i',false,{}},
        {"raft_layers","raft_layers",'i',false,{}},
        {"fuzzy_skin","fuzzy_skin",'e',false,{{"none","none"},{"external","external"},{"all","all"}}},
        {"spiral_vase","spiral_vase",'b',false,{}},
        {"wall_generator","perimeter_generator",'e',false,{{"classic","classic"},{"arachne","arachne"}}},
        {"elephant_foot","elefant_foot_compensation",'f',false,{}},
        {"inner_wall_speed","perimeter_speed",'f',false,{}},
        {"outer_wall_speed","external_perimeter_speed",'f',false,{}},
        {"infill_speed","infill_speed",'f',false,{}},
        {"travel_speed","travel_speed",'f',false,{}},
        {"first_layer_speed","first_layer_speed",'f',false,{}},
        {"nozzle_temp","temperature",'i',true,{}},
        {"first_layer_nozzle_temp","first_layer_temperature",'i',true,{}},
        {"bed_temp","bed_temperature",'i',true,{}},
        {"fan_max","max_fan_speed",'i',true,{}},
        {"retract_length","retract_length",'f',true,{}},
        {"retract_speed","retract_speed",'f',true,{}},
        {"z_hop","retract_lift",'f',true,{}},
        {"wipe_tower","wipe_tower",'b',false,{}},
        {"bottom_pattern","bottom_fill_pattern",'e',false,{{"monotonic","monotonic"},{"concentric","concentric"},{"aligned","alignedrectilinear"}}},
        {"solid_infill_speed","solid_infill_speed",'f',false,{}},
        {"top_surface_speed","top_solid_infill_speed",'f',false,{}},
        {"bridge_speed","bridge_speed",'f',false,{}},
        {"support_speed","support_material_speed",'f',false,{}},
        {"bridge_flow","bridge_flow_ratio",'f',false,{}},
        {"fan_min","min_fan_speed",'i',true,{}},
        {"supports_top_gap","support_material_contact_distance",'f',false,{}},
        {"supports_pattern","support_material_pattern",'e',false,{{"rectilinear","rectilinear"},{"grid","rectilinear-grid"},{"honeycomb","honeycomb"}}},
        {"supports_interface_layers","support_material_interface_layers",'i',false,{}},
        {"brim_gap","brim_separation",'f',false,{}},
        {"skirt_distance","skirt_distance",'f',false,{}},
        {"retract_on_layer_change","retract_layer_change",'b',false,{}},
        {"detect_thin_walls","thin_walls",'b',false,{}},
        {"avoid_crossing_walls","avoid_crossing_perimeters",'b',false,{}},
        {"resolution","resolution",'f',false,{}},
        {"infill_wall_overlap","infill_overlap",'p',false,{}},
    };
    return defs;
}
static const SettingDef *find_setting(const std::string &name)
{
    for (const auto &d : settings_defs()) if (d.name == name) return &d;
    return nullptr;
}
// Auto-resolve which edited preset config carries the key (print/filament/printer).
static DynamicPrintConfig *setting_cfg(const std::string &key, PresetCollection **out_col)
{
    PresetBundle *pb = GUI::wxGetApp().preset_bundle;
    PresetCollection *cols[] = { &pb->prints, &pb->filaments, &pb->printers };
    for (auto *col : cols) {
        DynamicPrintConfig &cfg = col->get_edited_preset().config;
        if (cfg.has(key)) { if (out_col) *out_col = col; return &cfg; }
    }
    return nullptr;
}
static void settings_set(const std::string &name, py::object val)
{
    const SettingDef *d = find_setting(name);
    if (!d) throw std::runtime_error("unknown setting: " + name);
    main_thread("Settings.set");
    std::string sval;
    if (d->kind == 'b') {
        sval = val.cast<bool>() ? "1" : "0";
    } else if (d->kind == 'e') {
        const std::string canon = py::str(val).cast<std::string>();
        bool ok = false;
        for (const auto &p : d->emap) if (p.first == canon) { sval = p.second; ok = true; break; }
        if (!ok) throw std::runtime_error("invalid option '" + canon + "' for setting '" + name + "'");
    } else if (d->kind == 'i') {
        sval = std::to_string((long long) std::llround(val.cast<double>()));
    } else if (d->kind == 'p') {
        // percent: force an explicit '%' so both forks parse it the same way
        // (Prusa's bare-number coPercent deserialize scales x100).
        std::string s = py::isinstance<py::str>(val) ? val.cast<std::string>()
                                                     : py::str(val).cast<std::string>();
        if (s.empty() || s.back() != '%') s += "%";
        sval = s;
    } else {
        sval = py::isinstance<py::str>(val) ? val.cast<std::string>()
                                            : py::str(val).cast<std::string>();
    }
    auto *plater = plater_or_throw("Settings.set");
    PresetCollection *col = nullptr;
    DynamicPrintConfig *cfg = setting_cfg(d->key, &col);
    if (!cfg) throw std::runtime_error("no preset config carries key: " + d->key);
    if (d->is_vector) {   // replicate a scalar across the current element count
        const std::string cur = cfg->opt_serialize(d->key);
        int n = cur.empty() ? 1 : (int) std::count(cur.begin(), cur.end(), ',') + 1;
        std::string rep;
        for (int i = 0; i < n; ++i) { if (i) rep += ","; rep += sval; }
        sval = rep;
    }
    cfg->set_deserialize_strict(d->key, sval);
    col->update_dirty();
    plater->on_config_change(*cfg);
}
static py::object settings_get(const std::string &name)
{
    const SettingDef *d = find_setting(name);
    if (!d) throw std::runtime_error("unknown setting: " + name);
    DynamicPrintConfig *cfg = setting_cfg(d->key, nullptr);
    if (!cfg || !cfg->has(d->key)) return py::none();
    std::string raw = cfg->opt_serialize(d->key);
    if (d->is_vector) { auto pos = raw.find(','); if (pos != std::string::npos) raw = raw.substr(0, pos); }
    try {
        if (d->kind == 'b') return py::bool_(raw == "1" || raw == "true");
        if (d->kind == 'e') { for (const auto &p : d->emap) if (p.second == raw) return py::str(p.first); return py::str(raw); }
        if (d->kind == 'i') return py::int_((long long) std::llround(std::stod(raw)));
        if (d->kind == 'f') return py::float_(std::stod(raw));
        if (d->kind == 'p') { std::string r = raw; if (!r.empty() && r.back() == '%') r.pop_back(); return py::float_(std::stod(r)); }
    } catch (...) { /* fall through to string */ }
    return py::str(raw);   // 'o' (float-or-percent) and any unparsable value
}


// ---- per-slot filament configuration ---------------------------------------
// Resolve the config carrying a per-filament key: filament preset config, else
// project_config (colour lives there on Bambu/Orca).
static DynamicPrintConfig *filament_cfg(const std::string &key)
{
    PresetBundle *pb = GUI::wxGetApp().preset_bundle;
    DynamicPrintConfig &fc = pb->filaments.get_edited_preset().config;
    if (fc.has(key)) return &fc;
    if (pb->project_config.has(key)) return &pb->project_config;
    return nullptr;
}
static py::object filament_get(size_t slot, const std::string &key)
{
    DynamicPrintConfig *cfg = filament_cfg(key);
    if (!cfg) return py::none();
    const ConfigOption *o = cfg->option(key);
    if (!o) return py::none();
    // Cast to the ConfigOptionVector<T> template base so this also matches the
    // *Nullable variants (Bambu/Orca use ConfigOptionIntsNullable etc.).
    if (auto *ov = dynamic_cast<const ConfigOptionVector<std::string>*>(o))
        return slot < ov->values.size() ? py::cast(ov->values[slot]) : py::none();
    if (auto *ov = dynamic_cast<const ConfigOptionVector<int>*>(o))
        return slot < ov->values.size() ? py::cast(ov->values[slot]) : py::none();
    if (auto *ov = dynamic_cast<const ConfigOptionVector<double>*>(o))
        return slot < ov->values.size() ? py::cast(ov->values[slot]) : py::none();
    if (auto *ov = dynamic_cast<const ConfigOptionVector<unsigned char>*>(o))
        return slot < ov->values.size() ? py::cast((bool) ov->values[slot]) : py::none();
    return py::cast(o->serialize());
}
static void filament_set(size_t slot, const std::string &key, py::object val)
{
    main_thread("Filament.set");
    auto *plater = plater_or_throw("Filament.set");
    PresetBundle *pb = GUI::wxGetApp().preset_bundle;
    DynamicPrintConfig *cfg = filament_cfg(key);
    if (!cfg) throw std::runtime_error("no filament config carries key: " + key);
    ConfigOption *o = cfg->option(key, true);
    if (auto *ov = dynamic_cast<ConfigOptionVector<std::string>*>(o)) {
        if (slot >= ov->values.size()) ov->values.resize(slot + 1);
        ov->values[slot] = val.cast<std::string>();
    } else if (auto *ov = dynamic_cast<ConfigOptionVector<int>*>(o)) {
        if (slot >= ov->values.size()) ov->values.resize(slot + 1);
        ov->values[slot] = (int) std::llround(val.cast<double>());
    } else if (auto *ov = dynamic_cast<ConfigOptionVector<double>*>(o)) {
        if (slot >= ov->values.size()) ov->values.resize(slot + 1);
        ov->values[slot] = val.cast<double>();
    } else if (auto *ov = dynamic_cast<ConfigOptionVector<unsigned char>*>(o)) {
        if (slot >= ov->values.size()) ov->values.resize(slot + 1);
        ov->values[slot] = val.cast<bool>() ? 1 : 0;
    } else {
        throw std::runtime_error("unsupported filament option type for key: " + key);
    }
    if (cfg == &pb->filaments.get_edited_preset().config) pb->filaments.update_dirty();
    plater->on_config_change(pb->full_config());
}


// ---- export current state as a pyslic3r script ------------------------------
static std::string py_repr(py::object v)
{
    return py::str(py::repr(v)).cast<std::string>();
}
static std::string build_state_script()
{
    PresetBundle *pb = GUI::wxGetApp().preset_bundle;
    std::string out = "import pyslic3r\ndoc = pyslic3r.app.active_document\n";

    struct Src { ConfigSource src; const char *cfgname; PresetCollection *col; };
    Src srcs[] = {
        {ConfigSource::Printer,  "printer_config",  &pb->printers},
        {ConfigSource::Print,    "print_config",    &pb->prints},
        {ConfigSource::Filament, "filament_config", &pb->filaments},
    };

    out += "\n# presets\n";
    for (auto &sc : srcs)
        out += std::string("doc.") + sc.cfgname + ".apply_preset("
             + py_repr(py::cast(sc.col->get_selected_preset_name())) + ")\n";

    // filaments (colours) when multi-colour
    int nfil = (int)(pb->printers.get_edited_preset().config.option<ConfigOptionFloats>("nozzle_diameter") ? pb->printers.get_edited_preset().config.option<ConfigOptionFloats>("nozzle_diameter")->size() : 1);
    if (nfil > 1) {
        out += "\n# filaments\n";
        std::string colors = "[";
        for (int i = 0; i < nfil; ++i) {
            py::object c = filament_get((size_t) i, "filament_colour");
            std::string cs = c.is_none() ? std::string("#FFFFFF") : c.cast<std::string>();
            colors += (i ? ", " : "") + py_repr(py::cast(cs));
        }
        colors += "]";
        out += "doc.set_filaments(" + colors + ")\n";
    }

    // gather dirty keys per source
    std::map<std::string, const char*> dirty;             // key -> cfgname
    std::map<std::string, PresetCollection*> dirty_col;   // key -> collection
    for (auto &sc : srcs)
        for (const std::string &k : sc.col->current_dirty_options()) {
            dirty[k] = sc.cfgname;
            dirty_col[k] = sc.col;
        }

    out += "\n# settings (changed from the selected presets)\n";
    std::set<std::string> handled;
    for (const auto &d : settings_defs()) {
        if (dirty.count(d.key) && !handled.count(d.key)) {
            py::object v = settings_get(d.name);
            if (!v.is_none()) {
                out += "doc.settings.set(" + py_repr(py::cast(d.name)) + ", " + py_repr(v) + ")\n";
                handled.insert(d.key);
            }
        }
    }
    // raw leftovers (dirty keys with no curated name)
    bool any_raw = false;
    for (const auto &kv : dirty) {
        if (handled.count(kv.first)) continue;
        if (!any_raw) { out += "\n# other changed settings (raw keys)\n"; any_raw = true; }
        DynamicPrintConfig &cfg = dirty_col[kv.first]->get_edited_preset().config;
        out += std::string("doc.") + kv.second + ".set("
             + py_repr(py::cast(kv.first)) + ", "
             + py_repr(py::cast(cfg.opt_serialize(kv.first))) + ")\n";
    }
    return out;
}

// World-space Z height of an object (fork-agnostic; avoids bounding_box vs
// bounding_box_approx drift) for slicing-parameter derivation.
static double object_max_z(const ModelObject *obj)
{
    double zmax = 0.0;
    for (const ModelVolume *v : obj->volumes) {
        if (!v->is_model_part()) continue;
        Transform3d tr = v->get_matrix();
        if (!obj->instances.empty()) tr = obj->instances[0]->get_matrix() * tr;
        for (const auto &vert : v->mesh().its.vertices) {
            const double z = (tr * vert.cast<double>()).z();
            if (z > zmax) zmax = z;
        }
    }
    return zmax;
}

// Merged world-space mesh of an object's model-part volumes (instance[0] frame)
// — the geometry a user sees on the plate. Used as boolean operands.
static TriangleMesh object_world_mesh(const ModelObject *obj)
{
    TriangleMesh out;
    const Transform3d inst = obj->instances.empty() ? Transform3d::Identity()
                                                     : obj->instances[0]->get_matrix();
    for (const ModelVolume *v : obj->volumes) {
        if (!v->is_model_part()) continue;
        TriangleMesh m(v->mesh());
        m.transform(inst * v->get_matrix(), true);
        out.merge(m);
    }
    return out;
}

// Extrude an SVG file into a solid mesh (the SVG-emboss gizmo core: NSVG parse
// -> union of 2D shapes -> Emboss::polygons2model with a depth projection).
static TriangleMesh svg_to_mesh(const std::string &path, double depth_mm)
{
    if (depth_mm <= 0.0) throw std::runtime_error("emboss_svg: depth must be > 0");
    NSVGimage_ptr img = nsvgParseFromFile(path, "mm", 96.0f);
    if (!img) throw std::runtime_error("emboss_svg: could not parse SVG: " + path);
    const double tol_mm = 0.1;   // curve tessellation tolerance (as the SVG gizmo uses)
    NSVGLineParams params((tol_mm * tol_mm) / (SCALING_FACTOR * SCALING_FACTOR));
    ExPolygonsWithIds ids = create_shape_with_ids(*img, params);
    ExPolygons shapes;
    for (const auto &sh : ids) if (!sh.expoly.empty()) expolygons_append(shapes, sh.expoly);
    shapes = union_ex(shapes);
    if (shapes.empty()) throw std::runtime_error("emboss_svg: no usable shapes in " + path);
    const double scale = SCALING_FACTOR;              // EmbossShape default scale
    const double depth = depth_mm / scale;            // SHAPE_SCALE applied in ProjectZ
    auto projectZ = std::make_unique<Emboss::ProjectZ>(depth);
    Transform3d tr = Eigen::Translation<double, 3>(0., 0., 0.) * Eigen::Scaling(scale);
    Emboss::ProjectTransform project(std::move(projectZ), tr);
    indexed_triangle_set its = Emboss::polygons2model(shapes, project);
    if (its.indices.empty()) throw std::runtime_error("emboss_svg: produced an empty mesh");
    return TriangleMesh(std::move(its));
}

} // anonymous namespace

// ---------------------------------------------------------------------------

void register_object_model(py::module_ &m)
{
    // ---- Config -----------------------------------------------------------
    py::class_<PyConfig>(m, "Config")
        .def("has", [](const PyConfig &c, const std::string &key) {
            return resolve_config(c, "Config.has")->has(key);
        })
        .def("get", [](const PyConfig &c, const std::string &key) -> py::object {
            const ConfigBase *cfg = resolve_config(c, "Config.get");
            if (!cfg->has(key))
                return py::none();
            return py::str(cfg->opt_serialize(key));   // serialized string form
        })
        .def("keys", [](const PyConfig &c) {
            return resolve_config(c, "Config.keys")->keys();   // -> list[str]
        })
        .def("dirty_keys", [](const PyConfig &c) {
            main_thread("Config.dirty_keys");
            PresetCollection *col = preset_collection(c.source);
            py::list out;
            if (col != nullptr) for (const std::string &k : col->current_dirty_options()) out.append(k);
            return out;
        })
        .def_property_readonly("is_dirty", [](const PyConfig &c) {
            PresetCollection *col = preset_collection(c.source);
            if (col == nullptr) return false;   // global/plate: not preset-dirty-tracked
            main_thread("Config.is_dirty");
            return col->current_is_dirty();
        })
        // ---- M2 mutation --------------------------------------------------
        .def("set", [](const PyConfig &c, const std::string &key, const std::string &value) {
            // GUI-parity: edit the working config the app watches, mark dirty,
            // and let Plater invalidate slicing exactly as an on-screen edit.
            if (c.source == ConfigSource::Global)
                throw std::runtime_error(
                    "the global config is derived and read-only; set on "
                    "print_config / filament_config / printer_config");
            if (c.source == ConfigSource::Plate)
                throw std::runtime_error(
                    "per-plate config is not supported on this platform (single bed); "
                    "use print_config");
            auto *plater = plater_or_throw("Config.set");
            if (c.source == ConfigSource::Volume) {
                ModelObject *obj = object_at(c.plate_idx, "Config.set");
                if (size_t(c.vol_idx) >= obj->volumes.size())
                    throw std::runtime_error("volume index out of range");
                ConfigSubstitutionContext ctx(ForwardCompatibilitySubstitutionRule::EnableSilent);
                obj->volumes[c.vol_idx]->config.set_deserialize(key, value, ctx);
                plater->changed_object(int(c.plate_idx));
                return;
            }
            if (c.source == ConfigSource::Object) {
                ModelObject *obj = object_at(c.plate_idx, "Config.set");
                ConfigSubstitutionContext ctx(ForwardCompatibilitySubstitutionRule::EnableSilent);
                obj->config.set_deserialize(key, value, ctx);
                plater->changed_object(int(c.plate_idx));
                return;
            }
            PresetCollection *col = preset_collection(c.source);
            DynamicPrintConfig &cfg = col->get_edited_preset().config;
            if (!cfg.has(key))
                throw std::runtime_error("unknown config key for this preset: " + key);
            cfg.set_deserialize_strict(key, value);
            col->update_dirty();                 // mark preset dirty like the GUI
            plater->on_config_change(cfg);        // diff + schedule reslice
        }, py::arg("key"), py::arg("value"))
        .def("presets", [](const PyConfig &c) {
            main_thread("Config.presets");
            PresetCollection *col = preset_collection(c.source);
            if (col == nullptr)
                throw std::runtime_error("presets() only on print/filament/printer config");
            py::list out;
            for (const Preset &p : col->get_presets())
                if (p.is_visible && p.is_compatible) out.append(p.name);   // selectable set
            return out;
        })
        .def_property_readonly("selected_preset", [](const PyConfig &c) {
            main_thread("Config.selected_preset");
            PresetCollection *col = preset_collection(c.source);
            if (col == nullptr)
                throw std::runtime_error("selected_preset only on print/filament/printer config");
            return col->get_selected_preset_name();
        })
        .def("save_preset", [](const PyConfig &c, const std::string &name) {
            main_thread("Config.save_preset");
            if (name.empty()) throw std::runtime_error("save_preset: empty name");
            PresetCollection *col = preset_collection(c.source);
            if (col == nullptr)
                throw std::runtime_error("save_preset only on print/filament/printer config");
            col->save_current_preset(name);   // current edited config -> named user preset
            return name;
        }, py::arg("name"))
        .def("delete_preset", [](const PyConfig &c, const std::string &name) {
            main_thread("Config.delete_preset");
            auto *plater = plater_or_throw("Config.delete_preset");
            PresetCollection *col = preset_collection(c.source);
            if (col == nullptr)
                throw std::runtime_error("delete_preset only on print/filament/printer config");
            col->select_preset_by_name(name, true);   // force-select the target
            if (col->get_selected_preset_name() != name)
                throw std::runtime_error("delete_preset: preset not found: " + name);
            if (!col->delete_current_preset())
                throw std::runtime_error("delete_preset: cannot delete (default/system/in-use): " + name);
            plater->on_config_change(GUI::wxGetApp().preset_bundle->full_config());
            return name;
        }, py::arg("name"))
        .def("apply_preset", [](const PyConfig &c, const std::string &name, bool force) {
            // Headless-safe: select via the PresetBundle directly (the Tab path
            // SIGSEGVs offscreen), resolve compatible print/filament, push to Plater.
            if (preset_type(c.source) == Preset::TYPE_INVALID)
                throw std::runtime_error("apply_preset only on print/filament/printer config");
            main_thread("Config.apply_preset");
            auto *plater = plater_or_throw("Config.apply_preset");
            PresetCollection *col = preset_collection(c.source);
            if (col == nullptr) throw std::runtime_error("no preset collection for this config");
            col->select_preset_by_name(name, force);
            PresetBundle *pb = GUI::wxGetApp().preset_bundle;
            pb->update_compatible(PresetSelectCompatibleType::Always);
            col->update_dirty();
            plater->on_config_change(pb->full_config());
            // Verify the END STATE, not select_preset_by_name's return: it returns
            // false when the preset is already selected, and update_compatible reverts
            // an incompatible one — so the real test is whether the target is selected.
            if (!force && col->get_selected_preset_name() != name)
                throw std::runtime_error("preset not applied (not found or incompatible): " + name);
        }, py::arg("name"), py::arg("force") = false);

    // ---- Volume -----------------------------------------------------------
    py::class_<PyVolume>(m, "Volume")
        .def_property_readonly("name", [](const PyVolume &v) {
            return volume_at(v, "Volume.name")->name;
        })
        .def_property_readonly("type", [](const PyVolume &v) {
            return std::string(volume_type_str(volume_at(v, "Volume.type")->type()));
        })
        .def_property_readonly("is_model_part", [](const PyVolume &v) {
            return volume_at(v, "Volume.is_model_part")->is_model_part();
        })
        .def("set_type", [](const PyVolume &v, const std::string &type_name) {
            main_thread("Volume.set_type");
            auto *plater = plater_or_throw("Volume.set_type");
            ModelVolume *vol = volume_at(v, "Volume.set_type");
            ModelObject *obj = object_at(v.obj_idx, "Volume.set_type");
            const ModelVolumeType type = volume_type_from_str(type_name);
            // mirror the GUI: keep at least one model part on the object
            if (vol->is_model_part() && type != ModelVolumeType::MODEL_PART) {
                int parts = 0;
                for (const ModelVolume *ov : obj->volumes) if (ov->is_model_part()) ++parts;
                if (parts <= 1)
                    throw std::runtime_error("set_type: cannot change the object's only model part");
            }
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: change part type"));
            vol->set_type(type);
            obj->invalidate_bounding_box();
            plater->changed_object(int(v.obj_idx));
            return std::string(volume_type_str(vol->type()));
        }, py::arg("type"))
        // ---- part transforms (UI-parity: reposition a part/modifier/negative volume
        //      within its object, the way dragging it in the GUI does) --------------
        .def("translate", [](const PyVolume &v, double dx, double dy, double dz) {
            main_thread("Volume.translate");
            auto *plater = plater_or_throw("Volume.translate");
            ModelVolume *vol = volume_at(v, "Volume.translate");
            ModelObject *obj = object_at(v.obj_idx, "Volume.translate");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: move part"));
            vol->translate(Vec3d(dx, dy, dz));
            obj->invalidate_bounding_box();
            plater->changed_object(int(v.obj_idx));
        }, py::arg("dx"), py::arg("dy"), py::arg("dz"))
        .def("rotate", [](const PyVolume &v, double rx, double ry, double rz) {
            main_thread("Volume.rotate");
            auto *plater = plater_or_throw("Volume.rotate");
            ModelVolume *vol = volume_at(v, "Volume.rotate");
            ModelObject *obj = object_at(v.obj_idx, "Volume.rotate");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: rotate part"));
            if (rx != 0.0) vol->rotate(rx * M_PI / 180.0, X);
            if (ry != 0.0) vol->rotate(ry * M_PI / 180.0, Y);
            if (rz != 0.0) vol->rotate(rz * M_PI / 180.0, Z);
            obj->invalidate_bounding_box();
            plater->changed_object(int(v.obj_idx));
        }, py::arg("rx") = 0.0, py::arg("ry") = 0.0, py::arg("rz") = 0.0)
        .def("scale", [](const PyVolume &v, double x, py::object y, py::object z) {
            const double sx = x;
            const double sy = y.is_none() ? x : y.cast<double>();
            const double sz = z.is_none() ? x : z.cast<double>();
            if (sx <= 0.0 || sy <= 0.0 || sz <= 0.0)
                throw std::runtime_error("scale factors must be > 0");
            main_thread("Volume.scale");
            auto *plater = plater_or_throw("Volume.scale");
            ModelVolume *vol = volume_at(v, "Volume.scale");
            ModelObject *obj = object_at(v.obj_idx, "Volume.scale");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: scale part"));
            vol->scale(Vec3d(sx, sy, sz));
            obj->invalidate_bounding_box();
            plater->changed_object(int(v.obj_idx));
        }, py::arg("x"), py::arg("y") = py::none(), py::arg("z") = py::none())
        .def("mirror", [](const PyVolume &v, const std::string &axis) {
            main_thread("Volume.mirror");
            Axis a;
            if      (axis == "x" || axis == "X") a = X;
            else if (axis == "y" || axis == "Y") a = Y;
            else if (axis == "z" || axis == "Z") a = Z;
            else throw std::runtime_error("mirror: axis must be x, y or z");
            auto *plater = plater_or_throw("Volume.mirror");
            ModelVolume *vol = volume_at(v, "Volume.mirror");
            ModelObject *obj = object_at(v.obj_idx, "Volume.mirror");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: mirror part"));
            vol->mirror(a);
            obj->invalidate_bounding_box();
            plater->changed_object(int(v.obj_idx));
        }, py::arg("axis"))
        .def("offset", [](const PyVolume &v) {
            return vec3(volume_at(v, "Volume.offset")->get_transformation().get_offset());
        })
        .def("bounding_box", [](const PyVolume &v) {
            const ModelVolume *vol = volume_at(v, "Volume.bounding_box");
            const Transform3d m = vol->get_matrix();
            BoundingBoxf3 bb;
            for (const auto &p : vol->mesh().its.vertices)
                bb.merge(Vec3d(m * p.cast<double>()));
            py::dict d;
            d["min"]    = vec3(bb.min);
            d["max"]    = vec3(bb.max);
            d["size"]   = vec3(bb.size());
            d["center"] = vec3(bb.center());
            return d;
        })
        .def_property_readonly("is_mm_painted", [](const PyVolume &v) {
            return volume_at(v, "Volume.is_mm_painted")->is_mm_painted();
        })
        .def_property_readonly("is_support_painted", [](const PyVolume &v) {
            return volume_at(v, "Volume.is_support_painted")->is_fdm_support_painted();
        })
        .def_property_readonly("is_seam_painted", [](const PyVolume &v) {
            return volume_at(v, "Volume.is_seam_painted")->is_seam_painted();
        })
        .def("paint_mmu_above", [](const PyVolume &v, double z, int extruder) {
            main_thread("Volume.paint_mmu_above");
            return mmu_paint_band(v, z, std::numeric_limits<double>::max(), extruder);
        }, py::arg("z"), py::arg("extruder"))
        .def("paint_mmu_band", [](const PyVolume &v, double z_min, double z_max, int extruder) {
            main_thread("Volume.paint_mmu_band");
            if (z_max <= z_min) throw std::runtime_error("paint_mmu_band: z_max must be > z_min");
            return mmu_paint_band(v, z_min, z_max, extruder);
        }, py::arg("z_min"), py::arg("z_max"), py::arg("extruder"))
        .def("clear_mmu_paint", [](const PyVolume &v) {
            main_thread("Volume.clear_mmu_paint");
            volume_at(v, "Volume.clear_mmu_paint")->mm_segmentation_facets.reset();
        })
        .def("paint_support_above", [](const PyVolume &v, double z, const std::string &mode) {
            main_thread("Volume.paint_support_above");
            return paint_ann_band(v, z, std::numeric_limits<double>::max(), eb_mode(mode), &ModelVolume::supported_facets);
        }, py::arg("z"), py::arg("mode") = "enforce")
        .def("paint_support_band", [](const PyVolume &v, double z_min, double z_max, const std::string &mode) {
            main_thread("Volume.paint_support_band");
            if (z_max <= z_min) throw std::runtime_error("paint_support_band: z_max must be > z_min");
            return paint_ann_band(v, z_min, z_max, eb_mode(mode), &ModelVolume::supported_facets);
        }, py::arg("z_min"), py::arg("z_max"), py::arg("mode") = "enforce")
        .def("clear_support_paint", [](const PyVolume &v) {
            main_thread("Volume.clear_support_paint");
            volume_at(v, "Volume.clear_support_paint")->supported_facets.reset();
        })
        .def("paint_seam_above", [](const PyVolume &v, double z, const std::string &mode) {
            main_thread("Volume.paint_seam_above");
            return paint_ann_band(v, z, std::numeric_limits<double>::max(), eb_mode(mode), &ModelVolume::seam_facets);
        }, py::arg("z"), py::arg("mode") = "enforce")
        .def("paint_seam_band", [](const PyVolume &v, double z_min, double z_max, const std::string &mode) {
            main_thread("Volume.paint_seam_band");
            if (z_max <= z_min) throw std::runtime_error("paint_seam_band: z_max must be > z_min");
            return paint_ann_band(v, z_min, z_max, eb_mode(mode), &ModelVolume::seam_facets);
        }, py::arg("z_min"), py::arg("z_max"), py::arg("mode") = "enforce")
        .def("clear_seam_paint", [](const PyVolume &v) {
            main_thread("Volume.clear_seam_paint");
            volume_at(v, "Volume.clear_seam_paint")->seam_facets.reset();
        })
        .def("paint_fuzzy_above", [](const PyVolume &v, double z, const std::string &mode) {
            main_thread("Volume.paint_fuzzy_above");
            return paint_ann_band(v, z, std::numeric_limits<double>::max(), eb_mode(mode), &ModelVolume::fuzzy_skin_facets);
        }, py::arg("z"), py::arg("mode") = "enforce")
        .def("paint_fuzzy_band", [](const PyVolume &v, double z_min, double z_max, const std::string &mode) {
            main_thread("Volume.paint_fuzzy_band");
            if (z_max <= z_min) throw std::runtime_error("paint_fuzzy_band: z_max must be > z_min");
            return paint_ann_band(v, z_min, z_max, eb_mode(mode), &ModelVolume::fuzzy_skin_facets);
        }, py::arg("z_min"), py::arg("z_max"), py::arg("mode") = "enforce")
        .def("clear_fuzzy_paint", [](const PyVolume &v) {
            main_thread("Volume.clear_fuzzy_paint");
            volume_at(v, "Volume.clear_fuzzy_paint")->fuzzy_skin_facets.reset();
        })
        .def_property_readonly("is_fuzzy_painted", [](const PyVolume &v) {
            return !volume_at(v, "Volume.is_fuzzy_painted")->fuzzy_skin_facets.empty();
        })
        .def_property_readonly("config", [](const PyVolume &v) {
            (void) volume_at(v, "Volume.config");   // bounds-check
            return PyConfig{ConfigSource::Volume, int(v.obj_idx), int(v.vol_idx)};
        });

    // ---- Text (editable emboss text; Prusa text_configuration + FreeType) ----
    py::class_<PyText>(m, "Text")
        .def_property_readonly("text", [](const PyText &t) {
            return text_at(t, "Text.text")->text_configuration->text;
        })
        .def_property_readonly("font_name", [](const PyText &t) {
            const EmbossStyle &st = text_at(t, "Text.font_name")->text_configuration->style;
            return st.prop.face_name.has_value() ? *st.prop.face_name : st.name;
        })
        .def_property_readonly("font_size", [](const PyText &t) {
            return text_at(t, "Text.font_size")->text_configuration->style.prop.size_in_mm;
        })
        .def_property_readonly("width", [](const PyText &t) {
            ModelVolume *v = text_at(t, "Text.width");
            return _prusa_text_width(v, v->text_configuration->text,
                                     v->text_configuration->style.prop,
                                     v->emboss_shape->scale);
        })
        .def("set_text", [](const PyText &t, const std::string &new_text,
                            bool fit, py::object max_width) -> py::object {
            main_thread("Text.set_text");
            ModelObject *obj = object_at(t.obj_idx, "Text.set_text");
            ModelVolume *vol = text_at(t, "Text.set_text");
            const FontProp &fp = vol->text_configuration->style.prop;
            double base_scale = vol->emboss_shape->scale;   // carries the size
            double base_size  = fp.size_in_mm;              // metadata only

            double target_w = max_width.is_none()
                ? _prusa_text_width(vol, vol->text_configuration->text, fp, base_scale)
                : max_width.cast<double>();

            double scale = base_scale;
            TriangleMesh mesh = _prusa_text_mesh(vol, new_text, fp, scale);
            double w = mesh.bounding_box().size().x();
            bool fitted = false;
            if (fit && w > target_w && w > 0.0) {
                scale *= target_w / w;                       // resize via scale, not size_in_mm
                mesh = _prusa_text_mesh(vol, new_text, fp, scale);
                w = mesh.bounding_box().size().x();
                fitted = true;
            }
            if (mesh.empty())
                throw std::runtime_error("Text.set_text: mesh generation failed");
            double ratio = scale / base_scale;               // size change factor
            double new_size = base_size * ratio;

            TextConfiguration tc = *vol->text_configuration;   // copy + update
            tc.text = new_text;
            tc.style.prop.size_in_mm = (float) new_size;
            EmbossShape es2 = *vol->emboss_shape;              // keep for re-editability
            es2.scale = scale;

            GUI::wxGetApp().plater()->take_snapshot(std::string("Edit Text"));
            // Re-align the regenerated mesh's Z base to the original so the text
            // keeps its exact surface placement (obj->volumes holds heap pointers,
            // so `vol` stays valid across add_volume until it is deleted below).
            double old_zmin = vol->mesh().bounding_box().min.z();
            mesh.translate(0.f, 0.f, (float)(old_zmin - mesh.bounding_box().min.z()));
            Geometry::Transformation tran = vol->get_transformation();
            ModelVolume *nv = obj->add_volume(std::move(mesh), vol->type());
            nv->calculate_convex_hull();
            nv->set_transformation(tran);
            nv->text_configuration = tc;
            nv->emboss_shape = es2;
            nv->name = vol->name;
            nv->config.apply(vol->config);
            std::swap(obj->volumes[t.vol_idx], obj->volumes.back());
            obj->delete_volume(obj->volumes.size() - 1);
            obj->invalidate_bounding_box();
            GUI::wxGetApp().plater()->changed_object(int(t.obj_idx));

            py::dict d;
            d["text"] = new_text; d["font_size"] = new_size; d["width"] = w;
            d["fit_target"] = target_w; d["fitted"] = fitted;
            return d;
        }, py::arg("text"), py::arg("fit") = true, py::arg("max_width") = py::none());

    // ---- Object -----------------------------------------------------------
    py::class_<PyObject>(m, "Object")
        .def_property_readonly("name", [](const PyObject &o) {
            return object_at(o.idx, "Object.name")->name;
        })
        .def_property_readonly("index", [](const PyObject &o) { return o.idx; })
        .def_property_readonly("instance_count", [](const PyObject &o) {
            return object_at(o.idx, "Object.instance_count")->instances.size();
        })
        .def_property_readonly("volumes", [](const PyObject &o) {
            ModelObject *obj = object_at(o.idx, "Object.volumes");
            py::list out;
            for (size_t i = 0; i < obj->volumes.size(); ++i)
                out.append(PyVolume{o.idx, i});
            return out;
        })
        .def("texts", [](const PyObject &o) {
            ModelObject *obj = object_at(o.idx, "Object.texts");
            py::list out;
            for (size_t i = 0; i < obj->volumes.size(); ++i)
                if (obj->volumes[i]->is_text())
                    out.append(PyText{o.idx, i});
            return out;
        })
        .def_property_readonly("config", [](const PyObject &o) {
            (void) object_at(o.idx, "Object.config");   // bounds-check
            return PyConfig{ConfigSource::Object, int(o.idx)};
        })
        .def("bounding_box", [](const PyObject &o) {
            const BoundingBoxf3 &bb = object_at(o.idx, "Object.bounding_box")->bounding_box_approx();
            py::dict d;
            d["min"]    = vec3(bb.min);
            d["max"]    = vec3(bb.max);
            d["size"]   = vec3(bb.size());
            d["center"] = vec3(bb.center());
            return d;
        })
        // ---- M2 mutation (UI-parity: snapshot + Plater refresh) -----------
        .def("translate", [](const PyObject &o, double dx, double dy, double dz) {
            auto *plater = plater_or_throw("Object.translate");
            ModelObject *obj = object_at(o.idx, "Object.translate");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: move object"));
            obj->translate_instances(Vec3d(dx, dy, dz));  // moves the placed instances
            obj->invalidate_bounding_box();
            plater->changed_object(int(o.idx));           // same refresh the GUI uses
        }, py::arg("dx"), py::arg("dy"), py::arg("dz"))
        .def("duplicate", [](const PyObject &o) {
            main_thread("Object.duplicate");
            auto *plater = plater_or_throw("Object.duplicate");
            Model &model = model_or_throw("Object.duplicate");
            ModelObject *src = object_at(o.idx, "Object.duplicate");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: duplicate object"));
            // Deep copy: geometry, volumes, per-object/volume config, all paint, instances.
            ModelObject *dup = model.add_object(*src);
            const size_t new_idx = model.objects.size() - 1;
            // Nudge the copy +X so it does not overlap the source.
            const double gap = src->bounding_box_approx().size().x() + 5.0;
            for (ModelInstance *inst : dup->instances)
                inst->set_offset(inst->get_offset() + Vec3d(gap, 0.0, 0.0));
            dup->invalidate_bounding_box();
            // Register like the GUI does (also assigns a plate on Bambu/Orca).
            GUI::wxGetApp().obj_list()->add_object_to_list(new_idx);
            plater->canvas3D()->update_instance_printable_state_for_object(new_idx);
            plater->changed_object(int(new_idx));
            return PyObject{ new_idx };
        })
        .def("delete", [](const PyObject &o) {
            auto *plater = plater_or_throw("Object.delete");
            (void) object_at(o.idx, "Object.delete");     // bounds-check
            plater->delete_object_from_model(o.idx);      // snapshots internally
        })
        // ---- transforms (UI-parity: the rotate / scale / mirror gizmos and
        //      the "set number of instances" action) --------------------------
        .def("rotate", [](const PyObject &o, double rx, double ry, double rz) {
            auto *plater = plater_or_throw("Object.rotate");
            ModelObject *obj = object_at(o.idx, "Object.rotate");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: rotate object"));
            if (rx != 0.0) obj->rotate(rx * M_PI / 180.0, X);
            if (ry != 0.0) obj->rotate(ry * M_PI / 180.0, Y);
            if (rz != 0.0) obj->rotate(rz * M_PI / 180.0, Z);
            obj->invalidate_bounding_box();
            plater->changed_object(int(o.idx));
        }, py::arg("rx") = 0.0, py::arg("ry") = 0.0, py::arg("rz") = 0.0)
        .def("scale", [](const PyObject &o, double x, py::object y, py::object z) {
            const double sx = x;
            const double sy = y.is_none() ? x : y.cast<double>();
            const double sz = z.is_none() ? x : z.cast<double>();
            if (sx <= 0.0 || sy <= 0.0 || sz <= 0.0)
                throw std::runtime_error("scale factors must be > 0");
            auto *plater = plater_or_throw("Object.scale");
            ModelObject *obj = object_at(o.idx, "Object.scale");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: scale object"));
            obj->scale(Vec3d(sx, sy, sz));
            obj->invalidate_bounding_box();
            plater->changed_object(int(o.idx));
        }, py::arg("x"), py::arg("y") = py::none(), py::arg("z") = py::none())
        .def("mirror", [](const PyObject &o, const std::string &axis) {
            Axis a;
            if      (axis == "x" || axis == "X") a = X;
            else if (axis == "y" || axis == "Y") a = Y;
            else if (axis == "z" || axis == "Z") a = Z;
            else throw std::runtime_error("axis must be 'x', 'y', or 'z'");
            auto *plater = plater_or_throw("Object.mirror");
            ModelObject *obj = object_at(o.idx, "Object.mirror");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: mirror object"));
            obj->mirror(a);
            obj->invalidate_bounding_box();
            plater->changed_object(int(o.idx));
        }, py::arg("axis"))
        .def("set_instances", [](const PyObject &o, int n) {
            if (n < 1) throw std::runtime_error("instance count must be >= 1");
            auto *plater = plater_or_throw("Object.set_instances");
            ModelObject *obj = object_at(o.idx, "Object.set_instances");
            const int cur = int(obj->instances.size());
            if (n > cur)      plater->increase_instances(size_t(n - cur), int(o.idx));
            else if (n < cur) plater->decrease_instances(size_t(cur - n), int(o.idx));
        }, py::arg("n"))
        .def("transform", [](const PyObject &o) {
            // Read-back of the placed instance's transform (bidirectional).
            ModelObject *obj = object_at(o.idx, "Object.transform");
            const BoundingBoxf3 bb = obj->bounding_box_approx();
            py::dict d;
            d["size"]   = vec3(bb.size());
            d["center"] = vec3(bb.center());
            if (!obj->instances.empty())
                d["position"] = vec3(obj->instances.front()->get_offset());
            d["instances"] = obj->instances.size();
            return d;
        })
        // ---- geometry finish (UI-parity: drop-to-bed, scale-to-fit, rename) --
        .def("split", [](const PyObject &o) {
            main_thread("Object.split");
            _select_only(o.idx, "Object.split");
            plater_or_throw("Object.split")->split_object();
            return model_or_throw("Object.split").objects.size();
        })
        .def("cut", [](const PyObject &o, double z, bool keep_upper, bool keep_lower) {
            main_thread("Object.cut");
            if (!keep_upper && !keep_lower)
                throw std::runtime_error("cut: keep_upper and keep_lower are both false");
            auto *plater = plater_or_throw("Object.cut");
            ModelObject *obj = object_at(o.idx, "Object.cut");
            if (obj->instances.empty())
                throw std::runtime_error("cut: object has no instances");
            const Vec3d off = obj->instances[0]->get_offset();
            ModelObjectCutAttributes attrs =
                only_if(keep_upper, ModelObjectCutAttribute::KeepUpper) |
                only_if(keep_lower, ModelObjectCutAttribute::KeepLower);
            Transform3d cut_matrix = Geometry::translation_transform(Vec3d(0.0, 0.0, z - off.z()));
            Slic3r::Cut cut(obj, 0, cut_matrix, attrs);
            const ModelObjectPtrs &new_objects = cut.perform_with_plane();
            plater->apply_cut_object_to_model(o.idx, new_objects);
            return model_or_throw("Object.cut").objects.size();
        }, py::arg("z"), py::arg("keep_upper") = true, py::arg("keep_lower") = true)

        .def("cut_with_connectors", [](const PyObject &o, double z, py::iterable connectors,
                                       bool keep_upper, bool keep_lower) {
            main_thread("Object.cut_with_connectors");
            auto *plater = plater_or_throw("Object.cut_with_connectors");
            ModelObject *obj = object_at(o.idx, "Object.cut_with_connectors");
            if (obj->instances.empty())
                throw std::runtime_error("cut_with_connectors: object has no instances");
            const Vec3d off = obj->instances[0]->get_offset();
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: cut with connectors"));
            // Each connector: a NEGATIVE_VOLUME frustum straddling the cut plane, flagged
            // as a connector so the Cut engine turns it into a plug (lower) + hole (upper).
            int nconn = 0;
            for (py::handle item : connectors) {
                py::sequence c = item.cast<py::sequence>();   // (x, y, diameter, depth)
                const double cx = c[0].cast<double>();
                const double cy = c[1].cast<double>();
                const double diameter = c[2].cast<double>();
                const double depth = c[3].cast<double>();
                if (diameter <= 0.0 || depth <= 0.0)
                    throw std::runtime_error("cut_with_connectors: diameter and depth must be > 0");
                const double radius = diameter / 2.0;
                TriangleMesh mesh(its_make_cylinder(1.0, 1.0));   // unit; scaled below
                ModelVolume *nv = obj->add_volume(std::move(mesh), ModelVolumeType::NEGATIVE_VOLUME);
                const Vec3d local_pos(cx - off.x(), cy - off.y(), (z - off.z()) - depth / 2.0);
                nv->set_transformation(Geometry::translation_transform(local_pos) *
                                       Geometry::scale_transform(Vec3d(radius, radius, depth)));
                nv->cut_info = ModelVolume::CutInfo(CutConnectorType::Plug, 0.0f, 0.1f);
                nv->name = "connector-" + std::to_string(++nconn);
            }
            if (nconn == 0)
                throw std::runtime_error("cut_with_connectors: no connectors given");
            ModelObjectCutAttributes attrs =
                only_if(keep_upper, ModelObjectCutAttribute::KeepUpper) |
                only_if(keep_lower, ModelObjectCutAttribute::KeepLower) |
                only_if(true,       ModelObjectCutAttribute::CreateDowels);
            const Transform3d cut_matrix = Geometry::translation_transform(Vec3d(0.0, 0.0, z - off.z()));
            Slic3r::Cut cut(obj, 0, cut_matrix, attrs);
            const ModelObjectPtrs &new_objects = cut.perform_with_plane();
            plater->apply_cut_object_to_model(o.idx, new_objects);
            return model_or_throw("Object.cut_with_connectors").objects.size();
        }, py::arg("z"), py::arg("connectors"), py::arg("keep_upper") = true, py::arg("keep_lower") = true)
        .def("add_part", [](const PyObject &o, const std::string &path,
                            const std::string &type_name) {
            main_thread("Object.add_part");
            auto *plater = plater_or_throw("Object.add_part");
            ModelObject *obj = object_at(o.idx, "Object.add_part");
            const ModelVolumeType type = volume_type_from_str(type_name);
            TriangleMesh mesh;
            if (!mesh.ReadSTLFile(path.c_str()) || mesh.empty())
                throw std::runtime_error("add_part: could not read a mesh (STL expected): " + path);
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: add part"));
            ModelVolume *v = obj->add_volume(std::move(mesh), type);
            v->name = boost::filesystem::path(path).stem().string();
            obj->invalidate_bounding_box();
            plater->changed_object(int(o.idx));
            return int(obj->volumes.size() - 1);   // index of the new volume
        }, py::arg("path"), py::arg("type") = "part")
        .def("simplify", [](const PyObject &o, double ratio) {
            main_thread("Object.simplify");
            if (ratio <= 0.0 || ratio >= 1.0)
                throw std::runtime_error("simplify: ratio must be in (0, 1) — the fraction of triangles to keep");
            auto *plater = plater_or_throw("Object.simplify");
            ModelObject *obj = object_at(o.idx, "Object.simplify");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: simplify"));
            plater->clear_before_change_mesh(int(o.idx), std::string("simplify"));
            int total = 0;
            for (ModelVolume *mv : obj->volumes) {
                if (!mv->is_model_part()) continue;
                indexed_triangle_set its = mv->mesh().its;   // editable copy
                const uint32_t cur = (uint32_t) its.indices.size();
                uint32_t target = (uint32_t) std::max<double>(4.0, double(cur) * ratio);
                if (target < cur) {
                    its_quadric_edge_collapse(its, target, nullptr, nullptr, nullptr);
                    TriangleMesh tm(its);
                    mv->set_mesh(std::move(tm));
                    mv->calculate_convex_hull();
                    mv->set_new_unique_id();
                }
                total += (int) mv->mesh().its.indices.size();
            }
            obj->invalidate_bounding_box();
            obj->ensure_on_bed();
            plater->changed_mesh(int(o.idx));
            return total;
        }, py::arg("ratio") = 0.5)
        .def_property_readonly("triangle_count", [](const PyObject &o) {
            ModelObject *obj = object_at(o.idx, "Object.triangle_count");
            int total = 0;
            for (const ModelVolume *mv : obj->volumes)
                if (mv->is_model_part()) total += (int) mv->mesh().its.indices.size();
            return total;
        })
        .def("set_layer_height_profile", [](const PyObject &o, py::list pairs) {
            main_thread("Object.set_layer_height_profile");
            auto *plater = plater_or_throw("Object.set_layer_height_profile");
            ModelObject *obj = object_at(o.idx, "Object.set_layer_height_profile");
            std::vector<coordf_t> flat;
            for (auto item : pairs) {
                py::sequence pr = item.cast<py::sequence>();
                flat.push_back(pr[0].cast<double>());   // z
                flat.push_back(pr[1].cast<double>());   // layer height
            }
            obj->layer_height_profile.set(std::move(flat));
            plater->changed_object(int(o.idx));
            return (int)(obj->layer_height_profile.get().size() / 2);
        }, py::arg("pairs"))
        .def("layer_height_profile", [](const PyObject &o) {
            ModelObject *obj = object_at(o.idx, "Object.layer_height_profile");
            std::vector<coordf_t> flat = obj->layer_height_profile.get();
            py::list out;
            for (size_t i = 0; i + 1 < flat.size(); i += 2) out.append(py::make_tuple(flat[i], flat[i + 1]));
            return out;
        })
        .def("clear_layer_height_profile", [](const PyObject &o) {
            main_thread("Object.clear_layer_height_profile");
            auto *plater = plater_or_throw("Object.clear_layer_height_profile");
            object_at(o.idx, "Object.clear_layer_height_profile")->layer_height_profile.clear();
            plater->changed_object(int(o.idx));
        })
        .def("adaptive_layer_height", [](const PyObject &o, double quality) {
            main_thread("Object.adaptive_layer_height");
            if (quality < 0.0 || quality > 1.0) throw std::runtime_error("quality must be in [0, 1]");
            auto *plater = plater_or_throw("Object.adaptive_layer_height");
            ModelObject *obj = object_at(o.idx, "Object.adaptive_layer_height");
            const DynamicPrintConfig cfg = GUI::wxGetApp().preset_bundle->full_config();
            const double maxz = object_max_z(obj);
            SlicingParameters params = PrintObject::slicing_parameters(cfg, *obj, (float) maxz, Vec3d(1., 1., 1.));
            std::vector<double> prof = layer_height_profile_adaptive(params, *obj, (float) quality);
            PrintObject::update_layer_height_profile(*obj, params, prof);
            obj->layer_height_profile.set(std::move(prof));
            plater->changed_object(int(o.idx));
            return (int)(obj->layer_height_profile.get().size() / 2);
        }, py::arg("quality") = 0.5)
        .def("boolean", [](const PyObject &o, const PyObject &other, const std::string &op) {
            main_thread("Object.boolean");
            if (o.idx == other.idx) throw std::runtime_error("boolean: the other object must be different");
            auto *plater = plater_or_throw("Object.boolean");
            ModelObject *A = object_at(o.idx, "Object.boolean");
            ModelObject *B = object_at(other.idx, "Object.boolean");
            TriangleMesh meshA = object_world_mesh(A);
            TriangleMesh meshB = object_world_mesh(B);
            if (meshA.its.indices.empty() || meshB.its.indices.empty())
                throw std::runtime_error("boolean: both objects must have model-part geometry");
            if (op == "union" || op == "plus")
                MeshBoolean::cgal::plus(meshA, meshB);
            else if (op == "difference" || op == "minus")
                MeshBoolean::cgal::minus(meshA, meshB);
            else if (op == "intersection" || op == "intersect")
                MeshBoolean::cgal::intersect(meshA, meshB);
            else
                throw std::runtime_error("boolean: op must be union | difference | intersection");
            plater->clear_before_change_mesh(int(o.idx), std::string("boolean"));
            // Write the world result back into A's first model-part volume (local frame).
            ModelVolume *v0 = nullptr;
            for (ModelVolume *v : A->volumes) if (v->is_model_part()) { v0 = v; break; }
            if (v0 == nullptr) throw std::runtime_error("boolean: object A has no model-part volume");
            const Transform3d inst = A->instances.empty() ? Transform3d::Identity()
                                                          : A->instances[0]->get_matrix();
            const Transform3d full = inst * v0->get_matrix();
            meshA.transform(full.inverse(), true);
            v0->set_mesh(std::move(meshA));
            v0->calculate_convex_hull();
            v0->set_new_unique_id();
            // Drop A's now-redundant extra model-part volumes (v0 holds the merged result).
            for (int i = (int) A->volumes.size() - 1; i >= 0; --i)
                if (A->volumes[i] != v0 && A->volumes[i]->is_model_part())
                    A->delete_volume((size_t) i);
            A->invalidate_bounding_box();
            A->ensure_on_bed();
            plater->changed_mesh(int(o.idx));
            plater->delete_object_from_model(other.idx);   // remove operand B (snapshots)
            int total = 0;
            for (const ModelVolume *v : A->volumes)
                if (v->is_model_part()) total += (int) v->mesh().its.indices.size();
            return total;
        }, py::arg("other"), py::arg("op"))
        .def("emboss_svg", [](const PyObject &o, const std::string &path, double depth) {
            main_thread("Object.emboss_svg");
            auto *plater = plater_or_throw("Object.emboss_svg");
            ModelObject *obj = object_at(o.idx, "Object.emboss_svg");
            TriangleMesh mesh = svg_to_mesh(path, depth);   // may throw before we mutate
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: emboss SVG"));
            ModelVolume *nv = obj->add_volume(std::move(mesh));   // MODEL_PART
            nv->name = boost::filesystem::path(path).stem().string();
            nv->calculate_convex_hull();
            obj->invalidate_bounding_box();
            obj->ensure_on_bed();
            plater->changed_object(int(o.idx));
            return PyVolume{ o.idx, obj->volumes.size() - 1 };
        }, py::arg("path"), py::arg("depth") = 2.0)
        .def("repair", [](const PyObject &o) {
            main_thread("Object.repair");
            auto *plater = plater_or_throw("Object.repair");
            ModelObject *obj = object_at(o.idx, "Object.repair");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: repair"));
            plater->clear_before_change_mesh(int(o.idx), std::string("repair"));
            int merged = 0, removed = 0, tris = 0;
            for (ModelVolume *mv : obj->volumes) {
                if (!mv->is_model_part()) continue;
                indexed_triangle_set its = mv->mesh().its;   // editable copy
                merged  += its_merge_vertices(its);
                removed += its_remove_degenerate_faces(its);
                its_compactify_vertices(its);
                TriangleMesh tm(its);
                mv->set_mesh(std::move(tm));
                mv->calculate_convex_hull();
                mv->set_new_unique_id();
                tris += (int) mv->mesh().its.indices.size();
            }
            obj->invalidate_bounding_box();
            obj->ensure_on_bed();
            plater->changed_mesh(int(o.idx));
            py::dict d;
            d["merged_vertices"] = merged;
            d["removed_faces"]   = removed;
            d["triangles"]       = tris;
            return d;
        })
        .def("set_printable", [](const PyObject &o, bool printable) {
            main_thread("Object.set_printable");
            auto *plater = plater_or_throw("Object.set_printable");
            ModelObject *obj = object_at(o.idx, "Object.set_printable");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: set printable"));
            for (ModelInstance *inst : obj->instances) inst->printable = printable;
            plater->changed_object(int(o.idx));
            return printable;
        }, py::arg("printable"))
        .def_property_readonly("is_printable", [](const PyObject &o) {
            ModelObject *obj = object_at(o.idx, "Object.is_printable");
            if (obj->instances.empty()) return false;
            for (const ModelInstance *inst : obj->instances) if (!inst->printable) return false;
            return true;
        })
        .def("center", [](const PyObject &o) {
            main_thread("Object.center");
            auto *plater = plater_or_throw("Object.center");
            ModelObject *obj = object_at(o.idx, "Object.center");
            if (obj->instances.empty()) throw std::runtime_error("center: object has no instances");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: center on bed"));
            const auto bc = plater->build_volume().bed_center();
            const Vec3d cur = obj->instances[0]->get_offset();
            obj->instances[0]->set_offset(Vec3d(bc.x(), bc.y(), cur.z()));
            obj->invalidate_bounding_box();
            obj->ensure_on_bed();
            plater->changed_object(int(o.idx));
        })
        .def("replace_mesh", [](const PyObject &o, const std::string &path) {
            main_thread("Object.replace_mesh");
            auto *plater = plater_or_throw("Object.replace_mesh");
            ModelObject *obj = object_at(o.idx, "Object.replace_mesh");
            Model tmp;
            if (!load_stl(path.c_str(), &tmp))
                throw std::runtime_error("replace_mesh: could not load STL: " + path);
            if (tmp.objects.empty() || tmp.objects.front()->volumes.empty())
                throw std::runtime_error("replace_mesh: no geometry in " + path);
            tmp.objects.front()->center_around_origin();
            TriangleMesh new_mesh = tmp.objects.front()->volumes.front()->mesh();
            ModelVolume *v0 = nullptr;
            for (ModelVolume *v : obj->volumes) if (v->is_model_part()) { v0 = v; break; }
            if (v0 == nullptr) throw std::runtime_error("replace_mesh: object has no model part");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: replace mesh"));
            plater->clear_before_change_mesh(int(o.idx), std::string("replace mesh"));
            v0->set_mesh(std::move(new_mesh));   // keeps the volume transform + config + type
            v0->calculate_convex_hull();
            v0->set_new_unique_id();
            v0->name = boost::filesystem::path(path).stem().string();
            obj->invalidate_bounding_box();
            obj->ensure_on_bed();
            plater->changed_mesh(int(o.idx));
            return (int) v0->mesh().its.indices.size();
        }, py::arg("path"))
        .def("measure", [](const PyObject &o) {
            ModelObject *obj = object_at(o.idx, "Object.measure");
            double vol = 0.0, area = 0.0; int tris = 0;
            for (const ModelVolume *v : obj->volumes) {
                if (!v->is_model_part()) continue;
                TriangleMesh tm(v->mesh());
                tm.transform(v->get_matrix(), true);
                if (!obj->instances.empty()) tm.transform(obj->instances[0]->get_matrix(), true);
                const indexed_triangle_set &its = tm.its;
                vol += std::abs((double) its_volume(its));
                tris += (int) its.indices.size();
                for (const auto &t : its.indices) {
                    const Vec3d a = its.vertices[t[0]].cast<double>();
                    const Vec3d b = its.vertices[t[1]].cast<double>();
                    const Vec3d c = its.vertices[t[2]].cast<double>();
                    area += 0.5 * (b - a).cross(c - a).norm();
                }
            }
            py::dict out;
            out["volume_mm3"] = vol;
            out["surface_area_mm2"] = area;
            out["triangles"] = tris;
            return out;
        })
        .def("convert_units", [](const PyObject &o, const std::string &conversion) {
            main_thread("Object.convert_units");
            double f;
            if      (conversion == "from_inch")  f = 25.4;
            else if (conversion == "to_inch")    f = 1.0 / 25.4;
            else if (conversion == "from_meter") f = 1000.0;
            else if (conversion == "to_meter")   f = 1.0 / 1000.0;
            else throw std::runtime_error("convert_units: unknown conversion '" + conversion +
                "' (from_inch/to_inch/from_meter/to_meter)");
            auto *plater = plater_or_throw("Object.convert_units");
            ModelObject *obj = object_at(o.idx, "Object.convert_units");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: convert units"));
            obj->scale(Vec3d(f, f, f));
            obj->invalidate_bounding_box();
            plater->changed_object(int(o.idx));
            return f;
        }, py::arg("conversion"))
        .def("place_on_bed", [](const PyObject &o) {
            auto *plater = plater_or_throw("Object.place_on_bed");
            ModelObject *obj = object_at(o.idx, "Object.place_on_bed");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: place on bed"));
            obj->ensure_on_bed();
            obj->invalidate_bounding_box();
            plater->changed_object(int(o.idx));
        })
        .def("scale_to_fit", [](const PyObject &o, double x, double y, double z) {
            if (x <= 0.0 || y <= 0.0 || z <= 0.0)
                throw std::runtime_error("fit size must be > 0");
            auto *plater = plater_or_throw("Object.scale_to_fit");
            ModelObject *obj = object_at(o.idx, "Object.scale_to_fit");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: scale to fit"));
            obj->scale_to_fit(Vec3d(x, y, z));
            obj->invalidate_bounding_box();
            plater->changed_object(int(o.idx));
        }, py::arg("x"), py::arg("y"), py::arg("z"))
        .def("rename", [](const PyObject &o, const std::string &name) {
            auto *plater = plater_or_throw("Object.rename");
            ModelObject *obj = object_at(o.idx, "Object.rename");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: rename object"));
            obj->name = name;
            plater->changed_object(int(o.idx));
        }, py::arg("name"))
        // ---- paint-by-height: per-Z-band config overrides ------------------
        .def("add_height_range", [](const PyObject &o, double min_z, double max_z,
                                    py::dict overrides) {
            if (!(max_z > min_z))
                throw std::runtime_error("max_z must be greater than min_z");
            auto *plater = plater_or_throw("Object.add_height_range");
            ModelObject *obj = object_at(o.idx, "Object.add_height_range");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: add height range"));

            const t_layer_height_range range{ min_z, max_z };
            ModelConfig &cfg = obj->layer_config_ranges[range];   // creates entry

            double lh = 0.2;
            {
                const DynamicPrintConfig &oc = obj->config.get();
                auto *pb = GUI::wxGetApp().preset_bundle;
                if (oc.has("layer_height"))
                    lh = oc.opt_float("layer_height");
                else if (pb && pb->prints.get_edited_preset().config.has("layer_height"))
                    lh = pb->prints.get_edited_preset().config.opt_float("layer_height");
            }
            cfg.set_key_value("layer_height", new ConfigOptionFloat(lh));
            cfg.set_key_value("extruder",     new ConfigOptionInt(0));

            for (auto kv : overrides) {
                const std::string key = kv.first.cast<std::string>();
                const std::string val = py::str(kv.second).cast<std::string>();
                ConfigSubstitutionContext ctx{ ForwardCompatibilitySubstitutionRule::Disable };
                cfg.set_deserialize(key, val, ctx);
            }
            plater->changed_object(int(o.idx));
            return py::make_tuple(min_z, max_z);
        }, py::arg("min_z"), py::arg("max_z"), py::arg("overrides") = py::dict())
        .def("clear_height_ranges", [](const PyObject &o) {
            auto *plater = plater_or_throw("Object.clear_height_ranges");
            ModelObject *obj = object_at(o.idx, "Object.clear_height_ranges");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: clear height ranges"));
            obj->layer_config_ranges.clear();
            plater->changed_object(int(o.idx));
        })
        .def("height_ranges", [](const PyObject &o) {
            ModelObject *obj = object_at(o.idx, "Object.height_ranges");
            py::list out;
            for (const auto &kv : obj->layer_config_ranges) {
                py::dict d;
                d["min_z"] = kv.first.first;
                d["max_z"] = kv.first.second;
                py::dict ov;
                const DynamicPrintConfig &c = kv.second.get();
                for (const std::string &k : c.keys())
                    ov[py::str(k)] = c.opt_serialize(k);
                d["overrides"] = ov;
                out.append(d);
            }
            return out;
        });

    // ---- Model ------------------------------------------------------------
    py::class_<PyModel>(m, "Model")
        .def_property_readonly("object_count", [](const PyModel &) {
            return model_or_throw("Model.object_count").objects.size();
        })
        .def_property_readonly("objects", [](const PyModel &) {
            Model &mo = model_or_throw("Model.objects");
            py::list out;
            for (size_t i = 0; i < mo.objects.size(); ++i)
                out.append(PyObject{i});
            return out;
        })
        // ---- M2 mutation --------------------------------------------------
        .def("add_svg", [](const PyModel &, const std::string &path, double depth) {
            main_thread("Model.add_svg");
            auto *plater = plater_or_throw("Model.add_svg");
            Model &model = model_or_throw("Model.add_svg");
            TriangleMesh mesh = svg_to_mesh(path, depth);   // may throw before we mutate
            namespace fs = boost::filesystem;
            const std::string stem = fs::path(path).stem().string();
            const fs::path tmp = fs::temp_directory_path() / (stem + "_pyslic3r_svg.stl");
            if (!mesh.write_binary(tmp.string().c_str()))
                throw std::runtime_error("add_svg: could not write a temp STL");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: add SVG object"));
            std::vector<fs::path> paths{ tmp };
            const size_t before = model.objects.size();
            std::vector<size_t> idxs = plater->load_files(paths, /*load_model=*/true,
                                                          /*load_config=*/false,
                                                          /*imperial_units=*/false);
            fs::remove(tmp);
            const size_t after = model.objects.size();
            if (after <= before)
                throw std::runtime_error("add_svg: load failed for " + path);
            const size_t new_idx = idxs.empty() ? after - 1 : idxs.back();
            model.objects[new_idx]->name = stem;
            return PyObject{ new_idx };
        }, py::arg("path"), py::arg("depth") = 2.0)
        .def("add", [](const PyModel &, const std::string &path) {
            // Import geometry via the same Plater path as GUI "Add"
            // (load_model=true, load_config=false), under a snapshot.
            auto *plater = plater_or_throw("Model.add");
            const size_t before = plater->model().objects.size();
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: add model"));
            std::vector<boost::filesystem::path> paths{ boost::filesystem::path(path) };
            std::vector<size_t> idxs = plater->load_files(paths, /*load_model=*/true,
                                                          /*load_config=*/false,
                                                          /*imperial_units=*/false);
            const size_t after = plater->model().objects.size();
            if (after <= before)
                throw std::runtime_error("no object added from: " + path);
            return PyObject{ idxs.empty() ? after - 1 : idxs.back() };
        }, py::arg("path"))
        .def("remove", [](const PyModel &, const PyObject &o) {
            auto *plater = plater_or_throw("Model.remove");
            (void) object_at(o.idx, "Model.remove");       // bounds-check
            plater->delete_object_from_model(o.idx);        // snapshots internally
        }, py::arg("object"))
        // ---- clear the scene (UI-parity: Edit -> Delete All) ---------------
        .def("clear", [](const PyModel &) {
            auto *plater = plater_or_throw("Model.clear");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: clear model"));
            while (!plater->model().objects.empty())
                plater->delete_object_from_model(int(plater->model().objects.size()) - 1);
            // delete_object_from_model updates the model but NOT the object-list
            // tree (upstream pairs it with delete_object_from_list; Prusa exposes no
            // public delete_all_objects_from_model). Without this the tree keeps stale
            // nodes, and a later load hits ObjectList::add_layer_root_item ->
            // object(obj_idx) out of bounds (SIGSEGV). Sync the tree to the now-empty model.
            if (auto *ol = GUI::wxGetApp().obj_list())
                ol->delete_all_objects_from_list();
        });

    // ---- Plate / PlateList (single bed) -----------------------------------
    py::class_<PyPlate>(m, "Plate")
        .def_property_readonly("index", [](const PyPlate &p) { return p.idx; })
        .def_property_readonly("object_count", [](const PyPlate &) {
            return plater_or_throw("Plate.object_count")->model().objects.size();
        })
        .def_property_readonly("is_sliceable", [](const PyPlate &) {
            return !plater_or_throw("Plate.is_sliceable")->model().objects.empty();
        })
        .def_property_readonly("config", [](const PyPlate &p) {
            return PyConfig{ConfigSource::Plate, p.idx};
        })
        // ---- paint-by-height: colour changes (the multi-colour primitive) --
        // UI-parity: mirrors the preview vertical-slider "add colour change".
        // Single-bed: writes Model::custom_gcode_per_print_z().
        .def("set_color_changes", [](const PyPlate &, py::list changes) {
            auto *plater = plater_or_throw("Plate.set_color_changes");
            std::vector<std::string> ext_colors =
                plater->get_extruder_color_strings_from_plater_config();  // 0-based

            CustomGCode::Info info;
            info.mode = CustomGCode::Undef;
            for (auto handle : changes) {
                py::dict d = handle.cast<py::dict>();
                CustomGCode::Item item;
                item.type = CustomGCode::ColorChange;
                if (!d.contains("z"))
                    throw std::runtime_error("each colour change needs a 'z' (print_z)");
                item.print_z = d["z"].cast<double>();
                if (!d.contains("extruder"))
                    throw std::runtime_error("each colour change needs an 'extruder' (1-based)");
                item.extruder = d["extruder"].cast<int>();
                if (item.extruder < 1)
                    throw std::runtime_error("extruder is 1-based; must be >= 1");
                if (d.contains("color") && !d["color"].is_none()) {
                    item.color = d["color"].cast<std::string>();
                } else {
                    const int i0 = item.extruder - 1;
                    item.color = (i0 >= 0 && i0 < int(ext_colors.size()))
                                     ? ext_colors[i0] : std::string("#FFFFFF");
                }
                info.gcodes.push_back(std::move(item));
            }
            std::sort(info.gcodes.begin(), info.gcodes.end());
            CustomGCode::check_mode_for_custom_gcode_per_print_z(info);

            GUI::Plater::TakeSnapshot snap(plater, std::string("API: set colour changes"));
            plater->model().custom_gcode_per_print_z() = std::move(info);
            plater->schedule_background_process();
        }, py::arg("changes"))
        .def("set_custom_gcodes", [](const PyPlate &, py::list items) {
            auto *plater = plater_or_throw("Plate.set_custom_gcodes");
            std::vector<std::string> ext_colors = plater->get_extruder_color_strings_from_plater_config();
            CustomGCode::Info info;
            info.mode = CustomGCode::Undef;
            for (auto handle : items) {
                py::dict d = handle.cast<py::dict>();
                if (!d.contains("z")) throw std::runtime_error("each marker needs a z (print_z)");
                CustomGCode::Item item;
                item.print_z = d["z"].cast<double>();
                const std::string t = d.contains("type") ? d["type"].cast<std::string>() : std::string("custom");
                if (t == "pause") {
                    item.type = CustomGCode::PausePrint;
                    item.extruder = 0;
                    item.extra = (d.contains("message") && !d["message"].is_none()) ? d["message"].cast<std::string>() : std::string();
                } else if (t == "custom") {
                    if (!d.contains("gcode")) throw std::runtime_error("a custom marker needs a gcode string");
                    item.type = CustomGCode::Custom;
                    item.extruder = 0;
                    item.extra = d["gcode"].cast<std::string>();
                } else if (t == "color" || t == "tool") {
                    item.type = (t == "color") ? CustomGCode::ColorChange : CustomGCode::ToolChange;
                    item.extruder = d.contains("extruder") ? d["extruder"].cast<int>() : 1;
                    if (item.extruder < 1) throw std::runtime_error("extruder is 1-based; must be >= 1");
                    if (d.contains("color") && !d["color"].is_none()) { item.color = d["color"].cast<std::string>(); }
                    else { const int i0 = item.extruder - 1; item.color = (i0 >= 0 && i0 < (int) ext_colors.size()) ? ext_colors[i0] : std::string("#FFFFFF"); }
                } else {
                    throw std::runtime_error("type must be pause | custom | color | tool");
                }
                info.gcodes.push_back(std::move(item));
            }
            std::sort(info.gcodes.begin(), info.gcodes.end());
            CustomGCode::check_mode_for_custom_gcode_per_print_z(info);
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: set custom G-code"));
            plater->model().custom_gcode_per_print_z() = std::move(info);
            plater->schedule_background_process();
        }, py::arg("items"))
        .def("custom_gcodes", [](const PyPlate &) {
            auto *plater = plater_or_throw("Plate.custom_gcodes");
            py::list out;
            for (const CustomGCode::Item &item : plater->model().custom_gcode_per_print_z().gcodes) {
                py::dict d;
                d["z"] = item.print_z;
                const char *ts = "custom";
                switch (item.type) {
                    case CustomGCode::ColorChange: ts = "color"; break;
                    case CustomGCode::PausePrint:  ts = "pause"; break;
                    case CustomGCode::ToolChange:  ts = "tool";  break;
                    case CustomGCode::Template:    ts = "template"; break;
                    default: ts = "custom"; break;
                }
                d["type"] = std::string(ts);
                d["extruder"] = item.extruder;
                d["color"] = item.color;
                d["extra"] = item.extra;
                out.append(d);
            }
            return out;
        })
        .def("clear_color_changes", [](const PyPlate &) {
            auto *plater = plater_or_throw("Plate.clear_color_changes");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: clear colour changes"));
            plater->model().custom_gcode_per_print_z() = CustomGCode::Info{};
            plater->schedule_background_process();
        })
        .def("color_changes", [](const PyPlate &) {
            auto *plater = plater_or_throw("Plate.color_changes");
            py::list out;
            const CustomGCode::Info &info = plater->model().custom_gcode_per_print_z();
            for (const CustomGCode::Item &item : info.gcodes) {
                py::dict d;
                d["z"]        = item.print_z;
                d["extruder"] = item.extruder;
                d["color"]    = item.color;
                d["type"]     = int(item.type);
                out.append(d);
            }
            return out;
        });

    py::class_<PyPlateList>(m, "PlateList")
        .def_property_readonly("count", [](const PyPlateList &) { return 1; })
        .def("__len__", [](const PyPlateList &) { return 1; })
        .def("__getitem__", [](const PyPlateList &, int i) {
            if (i < 0 || i >= 1) throw py::index_error("plate index out of range");
            return PyPlate{i};
        })
        // ---- M2 mutation --------------------------------------------------
        .def("arrange", [](const PyPlateList &, bool wait) {
            // Auto-arrange, same as the toolbar button. Asynchronous — starts a
            // background ArrangeJob. With wait=True, pump the event loop (GIL
            // released) until the job worker is idle.
            auto *plater = plater_or_throw("PlateList.arrange");
            plater->arrange(/*current_bed_only=*/false);
            if (!wait) return;
            main_thread("PlateList.arrange(wait=True)");
            py::gil_scoped_release nogil;
            using clock = std::chrono::steady_clock;
            const auto t0 = clock::now();
            for (;;) {
                if (wxTheApp != nullptr) wxTheApp->Yield(true);
                if (plater->get_ui_job_worker().is_idle()) break;
                if (clock::now() - t0 > std::chrono::seconds(120)) break;
                wxMilliSleep(40);
            }
        }, py::arg("wait") = false);

    // ---- SliceResult ------------------------------------------------------
    py::class_<PySliceResult>(m, "SliceResult")
        .def_property_readonly("success",       [](const PySliceResult &r) { return r.success; })
        .def_property_readonly("print_time_s",  [](const PySliceResult &r) { return r.print_time_s; })
        .def_property_readonly("layer_count",   [](const PySliceResult &r) { return r.layer_count; })
        .def_property_readonly("total_weight",  [](const PySliceResult &r) { return r.total_weight; })
        .def_property_readonly("total_cost",    [](const PySliceResult &r) { return r.total_cost; })
        .def_property_readonly("total_toolchanges", [](const PySliceResult &r) { return r.total_toolchanges; })
        .def_property_readonly("total_wipe_tower_filament", [](const PySliceResult &r) { return r.total_wipe_tower_filament; })
        .def_property_readonly("filament_g",    [](const PySliceResult &r) { return r.filament_g; })
        .def_property_readonly("filament_mm",   [](const PySliceResult &r) { return r.filament_mm; })
        .def_property_readonly("gcode_3mf_path",[](const PySliceResult &r) { return r.gcode_path; })
        .def_property_readonly("error",         [](const PySliceResult &r) { return r.error; });

    // ---- SliceJob ---------------------------------------------------------
    py::class_<PySliceJob>(m, "SliceJob")
        .def_property_readonly("done", [](const PySliceJob &) {
            auto *plater = plater_or_throw("SliceJob.done");
            return plater->active_fff_print().finished();
        })
        .def_property_readonly("progress", [](const PySliceJob &) {
            auto *plater = plater_or_throw("SliceJob.progress");
            const char *stage = plater->is_background_process_update_scheduled() ? "scheduled" : "idle";
            return py::make_tuple(py::none(), std::string(stage));
        })
        .def("cancel", [](const PySliceJob &) {
            plater_or_throw("SliceJob.cancel")->get_ui_job_worker().cancel_all();
        })
        .def("wait", [](const PySliceJob &j, py::object timeout) -> PySliceResult {
            // Runs on the wx main thread (Python's thread in this model). Pump
            // the event loop so the slicing worker's completion events land,
            // polling the print's finished() state — a controlled event pump,
            // NOT a nested modal loop. GIL released while pumping.
            main_thread("SliceJob.wait");
            auto *plater = plater_or_throw("SliceJob.wait");
            const double timeout_s = timeout.is_none() ? 300.0 : timeout.cast<double>();

            PySliceResult res;
            {
                py::gil_scoped_release nogil;
                using clock = std::chrono::steady_clock;
                const auto t0 = clock::now();
                // Don't accept a stale finished() from a previous slice: require
                // we've observed the process as running/scheduled at least once.
                bool saw_running = false;
                for (;;) {
                    if (wxTheApp != nullptr) wxTheApp->Yield(true);  // deliver events
                    const bool finished  = plater->active_fff_print().finished();
                    const bool scheduled = plater->is_background_process_update_scheduled();
                    if (!finished || scheduled) saw_running = true;
                    if (finished && saw_running && !scheduled) { res.success = true; break; }
                    const auto elapsed = clock::now() - t0;
                    if (elapsed > std::chrono::duration<double>(timeout_s)) {
                        res.success = false;
                        res.error = "timeout waiting for slice (print did not finish)";
                        break;
                    }
                    wxMilliSleep(40);
                }
            }
            if (res.success) fill_slice_result(j.plate_idx, res);
            return res;
        }, py::arg("timeout") = py::none());

    // ---- Document ---------------------------------------------------------
    // ---- Settings (curated/normalized) ------------------------------------
    py::class_<PySettings>(m, "Settings")
        .def("names", [](const PySettings &) {
            py::list out;
            for (const auto &d : settings_defs()) out.append(d.name);
            return out;
        })
        .def("set", [](const PySettings &, const std::string &name, py::object value) {
            settings_set(name, value);
        }, py::arg("name"), py::arg("value"))
        .def("get", [](const PySettings &, const std::string &name) {
            return settings_get(name);
        }, py::arg("name"))
        .def("options", [](const PySettings &, const std::string &name) -> py::object {
            const SettingDef *d = find_setting(name);
            if (!d) throw std::runtime_error("unknown setting: " + name);
            if (d->emap.empty()) return py::none();
            py::list out;
            for (const auto &p : d->emap) out.append(p.first);
            return out;
        }, py::arg("name"))
        .def("describe", [](const PySettings &, const std::string &name) {
            const SettingDef *d = find_setting(name);
            if (!d) throw std::runtime_error("unknown setting: " + name);
            py::dict out;
            out["name"] = d->name;
            out["key"] = d->key;
            const char k[2] = { d->kind, 0 };
            out["kind"] = std::string(k);
            out["is_vector"] = d->is_vector;
            if (!d->emap.empty()) {
                py::list opts;
                for (const auto &p : d->emap) opts.append(p.first);
                out["options"] = opts;
            }
            return out;
        }, py::arg("name"));

    // ---- Filament (per-slot configuration) --------------------------------
    py::class_<PyFilament>(m, "Filament")
        .def_property_readonly("slot", [](const PyFilament &f) { return (int) f.slot; })
        .def_property("type",
            [](const PyFilament &f) { return filament_get(f.slot, "filament_type"); },
            [](const PyFilament &f, py::object v) { filament_set(f.slot, "filament_type", v); })
        .def_property("color",
            [](const PyFilament &f) { return filament_get(f.slot, "filament_colour"); },
            [](const PyFilament &f, py::object v) { filament_set(f.slot, "filament_colour", v); })
        .def_property("nozzle_temp",
            [](const PyFilament &f) { return filament_get(f.slot, "temperature"); },
            [](const PyFilament &f, py::object v) { filament_set(f.slot, "temperature", v); })
        .def_property("bed_temp",
            [](const PyFilament &f) { return filament_get(f.slot, "bed_temperature"); },
            [](const PyFilament &f, py::object v) { filament_set(f.slot, "bed_temperature", v); })
        .def_property("flow_ratio",
            [](const PyFilament &f) { return filament_get(f.slot, "extrusion_multiplier"); },
            [](const PyFilament &f, py::object v) { filament_set(f.slot, "extrusion_multiplier", v); })
        .def_property("max_volumetric_speed",
            [](const PyFilament &f) { return filament_get(f.slot, "filament_max_volumetric_speed"); },
            [](const PyFilament &f, py::object v) { filament_set(f.slot, "filament_max_volumetric_speed", v); })
        .def_property("diameter",
            [](const PyFilament &f) { return filament_get(f.slot, "filament_diameter"); },
            [](const PyFilament &f, py::object v) { filament_set(f.slot, "filament_diameter", v); })
        .def_property("density",
            [](const PyFilament &f) { return filament_get(f.slot, "filament_density"); },
            [](const PyFilament &f, py::object v) { filament_set(f.slot, "filament_density", v); })
        .def_property("cost",
            [](const PyFilament &f) { return filament_get(f.slot, "filament_cost"); },
            [](const PyFilament &f, py::object v) { filament_set(f.slot, "filament_cost", v); })
        .def_property_readonly("preset", [](const PyFilament &f) -> py::object {
            auto *pb = GUI::wxGetApp().preset_bundle;
            return f.slot < pb->extruders_filaments.size()
                ? py::cast(pb->extruders_filaments[f.slot].get_selected_preset_name()) : py::none();
        })
        .def("presets", [](const PyFilament &) {
            main_thread("Filament.presets");
            py::list out;
            for (const Preset &p : GUI::wxGetApp().preset_bundle->filaments.get_presets())
                if (p.is_visible && p.is_compatible) out.append(p.name);
            return out;
        })
        .def("apply_preset", [](const PyFilament &f, const std::string &name) {
            main_thread("Filament.apply_preset");
            auto *plater = plater_or_throw("Filament.apply_preset");
            PresetBundle *pb = GUI::wxGetApp().preset_bundle;
            if (pb->filaments.find_preset(name) == nullptr)
                throw std::runtime_error("unknown filament preset: " + name);
            pb->set_filament_preset(f.slot, name);
            pb->update_multi_material_filament_presets();
            plater->on_config_change(pb->full_config());
        }, py::arg("name"))
        .def("__repr__", [](const PyFilament &f) {
            return "<Filament slot=" + std::to_string(f.slot) + ">"; });

    py::class_<PyDocument>(m, "Document")
        .def_property_readonly("object_count", [](const PyDocument &) {
            return model_or_throw("Document.object_count").objects.size();
        })
        .def_property_readonly("model", [](const PyDocument &) { return PyModel{}; })
        .def_property_readonly("plates", [](const PyDocument &) { return PyPlateList{}; })
        .def_property_readonly("config", [](const PyDocument &) {
            return PyConfig{ConfigSource::Global};
        })
        .def_property_readonly("print_config", [](const PyDocument &) {
            return PyConfig{ConfigSource::Print};
        })
        .def_property_readonly("filament_config", [](const PyDocument &) {
            return PyConfig{ConfigSource::Filament};
        })
        .def_property_readonly("printer_config", [](const PyDocument &) {
            return PyConfig{ConfigSource::Printer};
        })
        .def_property_readonly("settings", [](const PyDocument &) { return PySettings{}; })
        .def_property_readonly("filaments", [](const PyDocument &) {
            int n = (int)(GUI::wxGetApp().preset_bundle->printers.get_edited_preset().config.option<ConfigOptionFloats>("nozzle_diameter") ? GUI::wxGetApp().preset_bundle->printers.get_edited_preset().config.option<ConfigOptionFloats>("nozzle_diameter")->size() : 1);
            py::list out;
            for (int i = 0; i < n; ++i) out.append(PyFilament{(size_t) i});
            return out;
        })
        .def_property_readonly("filament_count", [](const PyDocument &) {
            auto *nd = GUI::wxGetApp().preset_bundle->printers.get_edited_preset()
                          .config.option<ConfigOptionFloats>("nozzle_diameter");
            return (int)(nd ? nd->size() : 1);
        })
        // Set up N project filaments (Prusa: extruder count = nozzle_diameter
        // .size(); resize per-extruder printer opts + sync MM filament presets).
        .def("set_filaments", [](const PyDocument &, py::list colors) {
            auto *plater = plater_or_throw("Document.set_filaments");
            auto *pb = GUI::wxGetApp().preset_bundle;
            const unsigned int n = (unsigned int) colors.size();
            if (n < 1 || n > 16) throw std::runtime_error("filament count must be 1..16");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: set filaments"));
            Preset &printer = pb->printers.get_edited_preset();
            printer.set_num_extruders(n);
            pb->update_multi_material_filament_presets();
            std::vector<std::string> cols;
            for (auto c : colors) { std::string h = c.cast<std::string>();
                if (!h.empty() && h[0] != '#') h = "#" + h; cols.push_back(h); }
            if (auto *opt = printer.config.option<ConfigOptionStrings>("extruder_colour"))
                opt->values = cols;
            plater->on_config_change(pb->full_config());
            return int(n);
        }, py::arg("colors"))
        // ---- M3 slicing ---------------------------------------------------
        .def("validate", [](const PyDocument &) {
            main_thread("Document.validate");
            auto *plater = plater_or_throw("Document.validate");
            // Apply the current model + edited config to the Print synchronously (the
            // same apply the background slicer does), then read validate() — no slice.
            Print &print = plater->active_fff_print();
            const DynamicPrintConfig cfg = GUI::wxGetApp().preset_bundle->full_config();
            print.apply(plater->model(), cfg);
            std::string err = print.validate();
            py::dict d;
            d["ok"]    = err.empty();
            d["error"] = err;
            return d;
        })
        .def("slice", [](const PyDocument &, py::object /*plate*/) {
            // Single-bed: slice the whole bed (reslice starts the worker).
            auto *plater = plater_or_throw("Document.slice");
            plater->reslice();
            return PySliceJob{0};
        }, py::arg("plate") = py::none())
        // Copy the last sliced G-code to `path` (the "Export G-code" result).
        .def("save_gcode", [](const PyDocument &, const std::string &path) {
            namespace fs = boost::filesystem;
            auto *plater = plater_or_throw("Document.save_gcode");
            const auto &results = plater->get_gcode_results();
            if (results.empty() || results.front().filename.empty())
                throw std::runtime_error("no sliced G-code yet; call slice().wait() first");
            fs::path src(results.front().filename);
            if (!fs::exists(src))
                throw std::runtime_error("sliced G-code file is gone: " + src.string());
            fs::copy_file(src, fs::path(path), fs::copy_option::overwrite_if_exists);
            return path;
        }, py::arg("path"))
        // Save the project as a .3mf (UI-parity: Save Project).
        // Export the arranged geometry as STL (UI-parity: Export -> STL, but
        // without the file dialog). Combined transformed mesh over model parts.
        .def("export_stl", [](const PyDocument &, const std::string &path, bool binary) {
            main_thread("Document.export_stl");
            Model &model = model_or_throw("Document.export_stl");
            if (model.objects.empty())
                throw std::runtime_error("export_stl: no objects to export");
            TriangleMesh out;
            for (const ModelObject *mo : model.objects) {
                TriangleMesh obj_mesh;
                for (const ModelVolume *v : mo->volumes)
                    if (v->is_model_part()) {
                        TriangleMesh vm(v->mesh());
                        vm.transform(v->get_matrix(), true);
                        obj_mesh.merge(vm);
                    }
                for (const ModelInstance *inst : mo->instances) {
                    TriangleMesh m(obj_mesh);
                    m.transform(inst->get_matrix(), true);
                    out.merge(m);
                }
            }
            if (out.empty())
                throw std::runtime_error("export_stl: combined mesh is empty");
            if (!Slic3r::store_stl(path.c_str(), &out, binary))
                throw std::runtime_error("export_stl: write failed: " + path);
            return path;
        }, py::arg("path"), py::arg("binary") = true)
        // Open a full 3MF project: geometry + its embedded config (unlike
        // model.add, which loads geometry only). Does not clear the scene —
        // call model.clear() first for a clean replace.
        .def("open_project", [](const PyDocument &, const std::string &path) {
            auto *plater = plater_or_throw("Document.open_project");
            GUI::Plater::TakeSnapshot snap(plater, std::string("API: open project"));
            std::vector<boost::filesystem::path> paths{ boost::filesystem::path(path) };
            plater->load_files(paths, /*load_model=*/true, /*load_config=*/true,
                               /*imperial_units=*/false);
            return plater->model().objects.size();
        }, py::arg("path"))
        .def("to_script", [](const PyDocument &) {
            main_thread("Document.to_script");
            return build_state_script();
        })
        .def("save_3mf", [](const PyDocument &, const std::string &path) {
            namespace fs = boost::filesystem;
            auto *plater = plater_or_throw("Document.save_3mf");
            plater->export_3mf(fs::path(path));
            if (!fs::exists(fs::path(path)) || fs::file_size(fs::path(path)) == 0)
                throw std::runtime_error("3mf not written: " + path);
            return path;
        }, py::arg("path"));

    // ---- Application ------------------------------------------------------
    py::class_<PyApp>(m, "Application")
        .def_property_readonly("version", [](const PyApp &) {
            main_thread("app.version");
            return std::string(SLIC3R_VERSION);
        })
        .def_property_readonly("name", [](const PyApp &) {
            main_thread("app.name");
            return std::string(SLIC3R_APP_NAME);   // BambuStudio / OrcaSlicer / PrusaSlicer
        })
        .def_property_readonly("active_document", [](const PyApp &) -> py::object {
            main_thread("app.active_document");
            if (GUI::wxGetApp().plater() == nullptr)
                return py::none();
            return py::cast(PyDocument{});
        })
        .def_property_readonly("selected_printer", [](const PyApp &) {
            main_thread("app.selected_printer");
            return GUI::wxGetApp().preset_bundle->printers.get_selected_preset_name();
        })
        .def("printers", [](const PyApp &) {
            main_thread("app.printers");
            std::vector<std::string> out;
            for (const Preset &p : GUI::wxGetApp().preset_bundle->printers.get_presets())
                if (p.is_visible)
                    out.push_back(p.name);
            return out;
        })
        // Open device plane (PrusaLink/OctoPrint). Registered in PyDeviceHost.cpp,
        // exposed there as pyslic3r.device.
        .def_property_readonly("device", [](const PyApp &) {
            return py::module_::import("pyslic3r").attr("device");
        });

    m.attr("app") = py::cast(PyApp{});
}

} // namespace pyslic3r
