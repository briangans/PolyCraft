//! Transform gizmo math, independent of the GPU and of windowing.
//!
//! The gizmo is a set of three coloured axes drawn at the current selection, used later to drag
//! components along an axis. This module answers the questions the drawing and dragging steps
//! will need, and nothing else: where the handle sits, how long its arms should be, and which
//! axis a pointer is nearest. Keeping it pure means all of it is testable without a window, and
//! matches the project rule that picking/selection/camera math never touch `wgpu` or `winit`.
//!
//! Two choices from the architecture shape the API here:
//! - The gizmo stays a constant size on screen (unlike the mesh). Its world length is therefore
//!   derived from a pixel length, using the same `world_per_pixel` the picker uses, so picking
//!   and drawing agree on what a "pixel" means.
//! - The active tool is separate from `EditMode` (move/rotate/scale, bound to W/E/R in a later step). `gizmo::pivot` takes
//!   a mode and a selection because where the handle sits depends on what is selected, not on
//!   whether we are moving, rotating or scaling.

use crate::mesh::{HalfEdgeMesh, VertexId};
use crate::picking::{world_per_pixel, Ray};
use crate::scene::EditMode;
use crate::selection::Selection;
use cgmath::{InnerSpace, Point3, Vector3};

/// Bright yellow highlight when an axis is hovered or actively dragged (Maya convention).
pub const COLOR_AXIS_ACTIVE: [f32; 3] = [1.0, 0.90, 0.15];

/// Pick tolerance for clicking gizmo arms, in screen pixels.
pub const GIZMO_PICK_TOLERANCE_PIXELS: f32 = 8.0;

/// The three coordinate axes of the transform gizmo.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GizmoAxis {
    X,
    Y,
    Z,
}

impl GizmoAxis {
    pub const ALL: [GizmoAxis; 3] = [GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z];

    #[allow(dead_code)]
    pub fn index(self) -> usize {
        match self {
            GizmoAxis::X => 0,
            GizmoAxis::Y => 1,
            GizmoAxis::Z => 2,
        }
    }

    #[allow(dead_code)]
    pub fn from_index(index: usize) -> Option<GizmoAxis> {
        match index {
            0 => Some(GizmoAxis::X),
            1 => Some(GizmoAxis::Y),
            2 => Some(GizmoAxis::Z),
            _ => None,
        }
    }

    pub fn unit_vector(self) -> Vector3<f32> {
        match self {
            GizmoAxis::X => Vector3::new(1.0, 0.0, 0.0),
            GizmoAxis::Y => Vector3::new(0.0, 1.0, 0.0),
            GizmoAxis::Z => Vector3::new(0.0, 0.0, 1.0),
        }
    }

    pub fn color(self) -> [f32; 3] {
        match self {
            GizmoAxis::X => COLOR_AXIS_X,
            GizmoAxis::Y => COLOR_AXIS_Y,
            GizmoAxis::Z => COLOR_AXIS_Z,
        }
    }
}

/// The three world-space unit axes of a translate gizmo, with their Maya convention colours.
/// Kept together so the drawing step and any later axis hit-testing agree on the same three.
// Built from plain struct literals rather than `Vector3::unit_x()`, because cgmath's unit-vector
// helpers are not `const fn`, and a `const` array cannot call them.
pub const AXES: [(Vector3<f32>, [f32; 3]); 3] = [
    (
        Vector3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        COLOR_AXIS_X,
    ),
    (
        Vector3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        },
        COLOR_AXIS_Y,
    ),
    (
        Vector3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        },
        COLOR_AXIS_Z,
    ),
];

pub const COLOR_AXIS_X: [f32; 3] = [0.85, 0.20, 0.18];
pub const COLOR_AXIS_Y: [f32; 3] = [0.25, 0.70, 0.24];
pub const COLOR_AXIS_Z: [f32; 3] = [0.20, 0.40, 0.95];

/// On-screen length of a gizmo arm, in pixels. Maya's handles stay roughly this size at any zoom.
pub const AXIS_LENGTH_PIXELS: f32 = 60.0;

/// Collects the unique vertices affected by the current mode and selection.
pub fn selected_vertices(
    mesh: &HalfEdgeMesh,
    mode: EditMode,
    selection: &Selection,
) -> Vec<VertexId> {
    let mut result = Vec::new();
    let mut seen = std::collections::HashSet::new();

    match mode {
        EditMode::Object => {
            if selection.is_object_selected() {
                for i in 0..mesh.vertices.len() {
                    result.push(VertexId(i));
                }
            }
        }
        EditMode::Vertex => {
            for &v in selection.vertices() {
                result.push(v);
            }
        }
        EditMode::Edge => {
            for &edge in selection.edges() {
                let v0 = VertexId(edge.0);
                let v1 = VertexId(edge.1);
                if seen.insert(v0) {
                    result.push(v0);
                }
                if seen.insert(v1) {
                    result.push(v1);
                }
            }
        }
        EditMode::Face => {
            for &face in selection.faces() {
                for v in mesh.face_vertices(face) {
                    if seen.insert(v) {
                        result.push(v);
                    }
                }
            }
        }
    }

    result
}

/// The point a gizmo is anchored to, if there is a selection to anchor it to. Maya puts the move
/// handle at the selection's centroid, and at the object when the whole object is selected.
pub fn pivot(mesh: &HalfEdgeMesh, mode: EditMode, selection: &Selection) -> Option<Point3<f32>> {
    let verts = selected_vertices(mesh, mode, selection);
    if verts.is_empty() {
        return None;
    }
    let mut sum = Vector3::new(0.0, 0.0, 0.0);
    for v in &verts {
        let p = mesh.vertex_position(*v);
        sum += Vector3::new(p.x, p.y, p.z);
    }
    let f = verts.len() as f32;
    Some(Point3::new(sum.x / f, sum.y / f, sum.z / f))
}

/// World-space length of one gizmo arm, so it measures `AXIS_LENGTH_PIXELS` on screen at the
/// pivot's distance from `camera`. Returns 0 for a degenerate view (pivot at the eye), so callers
/// can simply skip drawing an arm that would have no size.
pub fn axis_world_length(pivot: Point3<f32>, camera_eye: Point3<f32>, viewport_height: f32) -> f32 {
    let distance = (pivot - camera_eye).magnitude();
    if !distance.is_finite() || distance <= 0.0 {
        return 0.0;
    }
    world_per_pixel(distance, viewport_height) * AXIS_LENGTH_PIXELS
}

/// One gizmo arm: a line segment from the pivot along a world axis, in that axis's colour.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Arm {
    pub start: Point3<f32>,
    pub end: Point3<f32>,
    pub color: [f32; 3],
}

/// The translate gizmo's arms for the current selection and camera: one per axis, anchored at
/// `pivot`, each world-long enough to measure `AXIS_LENGTH_PIXELS` at the pivot's depth. Empty
/// with no selection. If an axis is active (hovered or dragged), it is highlighted in bright yellow.
pub fn translate_arms(
    mesh: &HalfEdgeMesh,
    mode: EditMode,
    selection: &Selection,
    camera_eye: Point3<f32>,
    viewport_height: f32,
    active_axis: Option<GizmoAxis>,
) -> Vec<Arm> {
    let pivot = match pivot(mesh, mode, selection) {
        Some(pivot) => pivot,
        None => return Vec::new(),
    };
    let length = axis_world_length(pivot, camera_eye, viewport_height);
    if length <= 0.0 {
        return Vec::new();
    }
    GizmoAxis::ALL
        .iter()
        .map(|&axis| {
            let color = if active_axis == Some(axis) {
                COLOR_AXIS_ACTIVE
            } else {
                axis.color()
            };
            Arm {
                start: pivot,
                end: pivot + axis.unit_vector() * length,
                color,
            }
        })
        .collect()
}

/// Tests if a ray passes within `tolerance_pixels` of any gizmo axis arm.
///
/// Depth testing is intentionally skipped (Always-pass), matching how the gizmo is rendered on top
/// of the mesh. Returns the axis closest to the ray within tolerance.
pub fn pick_gizmo_axis(
    pivot: Point3<f32>,
    camera_eye: Point3<f32>,
    viewport_height: f32,
    ray: &Ray,
    tolerance_pixels: f32,
) -> Option<GizmoAxis> {
    let length = axis_world_length(pivot, camera_eye, viewport_height);
    if length <= 0.0 {
        return None;
    }

    let mut best: Option<(f32, GizmoAxis)> = None;

    for axis in GizmoAxis::ALL {
        let arm_start = pivot;
        let arm_end = pivot + axis.unit_vector() * length;

        let Some((offset, distance)) = crate::picking::ray_segment(ray, arm_start, arm_end) else {
            continue;
        };

        let radius = world_per_pixel(distance, viewport_height) * tolerance_pixels;
        if offset > radius {
            continue;
        }

        let better = best
            .map(|(best_offset, _)| offset < best_offset)
            .unwrap_or(true);
        if better {
            best = Some((offset, axis));
        }
    }

    best.map(|(_, axis)| axis)
}

/// Calculates the scalar parameter `t` along the axis line `L(t) = pivot + t * axis_direction`
/// at the point of closest approach to `ray`.
///
/// Returns `None` if the ray is nearly parallel to the axis (degenerate angle).
pub fn project_ray_onto_axis(
    pivot: Point3<f32>,
    axis_direction: Vector3<f32>,
    ray: &Ray,
) -> Option<f32> {
    let u = axis_direction;
    let d = ray.direction;
    let w0 = ray.origin - pivot;

    let b = d.dot(u);
    let denom = 1.0 - b * b;
    if denom < 1e-4 {
        return None;
    }

    let ray_w0 = d.dot(w0);
    let u_w0 = u.dot(w0);

    let t = (u_w0 - b * ray_w0) / denom;
    Some(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selection::Selection;

    fn cube() -> HalfEdgeMesh {
        HalfEdgeMesh::create_cube()
    }

    #[test]
    fn no_selection_has_no_pivot() {
        let mesh = cube();
        let selection = Selection::new();
        assert_eq!(pivot(&mesh, EditMode::Object, &selection), None);
        assert_eq!(pivot(&mesh, EditMode::Vertex, &selection), None);
        assert_eq!(pivot(&mesh, EditMode::Edge, &selection), None);
        assert_eq!(pivot(&mesh, EditMode::Face, &selection), None);
    }

    #[test]
    fn object_mode_centres_the_whole_cube() {
        let mesh = cube();
        let mut selection = Selection::new();
        selection.apply_object_click(true, false);
        let pivot =
            pivot(&mesh, EditMode::Object, &selection).expect("selected object has a pivot");
        // The unit cube is symmetric about the origin.
        assert!(
            (pivot - Point3::new(0.0, 0.0, 0.0)).magnitude() < 1e-6,
            "pivot was {pivot:?}"
        );
    }

    #[test]
    fn vertex_mode_centres_the_selected_vertices() {
        let mesh = cube();
        let mut selection = Selection::new();
        selection.apply_click(Some(VertexId(0)), false, Selection::vertices_mut);
        selection.apply_click(Some(VertexId(1)), true, Selection::vertices_mut);
        // Vertices 0 (-0.5,-0.5,0.5) and 1 (0.5,-0.5,0.5): midpoint is (0,-0.5,0.5).
        let pivot = pivot(&mesh, EditMode::Vertex, &selection).expect("two selected vertices");
        assert!(
            (pivot - Point3::new(0.0, -0.5, 0.5)).magnitude() < 1e-6,
            "pivot was {pivot:?}"
        );
    }

    #[test]
    fn edge_mode_centres_the_endpoints_of_selected_edges() {
        let mesh = cube();
        let mut selection = Selection::new();
        // Edge 0-1 endpoints: (-0.5,-0.5,0.5) and (0.5,-0.5,0.5); midpoint (0,-0.5,0.5).
        selection.apply_click(Some(crate::mesh::EdgeId(0, 1)), false, Selection::edges_mut);
        let pivot = pivot(&mesh, EditMode::Edge, &selection).expect("one selected edge");
        assert!(
            (pivot - Point3::new(0.0, -0.5, 0.5)).magnitude() < 1e-6,
            "pivot was {pivot:?}"
        );
    }

    #[test]
    fn face_mode_centres_the_selected_face_vertices() {
        let mesh = cube();
        let mut selection = Selection::new();
        // Face 0 (front) has vertices 0..3, symmetric about (0,0,0.5).
        selection.apply_click(Some(crate::mesh::FaceId(0)), false, Selection::faces_mut);
        let pivot = pivot(&mesh, EditMode::Face, &selection).expect("one selected face");
        assert!(
            (pivot - Point3::new(0.0, 0.0, 0.5)).magnitude() < 1e-6,
            "pivot was {pivot:?}"
        );
    }

    #[test]
    fn pivot_is_none_when_nothing_selected_even_in_object_mode() {
        // Guards against "selected_count == 0 but object flag left set" type drift.
        let mesh = cube();
        let selection = Selection::new();
        assert_eq!(pivot(&mesh, EditMode::Object, &selection), None);
    }

    #[test]
    fn axis_grows_as_the_pivot_recedes() {
        let eye = Point3::new(0.0, 0.0, 3.0);
        let near = axis_world_length(Point3::new(0.0, 0.0, 2.0), eye, 600.0);
        let far = axis_world_length(Point3::new(0.0, 0.0, -3.0), eye, 600.0);
        assert!(near > 0.0);
        assert!(
            far > near,
            "far pivot must need a longer world arm: near {near}, far {far}"
        );
    }

    #[test]
    fn arm_scales_linearly_with_depth_to_hold_a_constant_pixel_size() {
        // On-screen size is fixed, so the world length must grow in proportion to the pivot's
        // distance from the eye. At 3x the depth, the arm must be 3x as long in world units.
        let eye = Point3::new(0.0, 0.0, 0.0);
        let at_1 = axis_world_length(Point3::new(0.0, 0.0, 1.0), eye, 600.0);
        let at_3 = axis_world_length(Point3::new(0.0, 0.0, 3.0), eye, 600.0);
        assert!(
            (at_3 - 3.0 * at_1).abs() < 1e-5,
            "arm must be proportional to depth: 1-unit {at_1}, 3-unit {at_3}"
        );
    }

    #[test]
    fn degenerate_view_yields_no_arm_rather_than_nan() {
        let eye = Point3::new(0.0, 0.0, 3.0);
        assert_eq!(axis_world_length(eye, eye, 600.0), 0.0);
    }

    #[test]
    fn gizmo_axis_indices_and_vectors() {
        assert_eq!(GizmoAxis::X.index(), 0);
        assert_eq!(GizmoAxis::Y.index(), 1);
        assert_eq!(GizmoAxis::Z.index(), 2);
        assert_eq!(GizmoAxis::from_index(0), Some(GizmoAxis::X));
        assert_eq!(GizmoAxis::from_index(1), Some(GizmoAxis::Y));
        assert_eq!(GizmoAxis::from_index(2), Some(GizmoAxis::Z));
        assert_eq!(GizmoAxis::from_index(3), None);
    }

    #[test]
    fn translate_arms_returns_empty_when_no_selection() {
        let mesh = cube();
        let selection = Selection::new();
        let eye = Point3::new(0.0, 0.0, 3.0);
        let arms = translate_arms(&mesh, EditMode::Vertex, &selection, eye, 600.0, None);
        assert!(arms.is_empty(), "unselected mesh must yield no gizmo arms");
    }

    #[test]
    fn translate_arms_returns_three_colored_arms_anchored_at_pivot() {
        let mesh = cube();
        let mut selection = Selection::new();
        selection.apply_click(Some(VertexId(0)), false, Selection::vertices_mut);
        let eye = Point3::new(0.0, 0.0, 3.0);
        let arms = translate_arms(&mesh, EditMode::Vertex, &selection, eye, 600.0, None);
        assert_eq!(arms.len(), 3, "must yield X, Y, and Z axes");

        let expected_pivot = mesh.vertex_position(VertexId(0));
        for arm in &arms {
            assert_eq!(
                arm.start, expected_pivot,
                "arm must start at selection pivot"
            );
        }

        assert_eq!(arms[0].color, COLOR_AXIS_X);
        assert_eq!(arms[1].color, COLOR_AXIS_Y);
        assert_eq!(arms[2].color, COLOR_AXIS_Z);

        // Arm ends must extend along each axis
        assert!(arms[0].end.x > expected_pivot.x);
        assert_eq!(arms[0].end.y, expected_pivot.y);
        assert_eq!(arms[0].end.z, expected_pivot.z);

        assert_eq!(arms[1].end.x, expected_pivot.x);
        assert!(arms[1].end.y > expected_pivot.y);
        assert_eq!(arms[1].end.z, expected_pivot.z);

        assert_eq!(arms[2].end.x, expected_pivot.x);
        assert_eq!(arms[2].end.y, expected_pivot.y);
        assert!(arms[2].end.z > expected_pivot.z);
    }

    #[test]
    fn translate_arms_highlights_active_axis_yellow() {
        let mesh = cube();
        let mut selection = Selection::new();
        selection.apply_click(Some(VertexId(0)), false, Selection::vertices_mut);
        let eye = Point3::new(0.0, 0.0, 3.0);

        let arms_highlight_y = translate_arms(
            &mesh,
            EditMode::Vertex,
            &selection,
            eye,
            600.0,
            Some(GizmoAxis::Y),
        );
        assert_eq!(arms_highlight_y[0].color, COLOR_AXIS_X);
        assert_eq!(arms_highlight_y[1].color, COLOR_AXIS_ACTIVE);
        assert_eq!(arms_highlight_y[2].color, COLOR_AXIS_Z);
    }

    #[test]
    fn selected_vertices_deduplicates_and_finds_components() {
        let mesh = cube();
        let mut selection = Selection::new();
        // Face 0 has 4 vertices
        selection.apply_click(Some(crate::mesh::FaceId(0)), false, Selection::faces_mut);
        let verts = selected_vertices(&mesh, EditMode::Face, &selection);
        assert_eq!(verts.len(), 4);

        // Object mode has all 8 cube vertices
        let mut obj_selection = Selection::new();
        obj_selection.apply_object_click(true, false);
        let obj_verts = selected_vertices(&mesh, EditMode::Object, &obj_selection);
        assert_eq!(obj_verts.len(), 8);
    }

    #[test]
    fn pick_gizmo_axis_finds_axis_within_pixel_tolerance() {
        let pivot = Point3::new(0.0, 0.0, 0.0);
        let eye = Point3::new(0.0, 0.0, 3.0);
        let viewport_height = 600.0;
        let arm_len = axis_world_length(pivot, eye, viewport_height);
        assert!(arm_len > 0.0);

        // Aim ray directly at the midpoint of the X axis arm (arm_len * 0.5, 0, 0)
        let target = Point3::new(arm_len * 0.5, 0.0, 0.0);
        let ray = Ray::new(eye, target - eye);

        let hit = pick_gizmo_axis(
            pivot,
            eye,
            viewport_height,
            &ray,
            GIZMO_PICK_TOLERANCE_PIXELS,
        );
        assert_eq!(hit, Some(GizmoAxis::X));

        // Aim ray directly at the midpoint of the Y axis arm (0, arm_len * 0.5, 0)
        let target_y = Point3::new(0.0, arm_len * 0.5, 0.0);
        let ray_y = Ray::new(eye, target_y - eye);

        let hit_y = pick_gizmo_axis(
            pivot,
            eye,
            viewport_height,
            &ray_y,
            GIZMO_PICK_TOLERANCE_PIXELS,
        );
        assert_eq!(hit_y, Some(GizmoAxis::Y));
    }

    #[test]
    fn pick_gizmo_axis_returns_none_when_ray_misses() {
        let pivot = Point3::new(0.0, 0.0, 0.0);
        let eye = Point3::new(0.0, 0.0, 3.0);
        let viewport_height = 600.0;

        // Aim far away from any axis
        let target = Point3::new(5.0, 5.0, 0.0);
        let ray = Ray::new(eye, target - eye);

        let hit = pick_gizmo_axis(
            pivot,
            eye,
            viewport_height,
            &ray,
            GIZMO_PICK_TOLERANCE_PIXELS,
        );
        assert_eq!(hit, None);
    }

    #[test]
    fn project_ray_onto_axis_exact() {
        let pivot = Point3::new(0.0, 0.0, 0.0);
        let axis = Vector3::new(1.0, 0.0, 0.0);
        let eye = Point3::new(0.0, 0.0, 5.0);

        // Ray passing through (2.5, 0.0, 0.0)
        let ray = Ray::new(eye, Point3::new(2.5, 0.0, 0.0) - eye);
        let t = project_ray_onto_axis(pivot, axis, &ray).expect("ray and axis are perpendicular");
        assert!((t - 2.5).abs() < 1e-4, "t must be 2.5, got {t}");
    }

    #[test]
    fn project_ray_onto_axis_parallel_returns_none() {
        let pivot = Point3::new(0.0, 0.0, 0.0);
        let axis = Vector3::new(1.0, 0.0, 0.0);
        // Ray pointing straight along +X
        let ray = Ray::new(Point3::new(-5.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));
        assert_eq!(project_ray_onto_axis(pivot, axis, &ray), None);
    }
}
