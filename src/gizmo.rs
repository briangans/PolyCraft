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
use crate::picking::world_per_pixel;
use crate::scene::EditMode;
use crate::selection::Selection;
use cgmath::{InnerSpace, Point3, Vector3};

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

/// The point a gizmo is anchored to, if there is a selection to anchor it to. Maya puts the move
/// handle at the selection's centroid, and at the object when the whole object is selected.
pub fn pivot(mesh: &HalfEdgeMesh, mode: EditMode, selection: &Selection) -> Option<Point3<f32>> {
    let mut sum = Vector3::new(0.0, 0.0, 0.0);
    let mut count = 0usize;
    let mut add = |point: Point3<f32>| {
        sum += Vector3::new(point.x, point.y, point.z);
        count += 1;
    };

    match mode {
        EditMode::Object => {
            if !selection.is_object_selected() {
                return None;
            }
            for vertex in 0..mesh.vertices.len() {
                add(mesh.vertex_position(VertexId(vertex)));
            }
        }
        EditMode::Vertex => {
            for vertex in selection.vertices().iter().copied() {
                add(mesh.vertex_position(vertex));
            }
        }
        EditMode::Edge => {
            for edge in selection.edges().iter().copied() {
                add(mesh.vertex_position(VertexId(edge.0)));
                add(mesh.vertex_position(VertexId(edge.1)));
            }
        }
        EditMode::Face => {
            for face in selection.faces().iter().copied() {
                for position in mesh.face_positions(face) {
                    add(position);
                }
            }
        }
    }

    if count == 0 {
        return None;
    }
    let f = count as f32;
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
/// with no selection. This is world-space geometry on purpose — the GPU's projection foreshortens
/// each arm by its angle to the view direction, so an axis pointing at the camera collapses to a
/// point and one broadside to the view spans the full 60 pixels. That is Maya's own gizmo look:
/// constant overall size at any zoom, orientation still readable.
pub fn translate_arms(
    mesh: &HalfEdgeMesh,
    mode: EditMode,
    selection: &Selection,
    camera_eye: Point3<f32>,
    viewport_height: f32,
) -> Vec<Arm> {
    let pivot = match pivot(mesh, mode, selection) {
        Some(pivot) => pivot,
        None => return Vec::new(),
    };
    let length = axis_world_length(pivot, camera_eye, viewport_height);
    if length <= 0.0 {
        return Vec::new();
    }
    AXES.iter()
        .map(|(axis, color)| Arm {
            start: pivot,
            end: pivot + *axis * length,
            color: *color,
        })
        .collect()
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
}
