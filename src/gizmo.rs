//! Transform gizmo math, independent of the GPU and of windowing.
//!
//! The gizmo is a set of three coloured axes drawn at the current selection and dragged to move
//! components along one of them. This module answers the questions the drawing and dragging steps
//! need, and nothing else: where the handle sits, how long its arms should be, and which axis a
//! pointer is nearest. Keeping it pure means all of it is testable without a window, and
//! matches the project rule that picking/selection/camera math never touch `wgpu` or `winit`.
//!
//! Three choices from the architecture shape the API here:
//! - The gizmo stays a constant size on screen (unlike the mesh). Its world length is therefore
//!   derived from a pixel length, using the same `world_per_pixel` the picker uses, so picking
//!   and drawing agree on what a "pixel" means.
//! - Orientation is a [`Frame`], never a hardcoded triple of world axes: the translate handle uses
//!   [`Frame::world`], the extrude handle builds one from the faces it will move. A world / object /
//!   component space toggle is therefore a choice of frame at one call site, not a second gizmo.
//! - The active tool is separate from `EditMode` (move/rotate/scale, bound to W/E/R in a later step). `gizmo::pivot` takes
//!   a mode and a selection because where the handle sits depends on what is selected, not on
//!   whether we are moving, rotating or scaling.

use crate::mesh::{HalfEdgeMesh, VertexId};
use crate::picking::{world_per_pixel, Ray};
use crate::scene::EditMode;
use crate::selection::Selection;
use cgmath::{vec3, InnerSpace, Point3, Vector3};

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

/// One gizmo arm: a line segment from the pivot along a direction, in that axis's colour.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Arm {
    pub start: Point3<f32>,
    pub end: Point3<f32>,
    pub color: [f32; 3],
}

/// The orientation a gizmo is drawn and dragged in: an origin plus three orthonormal directions.
///
/// [`Frame::world`] is the orientation the translate gizmo has always used. [`Frame::from_axis`]
/// takes it from the geometry instead - the extrude drag points its third axis along the face normal,
/// which is what turns the mouse into a depth control and keeps the handle visually explaining the
/// direction the faces will travel. Everything downstream - arm drawing, arm picking, projecting the
/// cursor - reads the frame rather than `GizmoAxis::unit_vector`, so a world/object/component space
/// toggle later becomes a choice of frame at one call site rather than a second copy of the gizmo.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Frame {
    pub origin: Point3<f32>,
    /// Right-handed orthonormal set, indexed by [`GizmoAxis::index`] so the slot colours line up.
    pub axes: [Vector3<f32>; 3],
}

impl Frame {
    /// The world axes, at `origin`.
    pub fn world(origin: Point3<f32>) -> Self {
        Frame {
            origin,
            axes: [
                GizmoAxis::X.unit_vector(),
                GizmoAxis::Y.unit_vector(),
                GizmoAxis::Z.unit_vector(),
            ],
        }
    }

    /// A frame whose third axis runs along `axis`, with the other two spanning the plane across it.
    ///
    /// `None` for a zero-length or non-finite axis: there is no plane across a vector that has no
    /// direction, and inventing one would draw a handle pointing somewhere the geometry never said.
    ///
    /// The helper axis is whichever world axis is furthest from `axis`, which is what keeps the first
    /// cross product well conditioned - crossing with a near-parallel vector would give a tangent of no
    /// length and, after normalising, a frame pointing at nothing.
    pub fn from_axis(origin: Point3<f32>, axis: Vector3<f32>) -> Option<Frame> {
        let magnitude = axis.magnitude();
        if !magnitude.is_finite() || magnitude <= 0.0 {
            return None;
        }
        let forward = axis / magnitude;
        let helper = if forward.z.abs() < 0.9 {
            Vector3::unit_z()
        } else {
            Vector3::unit_x()
        };
        let side = helper.cross(forward).normalize();
        let up = forward.cross(side);

        Some(Frame {
            origin,
            axes: [side, up, forward],
        })
    }

    /// The direction of one slot.
    pub fn direction(&self, axis: GizmoAxis) -> Vector3<f32> {
        self.axes[axis.index()]
    }

    /// Where the tip of one arm sits at world `length`.
    pub fn arm_end(&self, axis: GizmoAxis, length: f32) -> Point3<f32> {
        self.origin + self.direction(axis) * length
    }
}

/// Which orientation the transform gizmo is drawn and dragged in - Maya's "reference system" control.
///
/// `World` is the fixed coordinate frame the app has always used. `Component` takes the orientation from
/// the geometry under the selection, which is what makes a move out of a tilted face draggable without
/// hunting for the world axis that happens to point nearest it. It is the same arrangement the extrude
/// handle already uses, reached here from the selection instead of from the faces about to travel.
///
/// There is deliberately no `Object` variant. The document holds one mesh whose vertices are stored in
/// world coordinates - no transform, rotation or scale exists anywhere in `Scene` or `HalfEdgeMesh` - so
/// an object frame would be numerically identical to `World` while implying a hierarchy the app does not
/// have. Add the variant when an object transform lands, not before.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Space {
    /// The world axes, whatever is selected.
    #[default]
    World,
    /// An axis derived from the selected components, with the other two spanning across it.
    Component,
}

impl Space {
    /// Every space, in the order the label cycles.
    pub const ALL: [Space; 2] = [Space::World, Space::Component];

    /// Name for the panel button.
    pub fn label(self) -> &'static str {
        match self {
            Space::World => "World",
            Space::Component => "Component",
        }
    }

    /// The next space, so one control cycles rather than needing a picker.
    pub fn next(self) -> Space {
        match self {
            Space::World => Space::Component,
            Space::Component => Space::World,
        }
    }
}

/// The direction the selection itself points, for [`Space::Component`], or `None` when the geometry has
/// no single answer.
///
/// Per mode: a face gives its own normal; an edge gives the average of the normals of the faces that meet
/// at it; a vertex gives the average of the faces around it. The last two are the smooth-shading direction,
/// and they are what Maya's component space puts under the normal axis - for an edge it is the bisector of
/// the two faces, the direction a bevel would push it, which is why dragging it thickens a wall rather than
/// sliding the edge along its own length. `EditMode::Object` gives nothing, because there is no component
/// geometry to read a direction out of.
///
/// `None` also covers the case where the selection points two ways at once - two opposite faces, or two
/// edges whose bisectors run counter to each other - because their sum cancels, and pushing a selection by
/// a zero vector would move it nowhere while looking like a live gizmo. Callers fall back to world rather
/// than refusing to draw.
///
/// Reading the direction out of faces rather than out of the edge's own endpoints also drops an arbitrariness
/// the earlier version carried: [`EdgeId`] canonicalises its two vertices, so an edge-line axis had a sign
/// that came from vertex numbering and could flip between two visually identical meshes. A face normal has a
/// sign that comes from winding, which is the geometry's own.
///
/// A boundary edge has one face to average, so it tilts with that face; an edge a stale selection still
/// names has none, contributes nothing, and the frame falls back to world.
pub fn selection_axis(
    mesh: &HalfEdgeMesh,
    mode: EditMode,
    selection: &Selection,
) -> Option<Vector3<f32>> {
    let mut total = vec3(0.0, 0.0, 0.0);

    match mode {
        EditMode::Object => return None,
        EditMode::Face => {
            for &face in selection.faces() {
                if face.0 < mesh.faces.len() {
                    total += mesh.face_normal(face);
                }
            }
        }
        EditMode::Edge => {
            for &edge in selection.edges() {
                // Both faces that meet at the edge, or just the one a boundary edge has. `edge_faces`
                // answers (None, None) for an id the mesh no longer has, so a stale selection contributes
                // nothing and the frame falls back rather than pointing at a deleted edge.
                let (own, neighbour) = mesh.edge_faces(edge);
                for face in own.into_iter().chain(neighbour) {
                    total += mesh.face_normal(face);
                }
            }
        }
        EditMode::Vertex => {
            for &vertex in selection.vertices() {
                for &half_edge in &mesh.outgoing_half_edges(vertex) {
                    if let Some(face) = mesh.half_edges[half_edge.0].face {
                        total += mesh.face_normal(face);
                    }
                }
            }
        }
    }

    // `normalize` on a zero vector hands back NaN in every component, which would draw an arm pointing at
    // nothing and project the cursor onto a direction that does not exist.
    let magnitude = total.magnitude();
    if magnitude.is_finite() && magnitude > 0.0 {
        Some(total / magnitude)
    } else {
        None
    }
}

/// The frame the transform gizmo should use for the current selection and [`Space`], or `None` when there
/// is nothing to anchor it to.
///
/// `Space::Component` falls back to [`Frame::world`] wherever [`selection_axis`] has no answer, rather
/// than hiding the gizmo: a user who has switched to component space and then clicked a whole object
/// still needs to move it, and a vanished gizmo reads as a bug in the tool rather than as an empty
/// selection. The fallback is silent by design - the geometry simply has no preferred direction.
pub fn frame_for(
    mesh: &HalfEdgeMesh,
    mode: EditMode,
    selection: &Selection,
    space: Space,
) -> Option<Frame> {
    let pivot = pivot(mesh, mode, selection)?;
    if space == Space::World {
        return Some(Frame::world(pivot));
    }
    let frame = selection_axis(mesh, mode, selection)
        .and_then(|axis| Frame::from_axis(pivot, axis))
        .unwrap_or_else(|| Frame::world(pivot));
    Some(frame)
}

/// The three arms of a frame at world `length`, with `active_axis` highlighted in bright yellow.
///
/// This is the only place arm endpoints are computed, so drawing and picking cannot drift apart - a
/// gizmo drawn from one formula and picked with another silently moves the click target away from the
/// line the user aimed at.
pub fn arms(frame: &Frame, length: f32, active_axis: Option<GizmoAxis>) -> Vec<Arm> {
    GizmoAxis::ALL
        .iter()
        .map(|&axis| {
            let color = if active_axis == Some(axis) {
                COLOR_AXIS_ACTIVE
            } else {
                axis.color()
            };
            Arm {
                start: frame.origin,
                end: frame.arm_end(axis, length),
                color,
            }
        })
        .collect()
}

/// The three arms of `frame`, each long enough to measure `AXIS_LENGTH_PIXELS` at the frame's own depth.
///
/// Takes the frame rather than the selection because drawing and picking must agree exactly on where an
/// arm is: the two used to build their own world frame from the pivot independently, which was harmless
/// only while every gizmo was world-aligned. With [`Space::Component`] a drift between them would put the
/// click target on a different line than the one the user aimed at, and a component-space gizmo whose
/// arms tilt while the drag follows a world axis is worse still - it lies about where the geometry will
/// go. Both callers now take their frame from the same [`frame_for`].
///
/// Empty for a degenerate view where an arm would have no size.
pub fn frame_arms(
    frame: &Frame,
    camera_eye: Point3<f32>,
    viewport_height: f32,
    active_axis: Option<GizmoAxis>,
) -> Vec<Arm> {
    let length = axis_world_length(frame.origin, camera_eye, viewport_height);
    if length <= 0.0 {
        return Vec::new();
    }
    arms(frame, length, active_axis)
}

/// Tests if a ray passes within `tolerance_pixels` of any arm of `frame`.
///
/// Depth testing is intentionally skipped (Always-pass), matching how the gizmo is rendered on top
/// of the mesh. Returns the axis whose arm is closest to the ray within tolerance.
pub fn pick_frame_axis(
    frame: &Frame,
    length: f32,
    ray: &Ray,
    tolerance_pixels: f32,
    viewport_height: f32,
) -> Option<GizmoAxis> {
    if length <= 0.0 {
        return None;
    }

    let mut best: Option<(f32, GizmoAxis)> = None;

    for &axis in GizmoAxis::ALL.iter() {
        let arm_end = frame.arm_end(axis, length);

        let Some((offset, distance)) = crate::picking::ray_segment(ray, frame.origin, arm_end)
        else {
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
    use crate::mesh::FaceId;
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

    /// The arms the translate gizmo draws, assembled exactly as `State` assembles them: one call to
    /// [`frame_for`], then that frame's arms. The module used to offer a `translate_arms` that did both
    /// internally, which meant the renderer's pick path and its draw path each derived their own frame -
    /// harmless while every gizmo was world-aligned, and a misaligned handle the moment they are not.
    fn translate_arms(
        mesh: &HalfEdgeMesh,
        mode: EditMode,
        selection: &Selection,
        space: Space,
        eye: Point3<f32>,
        active: Option<GizmoAxis>,
    ) -> Vec<Arm> {
        let Some(frame) = frame_for(mesh, mode, selection, space) else {
            return Vec::new();
        };
        frame_arms(&frame, eye, VIEWPORT, active)
    }

    const VIEWPORT: f32 = 600.0;

    #[test]
    fn translate_arms_returns_empty_when_no_selection() {
        let mesh = cube();
        let selection = Selection::new();
        let eye = Point3::new(0.0, 0.0, 3.0);
        let arms = translate_arms(&mesh, EditMode::Vertex, &selection, Space::World, eye, None);
        assert!(arms.is_empty(), "unselected mesh must yield no gizmo arms");
    }

    #[test]
    fn translate_arms_returns_three_colored_arms_anchored_at_pivot() {
        let mesh = cube();
        let mut selection = Selection::new();
        selection.apply_click(Some(VertexId(0)), false, Selection::vertices_mut);
        let eye = Point3::new(0.0, 0.0, 3.0);
        let arms = translate_arms(&mesh, EditMode::Vertex, &selection, Space::World, eye, None);
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
            Space::World,
            eye,
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

    /// Asserts two directions coincide, in the tolerance that f32 vector maths deserves.
    fn same_direction(a: Vector3<f32>, b: Vector3<f32>) -> bool {
        (a - b).magnitude() < 1e-6
    }

    #[test]
    fn a_frame_from_an_axis_is_right_handed_and_points_where_the_geometry_said() {
        // The extrude drag reads its travel out of the third slot and draws its handle from the same
        // three directions, so a frame that skewed them would move faces somewhere the drawing never
        // pointed. Includes the axis-aligned cases, which are where a helper-axis shortcut usually breaks.
        let origin = Point3::new(1.0, 2.0, 3.0);
        for axis in [
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.5773, 0.5773, 0.5773),
            Vector3::new(3.0, -4.0, 12.0),
        ] {
            let frame =
                Frame::from_axis(origin, axis).expect("any finite non-zero axis defines a frame");

            assert_eq!(frame.origin, origin);
            assert!(
                same_direction(frame.direction(GizmoAxis::Z), axis.normalize()),
                "{axis:?} has to be the travel axis unchanged, got {:?}",
                frame.direction(GizmoAxis::Z)
            );
            for &slot in GizmoAxis::ALL.iter() {
                assert!(
                    (frame.direction(slot).magnitude() - 1.0).abs() < 1e-6,
                    "every slot must be unit length or the constant on-screen arm size goes wrong: {slot:?}"
                );
            }

            let [side, up, forward] = frame.axes;
            assert!(
                same_direction(side.cross(up), forward),
                "right-handed, so the third slot is the normal and not its negation for {axis:?}"
            );
            assert!(
                same_direction(forward.cross(side), up),
                "and the three must be mutually orthogonal for {axis:?}"
            );
        }
    }

    #[test]
    fn a_frame_refuses_an_axis_with_no_direction() {
        for axis in [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(f32::NAN, 1.0, 1.0),
            Vector3::new(0.0, f32::INFINITY, 0.0),
        ] {
            assert_eq!(
                Frame::from_axis(Point3::new(0.0, 0.0, 0.0), axis),
                None,
                "there is no plane across a vector with no direction: {axis:?}"
            );
        }
    }

    #[test]
    fn the_arms_of_a_frame_reach_along_its_own_directions() {
        // Drawing follows the frame, which is what lets the extrude handle be one tilted arm instead of a
        // second copy of the world gizmo bolted alongside it - and it is why the highlight has to be
        // resolved from the frame too, since a tilted handle is the only thing the drag offers.
        let frame =
            Frame::from_axis(Point3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 1.0)).unwrap();
        let drawn = arms(&frame, 2.0, Some(GizmoAxis::Z));

        assert_eq!(drawn.len(), GizmoAxis::ALL.len());
        for (slot, arm) in GizmoAxis::ALL.iter().zip(&drawn) {
            assert_eq!(arm.start, frame.origin);
            let tip = frame.origin + frame.direction(*slot) * 2.0;
            assert!(
                (arm.end - tip).magnitude() < 1e-6,
                "the {slot:?} arm must end one arm's length along its own direction"
            );
            let expected = if *slot == GizmoAxis::Z {
                COLOR_AXIS_ACTIVE
            } else {
                slot.color()
            };
            assert_eq!(arm.color, expected);
        }
    }

    #[test]
    fn picking_follows_the_frame_rather_than_the_world_axes() {
        // A tilted frame is what an armed extrude draws, so the arm under the cursor must be found from
        // that frame's directions. Picking the world axes instead would grab an axis that is not there.
        let eye = Point3::new(0.0, 0.0, 3.0);
        let viewport_height = 600.0;
        let frame =
            Frame::from_axis(Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 1.0)).unwrap();
        let length = axis_world_length(frame.origin, eye, viewport_height);
        assert!(length > 0.0);

        let midway = frame.arm_end(GizmoAxis::Z, length * 0.5);
        let ray = Ray::new(eye, midway - eye);
        assert_eq!(
            pick_frame_axis(
                &frame,
                length,
                &ray,
                GIZMO_PICK_TOLERANCE_PIXELS,
                viewport_height,
            ),
            Some(GizmoAxis::Z)
        );

        assert_eq!(
            pick_frame_axis(
                &frame,
                0.0,
                &ray,
                GIZMO_PICK_TOLERANCE_PIXELS,
                viewport_height,
            ),
            None,
            "an arm of no length is not a target"
        );
    }

    /// Picks the world-aligned gizmo the way `State` now does: take the frame, then pick that frame's
    /// arms. The `pick_gizmo_axis` this replaces was a second route to the same answer, and a second
    /// route is exactly where a component-space handle starts lying about where it is.
    fn pick_world_axis(
        pivot: Point3<f32>,
        eye: Point3<f32>,
        viewport_height: f32,
        ray: &Ray,
    ) -> Option<GizmoAxis> {
        let length = axis_world_length(pivot, eye, viewport_height);
        pick_frame_axis(
            &Frame::world(pivot),
            length,
            ray,
            GIZMO_PICK_TOLERANCE_PIXELS,
            viewport_height,
        )
    }

    #[test]
    fn a_world_gizmo_is_picked_within_pixel_tolerance() {
        let pivot = Point3::new(0.0, 0.0, 0.0);
        let eye = Point3::new(0.0, 0.0, 3.0);
        let viewport_height = 600.0;
        let arm_len = axis_world_length(pivot, eye, viewport_height);
        assert!(arm_len > 0.0);

        // Aim ray directly at the midpoint of the X axis arm (arm_len * 0.5, 0, 0)
        let target = Point3::new(arm_len * 0.5, 0.0, 0.0);
        let ray = Ray::new(eye, target - eye);

        assert_eq!(
            pick_world_axis(pivot, eye, viewport_height, &ray),
            Some(GizmoAxis::X)
        );

        // Aim ray directly at the midpoint of the Y axis arm (0, arm_len * 0.5, 0)
        let target_y = Point3::new(0.0, arm_len * 0.5, 0.0);
        let ray_y = Ray::new(eye, target_y - eye);

        assert_eq!(
            pick_world_axis(pivot, eye, viewport_height, &ray_y),
            Some(GizmoAxis::Y)
        );
    }

    #[test]
    fn a_world_gizmo_returns_none_when_the_ray_misses() {
        let pivot = Point3::new(0.0, 0.0, 0.0);
        let eye = Point3::new(0.0, 0.0, 3.0);
        let viewport_height = 600.0;

        // Aim far away from any axis
        let target = Point3::new(5.0, 5.0, 0.0);
        let ray = Ray::new(eye, target - eye);

        assert_eq!(pick_world_axis(pivot, eye, viewport_height, &ray), None);
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

    #[test]
    fn an_edge_aims_the_gizmo_between_the_faces_that_meet_at_it() {
        let mesh = cube();
        let mut selection = Selection::new();
        selection.apply_click(Some(crate::mesh::EdgeId(0, 1)), false, Selection::edges_mut);
        let axis = selection_axis(&mesh, EditMode::Edge, &selection).expect("a live edge aims");
        let bisector = Vector3::new(0.0, -1.0, 1.0) / 2.0f32.sqrt();
        assert!(
            (axis - bisector).magnitude() < 1e-6,
            "expected the bisector of the front and bottom faces, got {axis:?}"
        );
    }

    #[test]
    fn a_face_aims_the_gizmo_straight_out_of_itself() {
        let mesh = cube();
        let mut selection = Selection::new();
        selection.faces_mut().insert(FaceId(0));
        let axis = selection_axis(&mesh, EditMode::Face, &selection).expect("a live face aims");
        assert!(
            (axis - mesh.face_normal(FaceId(0))).magnitude() < 1e-6,
            "a lone face must aim along its own normal, got {axis:?}"
        );
    }

    #[test]
    fn a_vertex_aims_the_gizmo_the_way_the_faces_around_it_do() {
        let mesh = cube();
        let mut selection = Selection::new();
        selection.vertices_mut().insert(VertexId(0));
        let axis = selection_axis(&mesh, EditMode::Vertex, &selection).expect("a live vertex aims");
        // Three faces meet at a corner of the cube, so the average points out along the diagonal.
        let diagonal = Vector3::new(-1.0, -1.0, 1.0) / 3.0f32.sqrt();
        assert!(
            (axis - diagonal).magnitude() < 1e-5,
            "expected the corner's outward diagonal, got {axis:?}"
        );
    }

    #[test]
    fn a_selection_that_points_two_ways_has_no_axis_to_offer() {
        let mesh = cube();

        // Faces 0 and 2 are the front and the back, so their normals are exactly opposed.
        let mut faces = Selection::new();
        faces.faces_mut().insert(FaceId(0));
        faces.faces_mut().insert(FaceId(2));
        assert!(
            selection_axis(&mesh, EditMode::Face, &faces).is_none(),
            "opposite normals cancel, and a gizmo that pushed by a zero vector would move nothing"
        );

        // The two horizontal edges at opposite corners of the cube: front-bottom bisects to
        // (0, -1, 1) and back-top bisects to (0, 1, -1), so the pair sums to nothing.
        let mut edges = Selection::new();
        edges.edges_mut().insert(crate::mesh::EdgeId(0, 1));
        edges.edges_mut().insert(crate::mesh::EdgeId(6, 7));
        assert!(
            selection_axis(&mesh, EditMode::Edge, &edges).is_none(),
            "a selection whose bisectors cancel must fall back to world, not aim at zero"
        );
    }

    #[test]
    fn a_stale_selection_contributes_nothing_rather_than_pointing_at_nothing() {
        let mesh = cube();
        let mut selection = Selection::new();
        selection.faces_mut().insert(FaceId(99));
        selection.edges_mut().insert(crate::mesh::EdgeId(40, 41));
        selection.vertices_mut().insert(VertexId(99));

        for mode in [EditMode::Face, EditMode::Edge, EditMode::Vertex] {
            assert!(
                selection_axis(&mesh, mode, &selection).is_none(),
                "an id the mesh no longer has must yield no axis in {mode:?}, so the gizmo falls back \
                 to world instead of aiming at a deleted component"
            );
        }
    }
}
