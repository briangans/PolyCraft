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
use cgmath::{vec3, EuclideanSpace, InnerSpace, Point3, Vector3};

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
        .map(|&axis| Arm {
            start: frame.origin,
            end: frame.arm_end(axis, length),
            color: active_color(axis, active_axis),
        })
        .collect()
}

/// The colour one part of the handle draws in: bright yellow while its axis is hovered or dragged.
pub fn active_color(axis: GizmoAxis, active_axis: Option<GizmoAxis>) -> [f32; 3] {
    if active_axis == Some(axis) {
        COLOR_AXIS_ACTIVE
    } else {
        axis.color()
    }
}

/// The whole transform handle for one frame at one view: three arms, plus the three rings if `turning`.
///
/// Takes the frame rather than the selection because drawing and picking must agree exactly on where an
/// arm is: the two used to build their own world frame from the pivot independently, which was harmless
/// only while every gizmo was world-aligned. With [`Space::Component`] a drift between them would put the
/// click target on a different line than the one the user aimed at, and a component-space gizmo whose
/// arms tilt while the drag follows a world axis is worse still - it lies about where the geometry will
/// go. Both callers now take their frame from the same [`frame_for`].
///
/// A degenerate view - pivot at the eye, where [`axis_world_length`] answers zero - needs no special
/// case: every arm collapses onto its own start point and draws nothing, and the rings take the same zero
/// radius and answer empty. This is also the length the handle is picked at, so a squashed gizmo is a
/// gizmo that cannot be grabbed rather than one that grabs nothing where it appears to be.
pub fn handle(
    frame: &Frame,
    camera_eye: Point3<f32>,
    viewport_height: f32,
    active_axis: Option<GizmoAxis>,
    turning: bool,
) -> Vec<Arm> {
    let length = axis_world_length(frame.origin, camera_eye, viewport_height);
    let mut arms = arms(frame, length, active_axis);
    // The rings belong to the rotate handle alone. Drawn at the same radius as the arms are long, each
    // passes through the arm tips, so the handle reads as one object rather than two scales of thing
    // floating near each other - and the plane a ring lies in is the plane its drag measures.
    if turning {
        arms.extend(rings(frame, length, active_axis));
    }
    arms
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

/// Below this, a sight line counts as running along the plane rather than meeting it.
///
/// Matches the tolerance [`project_ray_onto_axis`] uses for the same hazard from the other side.
const PLANE_GRAZING_TOLERANCE: f32 = 1e-4;

/// Where `ray` meets the plane through `point` with the given `normal`.
///
/// The rotate drag needs this and cannot use [`project_ray_onto_axis`]: turning about an axis moves the
/// cursor *around* it, so a projection onto the axis itself reports that nothing happened however far the
/// handle is swung - and worse, reports whatever movement it does find as travel along the axis, which is
/// the gesture the tool must never make. The intersection with the plane across the axis is where the turn
/// actually reads.
///
/// `None` when the sight line runs *along* the plane, which is seeing the ring edge-on: the wheel has
/// collapsed to a line, its bearing cannot be read from a cursor position, and every pixel of travel slides
/// the hit arbitrarily far across the plane. Looking straight down the axis is the opposite case and the
/// best one - the ring is face-on - though a cursor aimed at the pivot itself then has no bearing, which
/// [`signed_angle`] refuses rather than reporting a turn of nothing.
///
/// Also `None` for the plane behind the eye, where the infinite line still meets it but the cursor points the
/// other way. Callers hold the angle they had rather than invent one, exactly as the extrude drag holds its
/// depth when the cursor runs along its travel axis.
pub fn project_ray_onto_plane(
    point: Point3<f32>,
    normal: Vector3<f32>,
    ray: &Ray,
) -> Option<Point3<f32>> {
    let magnitude = normal.magnitude();
    if !magnitude.is_finite() || magnitude <= 0.0 {
        return None;
    }
    let normal = normal / magnitude;

    let towards = ray.direction.dot(normal);
    if !towards.is_finite() || towards.abs() < PLANE_GRAZING_TOLERANCE {
        return None;
    }

    // Solving `((ray.origin + t * direction) - point) . normal == 0` for t. A negative root is the plane
    // behind the eye: the infinite *line* meets it, but the cursor points the other way, and a hit there
    // would turn the solid by an angle read from behind the camera.
    let distance = (point - ray.origin).dot(normal) / towards;
    if !distance.is_finite() || distance <= 0.0 {
        return None;
    }
    Some(ray.origin + ray.direction * distance)
}

/// The direction a cursor offset points, flattened into the plane across `axis`, as a unit vector.
///
/// `None` when the offset lies on the line of rotation, where it has no bearing to point in. Callers that
/// remember a cursor position between events need this distinction rather than just a zero: storing an
/// offset with no bearing as the reference for the next step would freeze the measurement from then on,
/// because every later step would be taken from somewhere that cannot be measured from.
///
/// Unit length so callers can hold it without carrying the cursor's distance from the pivot, which is not
/// part of a bearing and would make a drag that also pushed outward turn at a different rate.
pub fn bearing_across(offset: Vector3<f32>, axis: Vector3<f32>) -> Option<Vector3<f32>> {
    let magnitude = axis.magnitude();
    if !magnitude.is_finite() || magnitude <= 0.0 {
        return None;
    }
    let axis = axis / magnitude;

    // Flattening before measuring: the part of the offset that runs along the axis contributes nothing to
    // a bearing but does contribute to a length, which would let a cursor pushed far out along the axis
    // appear to have a bearing when it has none.
    let across = offset - axis * offset.dot(axis);
    let span = across.magnitude();
    if !span.is_finite() || span <= f32::EPSILON {
        return None;
    }
    Some(across / span)
}

/// The turn from `from` to `to` about `axis`, right-hand rule, in radians, in `(-pi, pi]`.
///
/// Both vectors are offsets from a common origin, so callers pass positions relative to the pivot. The
/// answer is measured *across* the axis - looking back down it, counter-clockwise positive - which is the
/// same convention [`crate::scene::Scene::rotate_vertices`] applies, and what makes a ring's colour and the
/// way it turns agree.
///
/// `None` when either vector has no span across the axis - it lies on the line of rotation, so there is no
/// bearing for a turn to have been measured between. Answering zero for that would report a measured turn
/// of nothing where in truth nothing was measured, and a drag that reads zeros keeps writing the mesh back
/// to where it started rather than holding still.
pub fn signed_angle(from: Vector3<f32>, to: Vector3<f32>, axis: Vector3<f32>) -> Option<f32> {
    let magnitude = axis.magnitude();
    if !magnitude.is_finite() || magnitude <= 0.0 {
        return None;
    }
    let axis = axis / magnitude;

    let across_from = bearing_across(from, axis)?;
    let across_to = bearing_across(to, axis)?;
    Some(
        across_from
            .cross(across_to)
            .dot(axis)
            .atan2(across_from.dot(across_to)),
    )
}

/// How far `positions` span along `direction`: the distance between the two that lie furthest apart
/// measured down the axis, or 0.0 when there is nothing to span.
///
/// This is the yardstick a scale drag works in. A factor has to be relative to something, and the only
/// honest choice is the geometry itself: a drag that doubles a wall should also double a face and not a
/// sliver. The alternative - reading the factor off how far the cursor sits from the pivot along the arm -
/// has a factor of infinity at the pivot and flips sign as the drag crosses it, both of which the user
/// discovers by losing the model.
///
/// A zero result needs flooring by the caller: a single vertex, or two corners level across the axis, span
/// nothing, and dividing by that turns a one-pixel move into an infinite stretch.
pub fn span_along(positions: &[Point3<f32>], direction: Vector3<f32>) -> f32 {
    let magnitude = direction.magnitude();
    if !magnitude.is_finite() || magnitude <= 0.0 || positions.is_empty() {
        return 0.0;
    }
    let direction = direction / magnitude;

    let mut lowest = f32::INFINITY;
    let mut highest = f32::NEG_INFINITY;
    for position in positions {
        let along = direction.dot(position.to_vec());
        if along.is_finite() {
            lowest = lowest.min(along);
            highest = highest.max(along);
        }
    }
    if !lowest.is_finite() || !highest.is_finite() {
        return 0.0;
    }
    highest - lowest
}

/// The stretch a scale drag has reached: 1.0 at the grab, 2.0 once the cursor has travelled along the arm
/// as far as the selection is wide.
///
/// The floor is deliberately not applied here - [`crate::scene::Scene::scale_vertices`] owns it, along with
/// the reason (a factor pushed through zero silently inverts every face it touches and no topology check
/// sees it), and a second copy here would only be a second thing to keep in step.
pub fn scale_factor(travel: f32, span: f32) -> f32 {
    if !travel.is_finite() || !span.is_finite() || span <= 0.0 {
        return 1.0;
    }
    1.0 + travel / span
}

/// How many chords a rotate ring is drawn with.
pub const RING_SEGMENTS: usize = 48;

/// The circle a rotate handle draws about one slot of `frame`, at `radius`, as `segments` chords.
///
/// Returns [`Arm`]s rather than inventing a shape: the gizmo pass is a line list, so a ring is drawn exactly
/// as an arm is and needs no second vertex format, buffer or pipeline. When the handle becomes lit geometry,
/// the torus generator replaces this function and nothing downstream notices.
///
/// Chords because the handle is roughly [`AXIS_LENGTH_PIXELS`] across on screen at any zoom, which makes
/// each of these a fraction of a pixel wide - the eye reads the polygon as a circle, and the vertex count
/// stays a rounding error in the gizmo buffer.
///
/// The chords sweep counter-clockwise about `axis` as their index rises, matching the right-hand rule
/// [`signed_angle`] measures and [`crate::scene::Scene::rotate_vertices`] applies, so a ring's sweep and the
/// turn it advertises cannot come apart. Empty for a radius of nothing.
pub fn ring(
    frame: &Frame,
    axis: GizmoAxis,
    radius: f32,
    segments: usize,
    active_axis: Option<GizmoAxis>,
) -> Vec<Arm> {
    if radius <= 0.0 || !radius.is_finite() || segments < 2 {
        return Vec::new();
    }

    // Two slots spanning the plane across `axis`, taken in the frame's own handedness so that
    // `side.cross(raise)` points back along that slot: a frame is right-handed by construction, so
    // consecutive slots cross into the one after them. Going the other way would draw a ring whose
    // increasing angle turns opposite to the angle the drag measures. The test is against the frame's own
    // slot rather than `axis.unit_vector()`, because in component space those are different vectors and
    // comparing the two would reject every tilted frame.
    let index = axis.index();
    let side = frame.axes[(index + 1) % 3];
    let raise = frame.axes[(index + 2) % 3];
    if side.cross(raise).dot(frame.axes[index]) <= 0.0 {
        return Vec::new();
    }

    let color = if active_axis == Some(axis) {
        COLOR_AXIS_ACTIVE
    } else {
        axis.color()
    };

    let step = std::f32::consts::TAU / segments as f32;
    let point_at = |i: usize| {
        let angle = step * i as f32;
        frame.origin + (side * angle.cos() + raise * angle.sin()) * radius
    };

    (0..segments)
        .map(|i| Arm {
            start: point_at(i),
            end: point_at((i + 1) % segments),
            color,
        })
        .collect()
}

/// The three rings a rotate handle draws, each passing through the arm tips at `radius`.
pub fn rings(frame: &Frame, radius: f32, active_axis: Option<GizmoAxis>) -> Vec<Arm> {
    GizmoAxis::ALL
        .iter()
        .flat_map(|&axis| ring(frame, axis, radius, RING_SEGMENTS, active_axis))
        .collect()
}

/// How many slices a solid tip is swept through.
///
/// A tip is only a few dozen pixels across on screen, so this is plenty to hide the facets and it
/// keeps the handle's triangle count at three cones' worth rather than three spheres'.
pub const TIP_SEGMENTS: usize = 16;

/// One flat shaded triangle of solid handle geometry.
///
/// The gizmo is on its way from lines to lit, occludable geometry, and a triangle is the smallest
/// thing a rasteriser can shade. It carries its own normal rather than having one derived per pass
/// because the flat face of a cone is the lighting - a smooth normal interpolated from the base
/// circle would wash the tip out into a blur.
///
/// Winding is counter-clockwise as seen from outside, which is what the mesh already uses, so the
/// handle and the model agree about which way a face points.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Triangle {
    pub vertices: [Point3<f32>; 3],
    pub normal: Vector3<f32>,
    pub color: [f32; 3],
}

/// The cone that caps one translate arm, its tip at the arm's end and its base back along the shaft.
///
/// `reach` is how far the cone runs back from the tip and `width` how far its base corners sit from
/// the axis; both are fractions of `length` at the call site, so the head scales with the handle
/// instead of drifting as the gizmo's world length changes with zoom.
///
/// Returns nothing when the axis projects to a point on screen. `length` comes from
/// [`axis_world_length`], which is zero for an arm pointing at or away from the eye, and a cone
/// built on such an axis would have its apex and base at the same place - a degenerate fan rather
/// than a handle.
pub fn arrowhead(
    frame: &Frame,
    axis: GizmoAxis,
    length: f32,
    reach: f32,
    width: f32,
    active_axis: Option<GizmoAxis>,
) -> Vec<Triangle> {
    let direction = frame.direction(axis);
    if length <= 0.0 || !length.is_finite() {
        return Vec::new();
    }
    let Some(crosswise) = Frame::from_axis(Point3::new(0.0, 0.0, 0.0), direction) else {
        return Vec::new();
    };

    let tip = frame.arm_end(axis, length);
    let base = tip - direction * reach;
    let color = active_color(axis, active_axis);

    let corners: Vec<Point3<f32>> = (0..TIP_SEGMENTS)
        .map(|step| {
            let tangent = crosswise.direction(GizmoAxis::X)
                * (std::f32::consts::TAU * step as f32 / TIP_SEGMENTS as f32).cos()
                + crosswise.direction(GizmoAxis::Y)
                    * (std::f32::consts::TAU * step as f32 / TIP_SEGMENTS as f32).sin();
            base + tangent * width
        })
        .collect();

    let mut triangles = Vec::with_capacity(TIP_SEGMENTS * 2);
    for step in 0..TIP_SEGMENTS {
        let next = (step + 1) % TIP_SEGMENTS;
        // Winding is counter-clockwise seen from outside, matching the mesh, and each normal is the
        // winding's own normal rather than a separately guessed one. If the two ever disagreed, the
        // face would be culled by the very lighting that shades it, and nothing would say so.
        triangles.push(face_triangle([tip, corners[step], corners[next]], color));
        // The cap closes the far end of the cone and faces away from the tip, so the head reads as a
        // solid rather than a paper bag seen through its own opening.
        triangles.push(face_triangle([corners[next], corners[step], base], color));
    }
    triangles
}

/// A triangle whose normal is the one its winding implies.
fn face_triangle(vertices: [Point3<f32>; 3], color: [f32; 3]) -> Triangle {
    let normal = (vertices[1] - vertices[0])
        .cross(vertices[2] - vertices[0])
        .normalize();
    Triangle {
        vertices,
        normal,
        color,
    }
}

/// The solid tips for every arm of a translate handle, as one flat list ready to upload.
pub fn arrowheads(frame: &Frame, length: f32, active_axis: Option<GizmoAxis>) -> Vec<Triangle> {
    GizmoAxis::ALL
        .iter()
        .flat_map(|&axis| {
            arrowhead(
                frame,
                axis,
                length,
                length * 0.22,
                length * 0.07,
                active_axis,
            )
        })
        .collect()
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
        handle(&frame, eye, VIEWPORT, active, false)
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
    // ---- Rotate / scale drag math --------------------------------------------------------------------

    /// Projects a ray aimed at `through` onto the plane through `origin` with `normal`.
    fn plane_hit(
        eye: Point3<f32>,
        origin: Point3<f32>,
        normal: Vector3<f32>,
        through: Point3<f32>,
    ) -> Point3<f32> {
        project_ray_onto_plane(origin, normal, &Ray::new(eye, through - eye))
            .expect("a sight line that is not parallel to the plane meets it")
    }

    #[test]
    fn a_ray_is_measured_where_it_meets_the_plane_across_the_axis() {
        let eye = Point3::new(0.0, 0.0, 3.0);
        let origin = Point3::new(0.0, 0.0, 0.0);
        let hit = plane_hit(eye, origin, Vector3::unit_z(), Point3::new(1.0, 0.0, 0.0));
        assert!(
            (hit - Point3::new(1.0, 0.0, 0.0)).magnitude() < 1e-4,
            "the sight line aims straight at that point, got {hit:?}"
        );

        // A pivot off the world origin, which is the normal case: the plane belongs to the pivot, not to
        // wherever the world happens to put it.
        let lifted = Point3::new(0.0, 0.0, 1.0);
        let hit = plane_hit(eye, lifted, Vector3::unit_z(), Point3::new(2.0, 0.0, 1.0));
        assert!(
            (hit - Point3::new(2.0, 0.0, 1.0)).magnitude() < 1e-4,
            "the plane must sit at the pivot, got {hit:?}"
        );
    }

    #[test]
    fn seeing_the_ring_edge_on_offers_no_turn_to_measure() {
        let eye = Point3::new(0.0, 0.0, 3.0);
        let origin = Point3::new(0.0, 0.0, 0.0);

        // The sight line running along the plane across the axis - here looking down -Z, which lies flat in
        // the plane across X, so the ring about X is seen edge on. The bearing of a point on a line seen
        // end-on is not readable, and the few pixels that do project land miles from the pivot.
        let edge_on = Ray::new(eye, Vector3::new(0.0, 0.0, -1.0));
        assert_eq!(
            project_ray_onto_plane(origin, Vector3::unit_x(), &edge_on),
            None
        );

        // The plane behind the eye: the infinite line still meets it, but the cursor points the other way,
        // and a turn read from behind the camera goes the wrong half of the way round.
        let away = Ray::new(eye, Vector3::new(0.0, 0.0, 1.0));
        assert_eq!(
            project_ray_onto_plane(origin, Vector3::unit_z(), &away),
            None
        );

        // And an axis with no direction has no plane across it at all.
        let anywhere = Ray::new(eye, Point3::new(1.0, 1.0, 0.0) - eye);
        assert_eq!(
            project_ray_onto_plane(origin, Vector3::new(0.0, 0.0, 0.0), &anywhere),
            None
        );
    }

    #[test]
    fn looking_down_the_axis_reads_the_turn_face_on() {
        // The case a rotate drag is most comfortable in, and worth stating because it is the one an
        // edge-on test tempts a reader to think is degenerate: sighting along the axis, the ring faces the
        // cursor and the screen centre maps exactly onto the pivot.
        let eye = Point3::new(0.0, 0.0, 3.0);
        let origin = Point3::new(0.0, 0.0, 0.0);
        let hit = plane_hit(eye, origin, Vector3::unit_z(), Point3::new(0.0, 0.0, 0.0));
        assert!(
            (hit - origin).magnitude() < 1e-5,
            "the pivot must project onto itself, got {hit:?}"
        );

        // And a quarter of the way round the ring reads as a quarter of a turn, which is the promise the
        // drawn chords make and the drag keeps.
        let start = plane_hit(eye, origin, Vector3::unit_z(), Point3::new(1.0, 0.0, 0.0));
        let end = plane_hit(eye, origin, Vector3::unit_z(), Point3::new(0.0, 1.0, 0.0));
        let angle = signed_angle(start - origin, end - origin, Vector3::unit_z())
            .expect("both cursors sit off the axis");
        assert!(
            (angle - std::f32::consts::FRAC_PI_2).abs() < 1e-4,
            "screen quarters must be ring quarters, got {angle}"
        );
    }

    #[test]
    fn a_quarter_turn_about_an_axis_is_positive_the_right_way_round() {
        // The sign is the whole difference between a ring that turns with its colour's axis and one that
        // turns against it, and only an explicit case can say which way is which: looking back down +Z,
        // +X sweeps counter-clockwise to +Y, so that pair must read +pi/2 and its reverse -pi/2.
        let quarter = std::f32::consts::FRAC_PI_2;
        let cases = [
            (Vector3::unit_x(), Vector3::unit_y(), GizmoAxis::Z, quarter),
            (Vector3::unit_y(), Vector3::unit_x(), GizmoAxis::Z, -quarter),
            (Vector3::unit_y(), Vector3::unit_z(), GizmoAxis::X, quarter),
            (Vector3::unit_z(), Vector3::unit_x(), GizmoAxis::Y, quarter),
        ];
        for (from, to, axis, expected) in cases {
            let angle = signed_angle(from, to, axis.unit_vector()).expect("two distinct bearings");
            assert!(
                (angle - expected).abs() < 1e-5,
                "{from:?} to {to:?} about {:?} should read {expected}, got {angle}",
                axis.unit_vector()
            );
        }
    }

    #[test]
    fn a_turn_is_measured_across_the_axis_rather_than_along_it() {
        // Both cursors sit far up the axis, so their raw dot product is dominated by the part that runs
        // along it. Measuring the vectors as they stand would report a shallow angle for a clean quarter
        // turn, which reads as a handle that has gone stiff rather than one that was measured wrong.
        let along = Vector3::new(0.0, 0.0, 40.0);
        let from = Vector3::unit_x() + along;
        let to = Vector3::unit_y() + along;
        let angle = signed_angle(from, to, Vector3::unit_z()).expect("both span the plane");
        assert!(
            (angle - std::f32::consts::FRAC_PI_2).abs() < 1e-5,
            "the height along the axis is not part of the turn, got {angle}"
        );
    }

    #[test]
    fn a_cursor_on_the_line_of_rotation_has_no_bearing_at_all() {
        // Answering zero here would look like a measured turn of nothing and write the mesh back to where it
        // started; `None` lets the drag hold still, which is what the view actually supports.
        assert_eq!(
            signed_angle(
                Vector3::new(0.0, 0.0, 7.0),
                Vector3::unit_x(),
                Vector3::unit_z()
            ),
            None
        );
        assert_eq!(
            signed_angle(
                Vector3::unit_x(),
                Vector3::unit_y(),
                Vector3::new(0.0, 0.0, 0.0)
            ),
            None
        );
    }

    #[test]
    fn a_ring_lies_in_the_plane_across_its_axis_and_closes_on_itself() {
        let frame = Frame::world(Point3::new(0.5, -0.5, 1.0));
        let radius = 0.75;
        for &axis in GizmoAxis::ALL.iter() {
            let chords = ring(&frame, axis, radius, RING_SEGMENTS, None);
            assert_eq!(chords.len(), RING_SEGMENTS);

            let normal = frame.direction(axis);
            for chord in &chords {
                for point in [chord.start, chord.end] {
                    let offset = point - frame.origin;
                    assert!(
                        (offset.magnitude() - radius).abs() < 1e-4,
                        "{axis:?} ring drifted to {:?}",
                        offset.magnitude()
                    );
                    assert!(
                        offset.dot(normal).abs() < 1e-4,
                        "{axis:?} ring tilted out of the plane it is meant to lie in"
                    );
                }
            }

            // Chords, not a scatter of points: each has to start where the last one ended, or the drawn
            // circle has a gap in it and reads as three arcs rather than one ring.
            for (index, chord) in chords.iter().enumerate() {
                let next = &chords[(index + 1) % chords.len()];
                assert!(
                    (chord.end - next.start).magnitude() < 1e-6,
                    "the {axis:?} ring broke between chord {index} and the next"
                );
            }
        }
    }

    #[test]
    fn a_ring_sweeps_the_way_its_axis_turns() {
        // The ring is a promise about direction, not only about a plane: it is drawn before the user has
        // moved, and the drag then measures with `signed_angle`. If the chords ran the other way round the
        // ring would advertise a turn and deliver its mirror, and nothing but this comparison would say so.
        let frame = Frame::world(Point3::new(0.0, 0.0, 0.0));
        for &axis in GizmoAxis::ALL.iter() {
            let normal = frame.direction(axis);
            let chords = ring(&frame, axis, 1.0, RING_SEGMENTS, None);
            let step = signed_angle(
                chords[0].start - frame.origin,
                chords[0].end - frame.origin,
                normal,
            )
            .expect("neighbouring ring points have a bearing between them");
            assert!(
                step > 0.0,
                "the {axis:?} ring's increasing index must turn the way {normal:?} turns, read {step}"
            );
        }
    }

    #[test]
    fn a_ring_takes_its_highlight_from_the_axis_being_held() {
        let frame = Frame::world(Point3::new(0.0, 0.0, 0.0));
        for &axis in GizmoAxis::ALL.iter() {
            let chords = ring(&frame, axis, 1.0, 8, Some(axis));
            assert!(
                chords.iter().all(|c| c.color == COLOR_AXIS_ACTIVE),
                "holding {axis:?} must light its own ring"
            );
            let other = GizmoAxis::ALL[(axis.index() + 1) % 3];
            let dimmed = ring(&frame, other, 1.0, 8, Some(axis));
            assert!(
                dimmed.iter().all(|c| c.color == other.color()),
                "holding {axis:?} must leave the {other:?} ring its own colour"
            );
        }
    }

    #[test]
    fn a_ring_of_no_radius_draws_nothing() {
        let frame = Frame::world(Point3::new(0.0, 0.0, 0.0));
        assert!(ring(&frame, GizmoAxis::X, 0.0, RING_SEGMENTS, None).is_empty());
        assert!(ring(&frame, GizmoAxis::X, -1.0, RING_SEGMENTS, None).is_empty());
        assert!(ring(&frame, GizmoAxis::X, f32::NAN, RING_SEGMENTS, None).is_empty());
        // Two points cannot be a circle, whatever the radius.
        assert!(ring(&frame, GizmoAxis::X, 1.0, 1, None).is_empty());
        assert_eq!(ring(&frame, GizmoAxis::X, 1.0, 2, None).len(), 2);
    }

    #[test]
    fn the_three_rings_cover_the_three_turn_planes() {
        let frame = Frame::from_axis(Point3::new(1.0, 1.0, 1.0), Vector3::new(1.0, 1.0, 1.0))
            .expect("a diagonal defines a component frame");
        let drawn = rings(&frame, 0.4, Some(GizmoAxis::Y));
        assert_eq!(drawn.len(), 3 * RING_SEGMENTS);
        assert_eq!(
            drawn
                .iter()
                .filter(|chord| chord.color == COLOR_AXIS_ACTIVE)
                .count(),
            RING_SEGMENTS,
            "exactly one of the three rings carries the highlight"
        );
    }

    #[test]
    fn span_along_measures_the_geometry_not_the_cursor() {
        let mesh = cube();
        let corners: Vec<Point3<f32>> = (0..mesh.vertices.len())
            .map(|i| mesh.vertex_position(VertexId(i)))
            .collect();

        for &axis in GizmoAxis::ALL.iter() {
            let span = span_along(&corners, axis.unit_vector());
            assert!(
                (span - 1.0).abs() < 1e-5,
                "the unit cube spans 1.0 along {:?}, measured {span}",
                axis.unit_vector()
            );
        }

        // Corner to corner, the span is the width of the shadow the points cast down the direction, so a
        // diagonal sees the cube sqrt(3) across.
        let diagonal = Vector3::new(1.0, 1.0, 1.0).normalize();
        assert!(
            (span_along(&corners, diagonal) - 3.0f32.sqrt()).abs() < 1e-5,
            "got {}",
            span_along(&corners, diagonal)
        );
    }

    #[test]
    fn a_selection_with_nothing_to_span_measures_nothing() {
        let mesh = cube();
        let one = vec![mesh.vertex_position(VertexId(0))];
        assert_eq!(span_along(&one, Vector3::unit_x()), 0.0);

        // Two corners level across the axis cast a shadow of no width - the case a scale must not divide by.
        let level = vec![Point3::new(-0.5, 0.5, 0.5), Point3::new(0.5, 0.5, 0.5)];
        assert_eq!(span_along(&level, Vector3::unit_z()), 0.0);
        assert_eq!(span_along(&[], Vector3::unit_x()), 0.0);
        assert_eq!(span_along(&level, Vector3::new(0.0, 0.0, 0.0)), 0.0);
    }

    #[test]
    fn the_rotate_handle_adds_three_rings_and_the_others_do_not() {
        let frame = Frame::world(Point3::new(0.0, 0.0, 0.0));
        let eye = Point3::new(0.0, 0.0, 3.0);

        let move_handle = handle(&frame, eye, VIEWPORT, None, false);
        assert_eq!(move_handle.len(), GizmoAxis::ALL.len());

        let rotate_handle = handle(&frame, eye, VIEWPORT, None, true);
        assert_eq!(
            rotate_handle.len(),
            GizmoAxis::ALL.len() + 3 * RING_SEGMENTS,
            "the rotate handle is the same three arms plus three rings, nothing else"
        );
        // The arms come first and unchanged, so the ring soup cannot shift where a tip is picked.
        assert_eq!(&rotate_handle[..GizmoAxis::ALL.len()], &move_handle[..]);
    }

    #[test]
    fn a_handle_at_the_eye_draws_nothing_worth_picking() {
        // Pivot at the eye is the view `axis_world_length` answers zero for. The old builder returned an
        // empty vec; this one returns arms of no length and no rings, which draws nothing and, more
        // importantly, contains no line a pick could land on more than a pixel from the pivot.
        let origin = Point3::new(0.0, 0.0, 0.0);
        let drawn = handle(&Frame::world(origin), origin, VIEWPORT, None, true);
        assert_eq!(drawn.len(), GizmoAxis::ALL.len());
        for arm in &drawn {
            assert_eq!(arm.start, arm.end, "an arm at zero length is a point");
            assert!((arm.end - origin).magnitude() < 1e-6);
        }
    }

    #[test]
    fn every_handle_vertex_is_finite_at_a_sharp_corner_of_the_view() {
        // The one thing the gizmo buffer cannot survive is a NaN, which wgpu takes as licence to draw
        // garbage across the viewport rather than as an error. A pivot on the eye axis and a frame tilted
        // by hand are the two ways the arithmetic gets nearest to dividing by something small.
        let eye = Point3::new(0.0, 0.0, 0.001);
        let frames = [
            Frame::world(Point3::new(0.0, 0.0, 0.0)),
            Frame::world(Point3::new(0.0, 0.0, 0.001)),
            Frame::from_axis(Point3::new(0.4, -0.2, 0.1), Vector3::new(1.0, 1.0, 1.0))
                .expect("a diagonal defines a frame"),
        ];
        for frame in &frames {
            for turning in [false, true] {
                for arm in handle(frame, eye, VIEWPORT, Some(GizmoAxis::X), turning) {
                    for point in [arm.start, arm.end] {
                        assert!(
                            point.x.is_finite() && point.y.is_finite() && point.z.is_finite(),
                            "handle geometry went non-finite at {point:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_scale_rate_doubles_once_the_drag_has_matched_the_span() {
        assert_eq!(scale_factor(0.0, 2.0), 1.0);
        assert!((scale_factor(2.0, 2.0) - 2.0).abs() < 1e-6);
        assert!((scale_factor(-1.0, 2.0) - 0.5).abs() < 1e-6);

        // Nothing to divide by, so the drag cannot state a rate. Returning 1.0 leaves the geometry where it
        // was rather than stretching it to infinity, which is the difference between a stuck handle and a
        // lost model.
        assert_eq!(scale_factor(1.0, 0.0), 1.0);
        assert_eq!(scale_factor(f32::NAN, 2.0), 1.0);
        assert_eq!(scale_factor(1.0, f32::INFINITY), 1.0);
    }

    /// The length a tip is built at in these tests: what a pivot at the origin sees from a typical eye.
    fn tip_length() -> f32 {
        axis_world_length(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 3.0),
            VIEWPORT,
        )
    }

    #[test]
    fn a_tip_is_a_closed_cone_wrapping_its_arm() {
        let frame = Frame::world(Point3::new(0.0, 0.0, 0.0));
        let length = tip_length();
        let tip = arrowhead(
            &frame,
            GizmoAxis::Z,
            length,
            length * 0.22,
            length * 0.07,
            None,
        );

        // One side per slice plus one cap per slice. The cap is what makes the head a solid instead of
        // an open bag whose inside the lighting has no answer for.
        assert_eq!(tip.len(), TIP_SEGMENTS * 2);
        for face in &tip {
            for point in face.vertices {
                assert!(
                    point.x.is_finite() && point.y.is_finite() && point.z.is_finite(),
                    "tip geometry went non-finite at {point:?}"
                );
            }
            assert!(
                (face.normal.magnitude() - 1.0).abs() < 1e-5,
                "a non-unit normal scales the lighting term it is fed into"
            );
        }
    }

    #[test]
    fn every_tip_face_winds_the_way_its_normal_points() {
        // The mesh pipeline culls back faces with `FrontFace::Ccw`. A tip whose winding and normal were
        // derived independently could disagree between the two, and the face would be culled by the very
        // lighting that shades it - an invisible handle with nothing in the log to explain it.
        let frame = Frame::world(Point3::new(0.4, -0.2, 0.1));
        let length = tip_length();
        for axis in GizmoAxis::ALL {
            for face in arrowhead(&frame, axis, length, length * 0.22, length * 0.07, None) {
                let wound = (face.vertices[1] - face.vertices[0])
                    .cross(face.vertices[2] - face.vertices[0])
                    .normalize();
                assert!(
                    wound.dot(face.normal) > 0.99,
                    "tip face wound clockwise against its own normal"
                );
            }
        }
    }

    #[test]
    fn a_tip_shrinks_towards_the_axis_it_caps_rather_than_across_it() {
        let frame = Frame::world(Point3::new(0.0, 0.0, 0.0));
        let length = tip_length();
        let reach = length * 0.22;
        let width = length * 0.07;
        let tip = arrowhead(&frame, GizmoAxis::Z, length, reach, width, None);

        let apex = frame.arm_end(GizmoAxis::Z, length);
        let axis: Vector3<f32> = frame.direction(GizmoAxis::Z);
        assert!(
            tip.iter().any(|face| face.vertices.contains(&apex)),
            "the cone's point must sit exactly on the arm's tip, or the head floats off the shaft"
        );

        for face in &tip {
            for point in &face.vertices {
                let along = (*point - apex).dot(axis);
                assert!(
                    along <= 1e-5 && along >= -(reach + 1e-5),
                    "a tip vertex escaped the span of the arm it caps"
                );
                let across = ((*point - apex) - axis * along).magnitude();
                assert!(
                    across <= width + 1e-5,
                    "a tip bulged wider than the cone it was asked for"
                );
            }
        }
    }

    #[test]
    fn a_tip_survives_a_component_frame_tilted_off_the_world_axes() {
        // Component space is the whole reason the gizmo is a `Frame` and not three world vectors, so a
        // cone built from `frame.direction` has to tilt with it rather than standing up regardless.
        let direction = Vector3::new(1.0, 1.0, 1.0);
        let frame =
            Frame::from_axis(Point3::new(0.4, -0.2, 0.1), direction).expect("diagonal frame");
        let length = tip_length();
        let tip = arrowhead(
            &frame,
            GizmoAxis::Z,
            length,
            length * 0.22,
            length * 0.07,
            None,
        );

        let axis = frame.direction(GizmoAxis::Z);
        let apex = frame.arm_end(GizmoAxis::Z, length);
        assert!(
            tip.iter().any(|face| face.vertices.contains(&apex)),
            "a tilted frame must still put the point of the cone on its own arm's tip"
        );

        // The cap's normal is the cone's axis, so it is the cheapest read on whether the cone tilted.
        let cap_normal = tip[1].normal;
        assert!(
            (cap_normal.dot(-axis) - 1.0).abs() < 1e-5,
            "the cap faces {cap_normal:?}, not back along the frame's axis"
        );
    }

    #[test]
    fn a_tip_on_an_axis_that_points_at_the_eye_draws_nothing() {
        // `axis_world_length` answers zero for an arm pointing at or away from the eye. A cone built on
        // that length would have its apex and base at the same place: a fan of degenerate triangles that
        // the rasteriser still shades, producing a smear where a handle should have vanished.
        let frame = Frame::world(Point3::new(0.0, 0.0, 0.0));
        assert!(arrowhead(&frame, GizmoAxis::X, 0.0, 0.1, 0.02, None).is_empty());
        assert!(arrowhead(&frame, GizmoAxis::X, -1.0, 0.1, 0.02, None).is_empty());
        assert!(arrowhead(&frame, GizmoAxis::X, f32::NAN, 0.1, 0.02, None).is_empty());
        assert!(arrowheads(&frame, 0.0, None).is_empty());
    }

    #[test]
    fn a_tip_lights_up_with_the_axis_it_caps() {
        let frame = Frame::world(Point3::new(0.0, 0.0, 0.0));
        let length = tip_length();
        let heads = arrowheads(&frame, length, Some(GizmoAxis::Y));
        assert_eq!(heads.len(), GizmoAxis::ALL.len() * TIP_SEGMENTS * 2);

        // The heads come out in axis order, so the highlight has to move with the arm and not with the
        // index - the same rule the arms follow, which is why both read `active_color`.
        for (index, axis) in GizmoAxis::ALL.iter().enumerate() {
            let head = &heads[index * TIP_SEGMENTS * 2];
            assert_eq!(head.color, active_color(*axis, Some(GizmoAxis::Y)));
        }
    }
}
