//! The editable document: the mesh being modeled, the current edit mode, and what is selected.
//!
//! Anything that changes the mesh or the selection goes through here, which flags the document as
//! dirty. `State::render` checks that flag once per frame and re-uploads the GPU buffers.

use crate::gizmo::Space;
use crate::history::HistoryStack;
use crate::mesh::{EdgeId, FaceId, HalfEdgeMesh, TopologyErrors};
use crate::picking::{self, Ray};
use crate::selection::Selection;
use cgmath::{InnerSpace, Point3, Vector3};
use std::collections::HashSet;

pub use crate::edit_mode::EditMode;
pub use crate::tool::Tool;

/// Click tolerance for components, in pixels. Measured on screen, so it feels the same at any zoom.
pub const VERTEX_PICK_RADIUS_PIXELS: f32 = 8.0;
pub const EDGE_PICK_RADIUS_PIXELS: f32 = 6.0;

/// How many actions `undo` can walk back before the oldest ones are dropped. Meshes here are small
/// (a snapshot is a few hundred KB at most), so they stay cheap enough to keep a deep stack.
pub const HISTORY_DEPTH: usize = 64;

/// The depth below which an extrude preview counts as "has not started yet", in model units.
///
/// [`crate::mesh_edit`] refuses an extrude whose offset has no length, because every wall it would grow
/// collapses onto its own base edge - so without this, a cursor sitting exactly on the handle would ask
/// for that refusal every frame and fill the command echo with an error for doing nothing. Matching the
/// primitive's own floor keeps the two from disagreeing about when movement begins.
const MIN_EXTRUDE_DEPTH: f32 = 1e-6;

/// The handle an armed extrude will travel along, handed back so the caller can project the cursor
/// without holding the document: where the faces sit, and the unit direction they will move.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ExtrudeGuide {
    pub pivot: Point3<f32>,
    pub direction: Vector3<f32>,
}

/// What a committed extrude moved, for the command echo.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ExtrudeCommit {
    /// How many faces the user had selected.
    pub faces: usize,
    /// How many walls the extrude grew - zero when the depth never left the surface.
    pub walls: usize,
    /// Whether a step landed in history, which is false exactly when `walls` is zero.
    pub recorded: bool,
}

/// What a completed edge loop cut, for the command echo.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct EdgeLoopCommit {
    /// How many faces the ring crossed, which is also how many new edges it laid down.
    pub faces_cut: usize,
    /// How many vertices were dropped into the ring's edges, one per edge the loop opened.
    pub new_vertices: usize,
}

/// An extrude awaiting the depth the mouse is giving it.
#[derive(Clone, Debug)]
struct ActiveExtrude {
    /// The document as it was the instant the extrude was armed. Every preview is measured from here, so
    /// the depth is a position rather than a running total of mouse movement.
    snapshot: HalfEdgeMesh,
    /// Frozen with the guide, so the region cannot drift if the selection changes mid-drag.
    faces: Vec<FaceId>,
    direction: Vector3<f32>,
    /// Walls in the preview currently installed, for the command echo.
    walls: usize,
}

/// The state undo/redo version: the geometry plus what was selected on it.
///
/// `EditMode` and `Tool` are deliberately left out the way Maya leaves them out: switching mode or
/// tool is a view/toolbox state, not a document edit, so `Ctrl+Z` never drags the user back into a
/// mode they have since left. See `Scene::restore`.
#[derive(Clone, Debug, PartialEq)]
pub struct DocumentSnapshot {
    mesh: HalfEdgeMesh,
    selection: Selection,
}

pub struct Scene {
    mesh: HalfEdgeMesh,
    dirty: bool,
    mode: EditMode,
    tool: Tool,
    /// Which orientation the transform gizmo is drawn and dragged in. Maya's "reference system".
    space: Space,
    selection: Selection,
    history: HistoryStack<DocumentSnapshot>,
    /// An extrude waiting for the mouse to give it a depth. `None` the rest of the time.
    extrude: Option<ActiveExtrude>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            mesh: HalfEdgeMesh::create_cube(),
            dirty: false,
            mode: EditMode::Object,
            tool: Tool::default(),
            space: Space::default(),
            selection: Selection::new(),
            history: HistoryStack::new(HISTORY_DEPTH),
            extrude: None,
        }
    }

    pub fn mesh(&self) -> &HalfEdgeMesh {
        &self.mesh
    }

    pub fn mode(&self) -> EditMode {
        self.mode
    }

    /// Switches modes, dropping the selections the new mode cannot hold (as Maya does).
    ///
    /// An armed extrude is abandoned rather than carried: it moves the faces the old mode selected, and
    /// those ids are gone the moment the selection is. Leaving it open would let a later click settle a
    /// lump the user can no longer point at, previewing under a mode that has no handle on it.
    pub fn set_mode(&mut self, mode: EditMode) {
        if mode == self.mode {
            return;
        }

        self.abandon_extrude();
        self.mode = mode;
        self.selection.retain_mode(mode);
        self.dirty = true;
    }

    pub fn tool(&self) -> Tool {
        self.tool
    }

    /// Sets the active manipulation or selection tool. Does not alter the current selection.
    ///
    /// Abandons an armed extrude the way [`Scene::set_mode`] does, for a different reason: while the drag
    /// is open its travel arm owns the handle and the translate gizmo is not drawn, so taking a tool
    /// change without closing it would look like a button that does nothing.
    pub fn set_tool(&mut self, tool: Tool) {
        self.abandon_extrude();
        self.tool = tool;
    }

    /// Which orientation the transform gizmo is drawn and dragged in.
    pub fn space(&self) -> Space {
        self.space
    }

    /// Chooses the gizmo's reference system.
    ///
    /// Unlike [`Scene::set_mode`] and [`Scene::set_tool`] this does not abandon an armed extrude, because
    /// it cannot strand one: an extrude travels along the normal of the faces it is moving, which no space
    /// changes. Nor does it disturb a translate drag already in flight - `DragState` keeps the frame it was
    /// grabbed with, so the arms can be redrawn in the new orientation without the geometry lurching
    /// sideways mid-pull. The new space takes effect on the next grab.
    ///
    /// The document is not marked dirty: no vertex moved, and the handle is re-uploaded every frame anyway.
    pub fn set_space(&mut self, space: Space) {
        self.space = space;
    }

    /// Steps World -> Component -> World, so a single control cycles the pair rather than needing a picker.
    pub fn cycle_space(&mut self) {
        let next = self.space.next();
        self.set_space(next);
    }

    pub fn selection(&self) -> &Selection {
        &self.selection
    }

    /// How many components of the current mode are selected.
    pub fn selected_count(&self) -> usize {
        self.selection.count(self.mode)
    }

    /// Returns the unique set of vertices affected by the current selection in the active mode.
    pub fn selected_vertices(&self) -> Vec<crate::mesh::VertexId> {
        crate::gizmo::selected_vertices(&self.mesh, self.mode, &self.selection)
    }

    /// Moves the specified vertices to `initial_position + delta`. Flags the document dirty so
    /// GPU vertex and overlay buffers re-upload next frame.
    pub fn translate_vertices(
        &mut self,
        vertices: &[(crate::mesh::VertexId, cgmath::Point3<f32>)],
        delta: cgmath::Vector3<f32>,
    ) {
        if vertices.is_empty() {
            return;
        }
        for &(v_id, initial_pos) in vertices {
            if v_id.0 < self.mesh.vertices.len() {
                self.mesh.vertices[v_id.0].position = initial_pos + delta;
            }
        }
        self.dirty = true;
    }

    /// Whether an extrude is available: faces are what extrude moves, and it needs at least one.
    ///
    /// Shared by the hotkey seam and the Tools panel so the button greys out for exactly the reason the
    /// key does nothing, rather than looking live and then failing quietly.
    pub fn can_extrude(&self) -> bool {
        self.mode == EditMode::Face
            && self
                .selection
                .faces()
                .iter()
                .any(|&face| face.0 < self.mesh.faces.len())
    }

    /// Whether an extrude drag is open, awaiting its depth.
    pub fn is_extruding(&self) -> bool {
        self.extrude.is_some()
    }

    /// Arms an extrude of the selected faces and returns the handle it will travel along: the pivot the
    /// faces sit around and the unit direction, which is the average of their own normals.
    ///
    /// Nothing moves yet. The depth arrives later from [`Scene::preview_extrude`], because the user sets
    /// it by moving the mouse - so the direction is decided now, while the selection is definitely what
    /// the user pointed at, and stays fixed for the gesture even as the preview moves the faces.
    ///
    /// Opening the history step here is what makes the whole drag one undo: [`Scene::begin_action`]
    /// records the document as it is at this instant, before a single wall exists.
    pub fn begin_extrude_drag(&mut self) -> Result<ExtrudeGuide, TopologyErrors> {
        if self.extrude.is_some() {
            return Err(vec![
                "an extrude is already in progress; click to set its depth or press Escape to abandon it"
                    .to_string(),
            ]);
        }

        let (faces, direction) = self.extrude_request()?;
        let pivot = self.extrude_pivot();

        self.begin_action(&extrude_label(faces.len()));
        self.extrude = Some(ActiveExtrude {
            snapshot: self.mesh.clone(),
            faces,
            direction,
            walls: 0,
        });

        Ok(ExtrudeGuide { pivot, direction })
    }

    /// Installs the extrude at `depth` units along the armed direction, positive being the way the faces
    /// point, and returns how many walls it has grown.
    ///
    /// Every preview is measured from the snapshot taken when the extrude was armed, never from the mesh
    /// as it currently stands, so depth is a position rather than a delta: bringing the cursor back always
    /// brings the faces back, however far they have been pushed. A depth too small to move anything shows
    /// the document exactly as it was before the extrude was armed.
    ///
    /// A depth that produces geometry the validator refuses leaves the *previous* preview standing. That
    /// is deliberate: snapping back to the snapshot would blink the whole lump out of existence every time
    /// the cursor crossed an impossible depth, which reads as a bug rather than as the limit it is.
    pub fn preview_extrude(&mut self, depth: f32) -> Result<usize, TopologyErrors> {
        let mut drag = match self.extrude.take() {
            Some(drag) => drag,
            None => return Err(vec!["no extrude is in progress".to_string()]),
        };

        let mut candidate = drag.snapshot.clone();
        let outcome = if depth.is_finite() && depth.abs() >= MIN_EXTRUDE_DEPTH {
            candidate
                .extrude_faces(&drag.faces, drag.direction * depth)
                .map(|shape| shape.side_faces.len())
        } else {
            Ok(0)
        };

        match outcome {
            Ok(walls) => {
                drag.walls = walls;
                self.mesh = candidate;
                self.dirty = true;
                self.extrude = Some(drag);
                Ok(walls)
            }
            Err(errors) => {
                self.extrude = Some(drag);
                Err(errors)
            }
        }
    }

    /// Settles the armed extrude at whatever depth was last previewed, recording one history step if the
    /// faces actually moved. Returns what moved for the command echo, or `None` when nothing was armed.
    ///
    /// The selection is left exactly as it was, which is the settled precedent for where a command leaves
    /// the highlight: the faces the user pointed at stay selected, and their ids remain valid because the
    /// primitive appends and re-points rather than replacing them. An extrude that never moved anything
    /// lands no step at all, so a stray keypress cannot pollute the undo chain.
    pub fn commit_extrude(&mut self) -> Option<ExtrudeCommit> {
        let drag = self.extrude.take()?;
        let recorded = self.end_action();
        Some(ExtrudeCommit {
            faces: drag.faces.len(),
            walls: drag.walls,
            recorded,
        })
    }

    /// Gives up the armed extrude, putting the document back exactly as it was when it was armed. Returns
    /// false when there was nothing to abandon.
    pub fn abandon_extrude(&mut self) -> bool {
        let Some(drag) = self.extrude.take() else {
            return false;
        };
        self.mesh = drag.snapshot;
        self.dirty = true;
        self.cancel_action();
        true
    }

    /// The faces to move and the direction to move them, or every reason the selection cannot extrude.
    ///
    /// The direction is the average of the selection's own normals and nothing else, always recomputed
    /// from the live mesh: averaging is what sends a two-face corner diagonally out the way Maya does, and
    /// refusing when the normals cancel is what stops opposite faces being pushed into each other.
    fn extrude_request(&self) -> Result<(Vec<FaceId>, Vector3<f32>), TopologyErrors> {
        if !self.can_extrude() {
            return Err(vec![
                "extrude moves faces, so it needs at least one face selected in Face mode"
                    .to_string(),
            ]);
        }

        let mut faces: Vec<FaceId> = self
            .selection
            .faces()
            .iter()
            .copied()
            .filter(|&face| face.0 < self.mesh.faces.len())
            .collect();
        faces.sort_by_key(|face| face.0);

        let Some(direction) = extrude_direction(&self.mesh, &faces) else {
            return Err(vec![
                "the selected faces point in opposite directions, so there is no single way to push them; \
                 extrude them one at a time"
                    .to_string(),
            ]);
        };

        Ok((faces, direction))
    }

    /// Where the extrude handle sits: the centroid of the selected faces, which is where the user is
    /// looking and the point the depth is measured from. The fallback covers a selection that was just
    /// emptied underneath us, which the callers already refuse to extrude.
    fn extrude_pivot(&self) -> Point3<f32> {
        crate::gizmo::pivot(&self.mesh, self.mode, &self.selection)
            .unwrap_or(Point3::new(0.0, 0.0, 0.0))
    }

    /// Whether an edge loop can be inserted here: one, and only one, edge selected in Edge mode.
    ///
    /// Shared by the hotkey seam and the Tools panel so the button greys out for the same reason the key
    /// does nothing. One at a time is not shyness about the primitive - it is that two selected edges can
    /// name the same ring, and the second would arrive to find its edges already split by the first. Which
    /// of them the user meant is not something this layer can decide, so it asks instead of guessing.
    pub fn can_insert_edge_loop(&self) -> bool {
        self.mode == EditMode::Edge && self.live_edges().len() == 1
    }

    /// Runs Maya's *Insert Edge Loop* through the single selected edge: a ring of new edges, one across
    /// each quad face the loop reaches, splitting every face it crosses. One undoable step.
    ///
    /// The new ring is left selected, which is both what Maya does and what makes pressing again useful -
    /// the ring under the cursor is what the next cut is measured from. Returns what was cut for the
    /// command echo, or every reason nothing happened, the document untouched.
    pub fn insert_edge_loop(&mut self) -> Result<EdgeLoopCommit, TopologyErrors> {
        let seed = self.edge_loop_seed()?;

        self.begin_action("Insert Edge Loop");
        let mut mesh = self.mesh.clone();
        match mesh.insert_edge_loop(seed) {
            Ok(cut) => {
                let commit = EdgeLoopCommit {
                    faces_cut: cut.new_edges.len(),
                    new_vertices: cut.new_vertices.len(),
                };
                self.mesh = mesh;
                self.dirty = true;
                self.selection.edges_mut().clear();
                self.selection.edges_mut().extend(cut.new_edges);
                self.end_action();
                Ok(commit)
            }
            Err(errors) => {
                // Nothing reached the document, so there is no step worth recording - and an action left
                // open would file the *next* edit under this one.
                self.cancel_action();
                Err(errors)
            }
        }
    }

    /// The one edge to run a loop through, or why the selection cannot name one.
    fn edge_loop_seed(&self) -> Result<EdgeId, TopologyErrors> {
        let edges = self.live_edges();
        match edges.len() {
            0 if self.mode != EditMode::Edge => Err(vec![
                "an edge loop cuts through edges, so switch to Edge mode and select the edge to run it \
                 through"
                    .to_string(),
            ]),
            0 => Err(vec![
                "an edge loop needs one edge selected; click an edge of the mesh first".to_string(),
            ]),
            1 => Ok(edges[0]),
            count => Err(vec![format!(
                "an edge loop runs through one edge at a time and {count} are selected; two of them can \
                 name the same ring, so narrow the selection to the edge you want cut"
            )]),
        }
    }

    /// The selected edges that still name a pair of vertices on the mesh. A selection survives an undo
    /// and a mesh load, so an id from an earlier document can be handed back here; running a ring off a
    /// dead edge would report a cut the user cannot see.
    fn live_edges(&self) -> Vec<EdgeId> {
        self.selection
            .edges()
            .iter()
            .copied()
            .filter(|&edge| edge.0 < self.mesh.vertices.len() && edge.1 < self.mesh.vertices.len())
            .collect()
    }

    /// Selects whatever the ray hits, in the current mode. `additive` keeps what was already
    /// selected and toggles the clicked component (Shift+click in Maya).
    pub fn select_at(&mut self, ray: &Ray, additive: bool, viewport_height: f32) {
        let changed = match self.mode {
            EditMode::Object => {
                let hit = picking::hits_object(&self.mesh, ray);
                self.selection.apply_object_click(hit, additive)
            }
            EditMode::Vertex => {
                let hit = picking::pick_vertex(
                    &self.mesh,
                    ray,
                    VERTEX_PICK_RADIUS_PIXELS,
                    viewport_height,
                )
                .map(|pick| pick.vertex);
                self.selection
                    .apply_click(hit, additive, Selection::vertices_mut)
            }
            EditMode::Edge => {
                let hit =
                    picking::pick_edge(&self.mesh, ray, EDGE_PICK_RADIUS_PIXELS, viewport_height)
                        .map(|pick| pick.edge);
                self.selection
                    .apply_click(hit, additive, Selection::edges_mut)
            }
            EditMode::Face => {
                let hit = picking::pick_face(&self.mesh, ray).map(|pick| pick.face);
                self.selection
                    .apply_click(hit, additive, Selection::faces_mut)
            }
        };

        if changed {
            self.dirty = true;
        }
    }

    /// Returns true if the document changed since the last call, and clears the flag.
    pub fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }

    // ---- Undo / redo ----------------------------------------------------------------------------

    /// Captures everything undo/redo versions: the geometry and its selection, not the mode or tool.
    pub fn snapshot(&self) -> DocumentSnapshot {
        DocumentSnapshot {
            mesh: self.mesh.clone(),
            selection: self.selection.clone(),
        }
    }

    /// Starts an undoable action named `label`, remembering the document as it is right now. Pair with
    /// [`Scene::end_action`] when the interaction finishes. A gizmo drag calls this on button press and
    /// `end_action` on release, so the whole drag is a single step.
    pub fn begin_action(&mut self, label: &str) {
        self.history.begin(label, &self.snapshot());
    }

    /// Finishes the open action, recording it only if the document actually changed. Returns true when
    /// a step landed in history, which is what the redo branch was forked by.
    pub fn end_action(&mut self) -> bool {
        self.history.commit(&self.snapshot())
    }

    /// Drops the open action without recording it. The caller is responsible for restoring whatever the
    /// aborted interaction had already changed.
    pub fn cancel_action(&mut self) {
        self.history.cancel();
    }

    /// Steps one action back, restoring the geometry and selection and flagging the document dirty.
    /// Returns false when there was nothing to undo. The action reverted is still named by
    /// [`Scene::redo_label`] once this returns, so the UI can read "Redo - Move Vertex" off it.
    ///
    /// An armed extrude is abandoned first, and before the current state is captured: its preview lives
    /// in the mesh, so snapshotting it would file the half-finished lump as a real step, and its frozen
    /// snapshot would then restore geometry that this undo has just rolled back.
    pub fn undo(&mut self) -> bool {
        self.abandon_extrude();
        let current = self.snapshot();
        match self.history.undo(&current) {
            Some(state) => {
                self.restore(state);
                true
            }
            None => false,
        }
    }

    /// Re-applies the last undone action. Returns false when there was nothing to redo. Abandons an armed
    /// extrude for the same reason [`Scene::undo`] does.
    pub fn redo(&mut self) -> bool {
        self.abandon_extrude();
        let current = self.snapshot();
        match self.history.redo(&current) {
            Some(state) => {
                self.restore(state);
                true
            }
            None => false,
        }
    }

    pub fn can_undo(&self) -> bool {
        self.history.can_undo()
    }

    pub fn can_redo(&self) -> bool {
        self.history.can_redo()
    }

    /// Name of the action `undo` would revert, for the UI's "Undo - Move" button.
    pub fn undo_label(&self) -> Option<&str> {
        self.history.undo_label()
    }

    /// Name of the action `redo` would re-apply.
    pub fn redo_label(&self) -> Option<&str> {
        self.history.redo_label()
    }

    /// Installs a snapshot, flags the document dirty, and clips the selection back to what is
    /// selectable. Mode and tool survive untouched: undo edits the document, not the toolbox.
    fn restore(&mut self, snapshot: DocumentSnapshot) {
        self.mesh = snapshot.mesh;
        self.selection = snapshot.selection;
        self.dirty = true;

        // The user may have switched modes after the action being undone, and future topology edits can
        // remove components a stale selection still names. Neither must survive into the document.
        self.selection.retain_mode(self.mode);
        let vertex_count = self.mesh.vertices.len();
        let face_count = self.mesh.faces.len();
        self.selection.vertices_mut().retain(|v| v.0 < vertex_count);
        self.selection.faces_mut().retain(|f| f.0 < face_count);
        if !self.selection.edges().is_empty() {
            let live: HashSet<crate::mesh::EdgeId> = self
                .mesh
                .edges()
                .into_iter()
                .map(|(edge, _)| edge)
                .collect();
            self.selection
                .edges_mut()
                .retain(|edge| live.contains(edge));
        }
    }
}

/// The single direction a group of faces should travel, or `None` when they have no common one.
///
/// The average of the faces' outward normals, which is what makes extruding a corner of the cube push it
/// diagonally out as Maya does rather than along whichever face was clicked last. Returns `None` when the
/// normals cancel out - two opposite faces averaged together point nowhere, and moving them by a zero
/// vector would be refused downstream as a degenerate offset with a message about wall area that says
/// nothing about the selection.
fn extrude_direction(mesh: &HalfEdgeMesh, faces: &[FaceId]) -> Option<Vector3<f32>> {
    let total = faces
        .iter()
        .map(|&face| mesh.face_normal(face))
        .fold(Vector3::new(0.0, 0.0, 0.0), |sum, normal| sum + normal);

    // `face_normal` returns a unit vector for a face with area and the zero vector for a collapsed one,
    // so a non-finite or vanishing sum means the faces give no usable direction rather than that the
    // arithmetic overflows.
    if !total.x.is_finite() || !total.y.is_finite() || !total.z.is_finite() {
        return None;
    }
    let magnitude = total.magnitude();
    if magnitude < f32::EPSILON {
        return None;
    }
    Some(total / magnitude)
}

/// The name the step takes in history, matching Maya's singular and plural command labels.
fn extrude_label(face_count: usize) -> String {
    if face_count == 1 {
        "Extrude Face".to_string()
    } else {
        "Extrude Faces".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{FaceId, VertexId};
    use cgmath::{Point3, Vector3};

    const VIEWPORT_HEIGHT: f32 = 600.0;

    /// A ray aimed at the front of the cube from 3 units out.
    fn front_ray() -> Ray {
        Ray::new(Point3::new(0.0, 0.0, 3.0), Vector3::new(0.0, 0.0, -1.0))
    }

    /// A ray aimed at the front-bottom-left corner of the cube.
    fn corner_ray() -> Ray {
        Ray::new(Point3::new(0.0, 0.0, 3.0), Vector3::new(-0.5, -0.5, -2.5))
    }

    /// A ray pointing away from the cube entirely.
    fn miss_ray() -> Ray {
        Ray::new(Point3::new(0.0, 0.0, 3.0), Vector3::new(3.0, 3.0, -1.0))
    }

    fn face_mode_scene() -> Scene {
        let mut scene = Scene::new();
        scene.set_mode(EditMode::Face);
        scene.take_dirty();
        scene
    }

    #[test]
    fn selecting_a_face_marks_the_document_dirty() {
        let mut scene = face_mode_scene();
        scene.select_at(&front_ray(), false, VIEWPORT_HEIGHT);
        assert_eq!(scene.selected_count(), 1);
        assert!(scene.take_dirty(), "the renderer must be told to redraw");
    }

    #[test]
    fn reselecting_the_same_component_does_not_dirty_the_document() {
        let mut scene = face_mode_scene();
        scene.select_at(&front_ray(), false, VIEWPORT_HEIGHT);
        scene.take_dirty();

        scene.select_at(&front_ray(), false, VIEWPORT_HEIGHT);
        assert!(!scene.take_dirty(), "no change, no re-upload");
    }

    #[test]
    fn clicking_empty_space_clears_the_selection() {
        let mut scene = face_mode_scene();
        scene.select_at(&front_ray(), false, VIEWPORT_HEIGHT);
        scene.take_dirty();

        scene.select_at(&miss_ray(), false, VIEWPORT_HEIGHT);
        assert_eq!(scene.selected_count(), 0);
        assert!(scene.take_dirty());
    }

    #[test]
    fn shift_click_toggles_a_component_off() {
        let mut scene = face_mode_scene();
        scene.select_at(&front_ray(), false, VIEWPORT_HEIGHT);
        assert_eq!(scene.selected_count(), 1);

        scene.select_at(&front_ray(), true, VIEWPORT_HEIGHT);
        assert_eq!(scene.selected_count(), 0, "shift-click toggles it off");
    }

    #[test]
    fn selected_face_is_the_one_the_ray_hit() {
        let mut scene = face_mode_scene();
        scene.select_at(&front_ray(), false, VIEWPORT_HEIGHT);
        assert!(scene.selection().is_face_selected(FaceId(0)));
        assert!(!scene.selection().is_face_selected(FaceId(2)));
    }

    #[test]
    fn vertex_mode_selects_a_vertex_not_a_face() {
        let mut scene = Scene::new();
        scene.set_mode(EditMode::Vertex);
        scene.select_at(&corner_ray(), false, VIEWPORT_HEIGHT);

        assert_eq!(scene.selected_count(), 1);
        assert!(scene.selection().is_vertex_selected(VertexId(0)));
    }

    #[test]
    fn switching_modes_drops_the_previous_kind_of_selection() {
        let mut scene = Scene::new();
        scene.set_mode(EditMode::Vertex);
        scene.select_at(&corner_ray(), false, VIEWPORT_HEIGHT);
        assert_eq!(scene.selected_count(), 1);

        scene.set_mode(EditMode::Face);
        assert_eq!(scene.selected_count(), 0);
        assert!(!scene.selection().is_vertex_selected(VertexId(0)));
    }

    #[test]
    fn object_mode_selects_the_object_only_when_the_ray_hits_it() {
        let mut scene = Scene::new();
        scene.select_at(&front_ray(), false, VIEWPORT_HEIGHT);
        assert!(scene.selection().is_object_selected());
        assert_eq!(scene.selected_count(), 1);

        scene.select_at(&miss_ray(), false, VIEWPORT_HEIGHT);
        assert!(!scene.selection().is_object_selected());
        assert_eq!(scene.selected_count(), 0);
    }

    #[test]
    fn tool_can_be_switched_independently_of_selection_and_mode() {
        let mut scene = Scene::new();
        assert_eq!(scene.tool(), Tool::Translate);

        scene.set_tool(Tool::Select);
        assert_eq!(scene.tool(), Tool::Select);

        // Selection remains intact across tool changes
        scene.select_at(&front_ray(), false, VIEWPORT_HEIGHT);
        assert_eq!(scene.selected_count(), 1);

        scene.set_tool(Tool::Translate);
        assert_eq!(scene.selected_count(), 1);
        assert_eq!(scene.tool(), Tool::Translate);

        // Mode changes preserve active tool
        scene.set_mode(EditMode::Vertex);
        assert_eq!(scene.tool(), Tool::Translate);
    }

    #[test]
    fn translate_vertices_moves_positions_and_dirties_document() {
        let mut scene = Scene::new();
        let v0 = crate::mesh::VertexId(0);
        let initial_pos = scene.mesh().vertex_position(v0);
        assert!(!scene.take_dirty());

        let delta = cgmath::Vector3::new(1.0, 2.0, 3.0);
        scene.translate_vertices(&[(v0, initial_pos)], delta);

        assert!(scene.take_dirty(), "translating vertices must flag dirty");
        let new_pos = scene.mesh().vertex_position(v0);
        assert_eq!(new_pos, initial_pos + delta);
    }

    // ---- Undo / redo -----------------------------------------------------------------------------

    /// Moves vertex 0 by `delta` as one named, undoable action, exactly as a gizmo drag does.
    fn move_vertex(scene: &mut Scene, delta: Vector3<f32>) {
        let v0 = VertexId(0);
        let start = scene.mesh().vertex_position(v0);
        scene.begin_action("Move Vertex");
        scene.translate_vertices(&[(v0, start)], delta);
        scene.end_action();
    }

    #[test]
    fn an_undoable_action_can_be_stepped_back_and_reapplied() {
        let mut scene = Scene::new();
        let start = scene.mesh().vertex_position(VertexId(0));

        move_vertex(&mut scene, Vector3::new(1.0, 0.0, 0.0));
        assert!(scene.can_undo());
        assert_eq!(scene.undo_label(), Some("Move Vertex"));

        scene.take_dirty();
        assert!(scene.undo());
        assert_eq!(scene.mesh().vertex_position(VertexId(0)), start);
        assert!(
            scene.take_dirty(),
            "undo must dirty the document or the GPU buffers keep the undone geometry"
        );

        assert!(scene.redo());
        assert_eq!(
            scene.mesh().vertex_position(VertexId(0)),
            start + Vector3::new(1.0, 0.0, 0.0)
        );
    }

    #[test]
    fn a_whole_drag_is_one_undo_step() {
        let mut scene = Scene::new();
        let start = scene.mesh().vertex_position(VertexId(0));
        let v0 = VertexId(0);

        scene.begin_action("Move Vertex");
        for step in 1..=10 {
            let delta = Vector3::new(step as f32, 0.0, 0.0);
            scene.translate_vertices(&[(v0, start)], delta);
        }
        scene.end_action();

        assert!(scene.undo());
        assert_eq!(
            scene.mesh().vertex_position(VertexId(0)),
            start,
            "ten drag frames must roll back together, not one frame at a time"
        );
        assert!(
            !scene.can_undo(),
            "the drag left exactly one step in history"
        );
    }

    #[test]
    fn a_drag_that_never_moved_records_nothing() {
        let mut scene = Scene::new();
        scene.begin_action("Move Vertex");

        assert!(
            !scene.end_action(),
            "pressing a gizmo axis and releasing without dragging must not create a step"
        );
        assert!(!scene.can_undo());
    }

    #[test]
    fn a_cancelled_drag_leaves_no_trace() {
        let mut scene = Scene::new();
        let start = scene.mesh().vertex_position(VertexId(0));

        scene.begin_action("Move Vertex");
        scene.translate_vertices(&[(VertexId(0), start)], Vector3::new(5.0, 0.0, 0.0));
        scene.cancel_action();

        assert!(!scene.can_undo());
    }

    #[test]
    fn stepping_back_twice_walks_the_stack_in_order() {
        let mut scene = Scene::new();
        let start = scene.mesh().vertex_position(VertexId(0));

        move_vertex(&mut scene, Vector3::new(1.0, 0.0, 0.0));
        move_vertex(&mut scene, Vector3::new(2.0, 0.0, 0.0));

        assert!(scene.undo());
        assert_eq!(
            scene.mesh().vertex_position(VertexId(0)),
            start + Vector3::new(1.0, 0.0, 0.0),
            "the first undo returns to the state after the first action"
        );
        assert!(scene.undo());
        assert_eq!(scene.mesh().vertex_position(VertexId(0)), start);
        assert!(
            !scene.undo(),
            "history is exhausted, and undo must not wipe the document"
        );
        assert_eq!(scene.mesh().vertex_position(VertexId(0)), start);
    }

    #[test]
    fn a_new_action_after_an_undo_discards_the_redo_branch() {
        let mut scene = Scene::new();
        move_vertex(&mut scene, Vector3::new(1.0, 0.0, 0.0));
        scene.undo();
        assert!(scene.can_redo());

        move_vertex(&mut scene, Vector3::new(4.0, 0.0, 0.0));
        assert!(
            !scene.can_redo(),
            "editing after an undo forks the timeline, as in Maya"
        );
    }

    #[test]
    fn undo_does_not_undo_the_current_mode_or_tool() {
        let mut scene = Scene::new();
        move_vertex(&mut scene, Vector3::new(1.0, 0.0, 0.0));

        scene.set_mode(EditMode::Vertex);
        scene.set_tool(Tool::Rotate);
        scene.undo();

        assert_eq!(
            scene.mode(),
            EditMode::Vertex,
            "undo edits the document, not the mode"
        );
        assert_eq!(
            scene.tool(),
            Tool::Rotate,
            "undo must not touch the toolbox"
        );
    }

    #[test]
    fn undo_clips_a_selection_the_current_mode_cannot_hold() {
        let mut scene = face_mode_scene();
        scene.select_at(&front_ray(), false, VIEWPORT_HEIGHT);
        assert_eq!(scene.selected_count(), 1, "a face is selected");

        // The user then switches to vertex mode and selects there. Undoing that action restores the
        // face selection, but the mode stays Vertex, so the stale face must be clipped away.
        scene.set_mode(EditMode::Vertex);
        scene.begin_action("Select Vertex");
        scene.select_at(&corner_ray(), false, VIEWPORT_HEIGHT);
        scene.end_action();

        assert!(scene.undo());
        assert_eq!(scene.mode(), EditMode::Vertex);
        assert_eq!(
            scene.selected_count(),
            0,
            "a restored face selection must not survive in vertex mode"
        );
    }

    #[test]
    fn undo_restores_a_selection_that_is_still_valid() {
        let mut scene = face_mode_scene();
        scene.select_at(&front_ray(), false, VIEWPORT_HEIGHT);
        scene.begin_action("Nudge");
        scene.translate_vertices(
            &[(VertexId(0), Point3::new(0.0, 0.0, 0.5))],
            Vector3::unit_x(),
        );
        scene.end_action();

        assert_eq!(
            scene.selected_count(),
            1,
            "selection survives an unrelated edit"
        );
        assert!(scene.undo());
        assert_eq!(
            scene.selected_count(),
            1,
            "undo keeps the selection alongside the geometry it was made with"
        );
    }

    /// Asserts a refusal named `needle` and never came back empty-handed.
    fn expect_reason(errors: &[String], needle: &str) {
        assert!(!errors.is_empty(), "a refusal must always give a reason");
        assert!(
            errors.iter().any(|error| error.contains(needle)),
            "expected a reason containing {needle:?}, got {errors:?}"
        );
    }

    /// The depth the extrude tests push a face by, in model units - what a short drag reaches on the
    /// unit cube the document opens on.
    const DEPTH: f32 = 0.25;

    /// Drives a whole extrude gesture - arm, preview, settle - standing in for the mouse.
    ///
    /// Tests that only care where the faces end up ask for the gesture here rather than repeating three
    /// calls at every site. Tests about the gesture itself drive the three steps by hand.
    fn extrude(scene: &mut Scene, depth: f32) -> Result<usize, TopologyErrors> {
        scene.begin_extrude_drag()?;
        scene.preview_extrude(depth)?;
        Ok(scene
            .commit_extrude()
            .expect("an armed extrude always settles")
            .walls)
    }

    /// Selects the front face by ray, the way a click in Face mode would.
    fn select_front_face(scene: &mut Scene) {
        scene.select_at(&front_ray(), false, VIEWPORT_HEIGHT);
        assert_eq!(
            scene.selected_count(),
            1,
            "the test depends on hitting one face"
        );
    }

    #[test]
    fn extrude_is_off_until_a_face_is_selected_in_face_mode() {
        let mut scene = Scene::new();
        assert!(
            !scene.can_extrude(),
            "Object mode has no faces to move, so the button must not look live"
        );

        scene.set_mode(EditMode::Face);
        assert!(
            !scene.can_extrude(),
            "Face mode with nothing selected still has nothing to move"
        );

        select_front_face(&mut scene);
        assert!(scene.can_extrude(), "one face selected is enough");
    }

    #[test]
    fn extrude_is_refused_in_vertex_mode_without_touching_the_mesh() {
        let mut scene = Scene::new();
        scene.set_mode(EditMode::Vertex);
        scene.select_at(&corner_ray(), false, VIEWPORT_HEIGHT);
        scene.take_dirty();
        let before = scene.mesh().clone();

        let errors = extrude(&mut scene, DEPTH).expect_err("vertices do not extrude");
        expect_reason(&errors, "needs at least one face selected in Face mode");
        assert_eq!(scene.mesh(), &before);
        assert!(
            !scene.can_undo(),
            "a refusal records no step, so Undo still belongs to whatever came before"
        );
    }

    #[test]
    fn extruding_a_selected_face_lands_one_undoable_step_named_for_the_command() {
        let mut scene = face_mode_scene();
        select_front_face(&mut scene);
        let before = scene.mesh().clone();

        let walls = extrude(&mut scene, DEPTH).expect("a selected face of the cube should extrude");

        assert_eq!(walls, 4);
        assert_eq!(scene.mesh().faces.len(), 10);
        assert!(scene.can_undo());
        assert_eq!(
            scene.undo_label(),
            Some("Extrude Face"),
            "the step is named after the command, as Maya's history panel shows it"
        );

        assert!(scene.undo());
        assert_eq!(
            scene.mesh(),
            &before,
            "one undo returns the whole cube, not half a prism"
        );
        assert!(scene.can_redo());
        assert!(scene.redo());
        assert_eq!(
            scene.mesh().faces.len(),
            10,
            "redo re-grows the same extrusion"
        );
    }

    #[test]
    fn an_extrude_leaves_the_face_that_was_pointed_at_selected() {
        // The highlight stays on the geometry the user clicked. That is also what keeps the direction
        // honest: a face carries its own normal, so the thing under the cursor is what decides where it
        // goes next - no remembered direction is involved anywhere.
        let mut scene = face_mode_scene();
        select_front_face(&mut scene);

        extrude(&mut scene, DEPTH).expect("a selected face of the cube should extrude");

        assert_eq!(
            scene.selection.faces().len(),
            1,
            "the selection is exactly the face that was clicked"
        );
        assert!(
            scene.selection.faces().contains(&FaceId(0)),
            "the front face keeps its id across an extrude, because the primitive appends and re-points"
        );
        assert_near(
            scene.mesh().face_center(FaceId(0)),
            Point3::new(0.0, 0.0, 0.5 + DEPTH),
            "and it is the cap that travelled, not the hole it left behind",
        );
    }

    #[test]
    fn pressing_extrude_twice_pushes_straight_out_along_the_face_normal() {
        let mut scene = face_mode_scene();
        select_front_face(&mut scene);

        extrude(&mut scene, DEPTH).expect("the first press should extrude");
        extrude(&mut scene, DEPTH).expect("the second press should extrude the cap again");

        assert_near(
            scene.mesh().face_center(FaceId(0)),
            Point3::new(0.0, 0.0, 0.5 + 2.0 * DEPTH),
            "two presses travel along +z with no sideways drift, since the cap still faces +z",
        );
        assert!(
            scene.mesh().boundary_edges().is_empty(),
            "the solid stays closed however many times it is extruded"
        );
    }

    #[test]
    fn a_wall_picked_after_an_extrude_travels_along_its_own_normal() {
        // The regression this replaces: the direction of the first extrude used to be remembered and
        // reused whenever the selection was a subset of it, so picking one wall and pressing the key slid
        // that wall along the *original* face's normal - through the prism - instead of out of the wall.
        let mut scene = face_mode_scene();
        select_front_face(&mut scene);
        extrude(&mut scene, DEPTH).expect("a selected face of the cube should extrude");

        let walls: Vec<FaceId> = (6..scene.mesh().faces.len()).map(FaceId).collect();
        let right_wall = walls
            .into_iter()
            .find(|&wall| scene.mesh().face_normal(wall).x > 0.9)
            .expect("one of the four walls must face +x");
        let before = scene.mesh().face_center(right_wall);

        scene.selection.faces_mut().clear();
        scene.selection.faces_mut().insert(right_wall);
        extrude(&mut scene, DEPTH).expect("a single wall should extrude on its own");

        assert_near(
            scene.mesh().face_center(right_wall),
            Point3::new(before.x + DEPTH, before.y, before.z),
            "the wall moves out along +x, its own normal, and not along the front face's +z",
        );
    }

    /// Position equality with the tolerance floating-point accumulation deserves, so a test failure means
    /// the geometry moved wrong rather than that an addition rounded.
    fn assert_near(actual: Point3<f32>, expected: Point3<f32>, note: &str) {
        let drift = (actual - expected).magnitude();
        assert!(
            drift < 1e-5,
            "{note}: expected {expected:?}, got {actual:?}"
        );
    }

    /// The same for a direction. Kept separate rather than generic because `Point3` and `Vector3` share
    /// no subtraction trait in cgmath that a single bound could name.
    fn assert_dir_near(actual: Vector3<f32>, expected: Vector3<f32>, note: &str) {
        let drift = (actual - expected).magnitude();
        assert!(
            drift < 1e-5,
            "{note}: expected {expected:?}, got {actual:?}"
        );
    }

    #[test]
    fn an_extrude_refusal_leaves_history_and_selection_exactly_as_they_were() {
        // Selecting every face is reachable through Ctrl+A. Two opposite faces averaged together point
        // nowhere, so this is refused before the mesh is ever asked for walls. What matters is that the
        // failed command is invisible to history: an empty step filed under "Extrude Faces" would make
        // the next Undo look like it had done nothing.
        let mut scene = face_mode_scene();
        for face in 0..6 {
            scene.selection.faces_mut().insert(FaceId(face));
        }
        scene.take_dirty();
        let before = scene.mesh().clone();

        let errors = extrude(&mut scene, DEPTH)
            .expect_err("opposite faces have no common direction to travel");
        expect_reason(&errors, "no single way to push them");
        assert_eq!(scene.mesh(), &before);
        assert!(!scene.can_undo(), "a refused command never reaches history");
        assert_eq!(
            scene.selected_count(),
            6,
            "the selection the user made is not this command's to change"
        );
    }

    #[test]
    fn a_closed_selection_is_refused_before_the_mesh_is_touched() {
        // Any region forming a closed surface has its outward normals sum to nothing, so selecting a whole
        // solid gives the command nowhere to send it and it is turned away here rather than at the mesh.
        // Reached the way a user reaches it: grow a lump, then Ctrl+A the result.
        let mut scene = face_mode_scene();
        select_front_face(&mut scene);
        extrude(&mut scene, DEPTH).expect("a selected face of the cube should extrude");

        let faces = scene.mesh().faces.len();
        scene.selection.faces_mut().clear();
        for face in 0..faces {
            scene.selection.faces_mut().insert(FaceId(face));
        }
        let before = scene.mesh().clone();

        let errors =
            extrude(&mut scene, DEPTH).expect_err("a closed solid has nothing to grow from");
        expect_reason(&errors, "no single way to push them");
        assert_eq!(
            scene.mesh(),
            &before,
            "the refusal must not have cut into the solid"
        );
    }

    /// The z of every corner of the front face, which is the one the extrude pushes. A drag that drifted
    /// sideways or grew unevenly would not leave all four equal.
    fn front_depth(scene: &Scene) -> f32 {
        let mesh = scene.mesh();
        let depths: Vec<f32> = mesh
            .face_vertices(FaceId(0))
            .iter()
            .map(|&corner| mesh.vertices[corner.0].position.z)
            .collect();
        assert_eq!(
            depths.len(),
            4,
            "the front face of the cube has four corners"
        );
        let first = depths[0];
        assert!(
            depths.iter().all(|&z| (z - first).abs() < 1e-5),
            "the face must move rigidly, got {depths:?}"
        );
        first
    }

    #[test]
    fn arming_reports_the_travel_axis_and_moves_nothing() {
        // The guide is what the handle is drawn from and what the cursor is projected onto, so its
        // direction has to be the selected face's own normal - and arming must stay side-effect free, or
        // pressing the key would already have grown geometry the user never dragged for.
        let mut scene = face_mode_scene();
        select_front_face(&mut scene);
        let before = scene.mesh().clone();

        let guide = scene
            .begin_extrude_drag()
            .expect("a selected face of the cube should arm");

        assert_dir_near(
            guide.direction,
            Vector3::new(0.0, 0.0, 1.0),
            "the front face points along +z, so that is the axis the depth is measured along",
        );
        assert_near(
            guide.pivot,
            Point3::new(0.0, 0.0, 0.5),
            "the handle sits on the face the user is looking at",
        );
        assert_eq!(scene.mesh(), &before, "arming alone must not move a corner");
        assert!(scene.is_extruding());
    }

    #[test]
    fn the_depth_is_a_position_so_courting_the_cursor_back_moves_the_faces_back() {
        // Depth measured from the armed snapshot rather than summed from mouse deltas. Were it a running
        // total, pulling the cursor back over the handle would keep growing the lump instead of shrinking
        // it, and a drag that crossed zero would extrude the wrong way with no way to see it happening.
        let mut wandered = face_mode_scene();
        select_front_face(&mut wandered);
        wandered
            .begin_extrude_drag()
            .expect("arming should succeed");
        wandered
            .preview_extrude(0.9)
            .expect("a long preview should apply");
        wandered
            .preview_extrude(0.25)
            .expect("a shorter preview should apply");

        let mut direct = face_mode_scene();
        select_front_face(&mut direct);
        direct.begin_extrude_drag().expect("arming should succeed");
        direct
            .preview_extrude(0.25)
            .expect("the same depth applied once");

        assert!(
            (front_depth(&wandered) - front_depth(&direct)).abs() < 1e-5,
            "0.9 then 0.25 must land where 0.25 alone lands"
        );
        assert_eq!(
            wandered.mesh(),
            direct.mesh(),
            "and the whole document must match, not just the face that moved"
        );
    }

    #[test]
    fn bringing_the_drag_back_onto_the_surface_restores_the_cube_exactly() {
        let mut scene = face_mode_scene();
        select_front_face(&mut scene);
        let before = scene.mesh().clone();
        scene.begin_extrude_drag().expect("arming should succeed");
        scene.preview_extrude(0.6).expect("a preview should apply");
        assert!(front_depth(&scene) > 0.5, "and it should have moved");

        scene
            .preview_extrude(0.0)
            .expect("zero depth is the surface");

        assert_eq!(
            scene.mesh(),
            &before,
            "a drag returned to where it started must leave no trace"
        );
    }

    #[test]
    fn a_negative_depth_pushes_the_face_inward_through_the_solid() {
        let mut scene = face_mode_scene();
        select_front_face(&mut scene);
        scene.begin_extrude_drag().expect("arming should succeed");
        scene
            .preview_extrude(-0.25)
            .expect("dragging the other way is a legitimate extrude");

        assert!(
            (front_depth(&scene) - 0.25).abs() < 1e-5,
            "the front face travels inward, into the cube"
        );
    }

    #[test]
    fn abandoning_puts_the_document_back_and_lands_no_history_step() {
        let mut scene = face_mode_scene();
        select_front_face(&mut scene);
        let before = scene.mesh().clone();
        scene.take_dirty();
        scene.begin_extrude_drag().expect("arming should succeed");
        scene.preview_extrude(0.4).expect("a preview should apply");

        assert!(scene.abandon_extrude(), "an open drag says it closed");

        assert_eq!(scene.mesh(), &before, "Escape removes the preview lump");
        assert!(
            !scene.can_undo(),
            "an abandoned drag is not an edit, so Undo still belongs to what came before"
        );
        assert!(!scene.is_extruding());
        assert!(
            !scene.abandon_extrude(),
            "and with nothing open there is nothing to abandon"
        );
    }

    #[test]
    fn settling_a_drag_that_never_left_the_surface_records_nothing() {
        let mut scene = face_mode_scene();
        select_front_face(&mut scene);
        let before = scene.mesh().clone();
        scene.begin_extrude_drag().expect("arming should succeed");

        let commit = scene
            .commit_extrude()
            .expect("an armed drag always settles");

        assert_eq!(commit.walls, 0, "no depth was ever previewed");
        assert!(
            !commit.recorded,
            "so there is no step, and the undo chain is undisturbed"
        );
        assert_eq!(scene.mesh(), &before);
        assert!(!scene.can_undo());
    }

    #[test]
    fn switching_modes_abandons_an_extrude_awaiting_its_depth() {
        // The drag moves the faces Face mode had selected, and `set_mode` drops that selection, so
        // carrying the drag across the switch would leave a previewed lump in the mesh that nothing can
        // finish or point at any more.
        let mut scene = face_mode_scene();
        select_front_face(&mut scene);
        let before = scene.mesh().clone();
        scene.begin_extrude_drag().expect("arming should succeed");
        scene.preview_extrude(0.4).expect("a preview should apply");
        assert_ne!(
            scene.mesh(),
            &before,
            "the preview is live before the switch"
        );

        scene.set_mode(EditMode::Vertex);

        assert_eq!(
            scene.mesh(),
            &before,
            "the switch restores what it interrupted"
        );
        assert!(!scene.is_extruding());
        assert!(
            scene.commit_extrude().is_none(),
            "and the click that would have settled it now selects instead"
        );
        assert!(
            !scene.can_undo(),
            "an abandoned drag is no step, so Undo still belongs to whatever came before"
        );
    }

    #[test]
    fn switching_tools_releases_the_handle_an_extrude_was_holding() {
        // While the drag is open its travel arm replaces the translate gizmo, so accepting a tool change
        // without closing the drag would make the panel look dead. Restoring the mesh is what lets the
        // freshly selected tool act on the cube the user can see, not on a lump that is no longer there.
        let mut scene = face_mode_scene();
        select_front_face(&mut scene);
        let before = scene.mesh().clone();
        scene.begin_extrude_drag().expect("arming should succeed");
        scene.preview_extrude(0.4).expect("a preview should apply");

        scene.set_tool(Tool::Rotate);

        assert_eq!(scene.tool(), Tool::Rotate);
        assert_eq!(scene.mesh(), &before);
        assert!(!scene.is_extruding());
    }

    #[test]
    fn a_second_extrude_is_refused_while_one_is_awaiting_its_depth() {
        // Nesting would take its snapshot of a mesh that is already a preview, and abandoning the inner
        // drag would then restore the outer drag's lump as though it were original geometry.
        let mut scene = face_mode_scene();
        select_front_face(&mut scene);
        scene.begin_extrude_drag().expect("arming should succeed");
        scene.preview_extrude(0.3).expect("a preview should apply");
        let previewed = scene.mesh().clone();

        let errors = scene
            .begin_extrude_drag()
            .expect_err("a second extrude cannot open on top of the first");
        expect_reason(&errors, "already in progress");
        assert_eq!(
            scene.mesh(),
            &previewed,
            "and the refusal leaves the drag mid-flight, still draggable"
        );

        scene
            .preview_extrude(0.5)
            .expect("the first drag still accepts depth after the refusal");
        assert!(
            (front_depth(&scene) - 1.0).abs() < 1e-5,
            "0.5 out from the original surface"
        );
    }

    #[test]
    fn one_whole_drag_is_one_undo_step_however_many_previews_it_had() {
        let mut scene = face_mode_scene();
        select_front_face(&mut scene);
        let before = scene.mesh().clone();
        scene.begin_extrude_drag().expect("arming should succeed");
        for depth in [0.1, 0.2, 0.35, 0.3, 0.45] {
            scene
                .preview_extrude(depth)
                .expect("each preview should apply");
        }

        let commit = scene
            .commit_extrude()
            .expect("an armed drag always settles");
        assert!(commit.recorded, "the faces did move, so a step landed");
        assert_eq!(commit.faces, 1);
        assert_eq!(commit.walls, 4);

        assert!(scene.undo(), "one step back");
        assert_eq!(
            scene.mesh(),
            &before,
            "returns the untouched cube, not the shape from some earlier preview"
        );
        assert!(!scene.can_undo(), "and no half-step is left to unwind");
    }

    // ---- Edge loop ---------------------------------------------------------------------------

    fn edge_mode_scene() -> Scene {
        let mut scene = Scene::new();
        scene.set_mode(EditMode::Edge);
        scene.take_dirty();
        scene
    }

    /// A scene in Edge mode with `edges` selected directly, which is how a selection looks that arrived
    /// from clicks and then survived an edit - building it through `select_at` would tie every test here to
    /// where the viewport happens to put an edge on screen.
    fn scene_with_edges(edges: &[EdgeId]) -> Scene {
        let mut scene = edge_mode_scene();
        scene.selection.edges_mut().extend(edges.iter().copied());
        scene.take_dirty();
        scene
    }

    #[test]
    fn an_edge_loop_needs_exactly_one_edge_selected_in_edge_mode() {
        // The button and the hotkey both read this, so the four states here are the four the user can tell
        // apart on screen: wrong mode, right mode nothing picked, one picked, and the ambiguous case.
        let mut object_mode = Scene::new();
        object_mode
            .selection
            .edges_mut()
            .insert(EdgeId::between(VertexId(0), VertexId(1)));
        assert!(
            !object_mode.can_insert_edge_loop(),
            "Object mode has no edge to seed a ring from"
        );

        assert!(
            !edge_mode_scene().can_insert_edge_loop(),
            "Edge mode with nothing selected"
        );

        let mut one = edge_mode_scene();
        one.selection
            .edges_mut()
            .insert(EdgeId::between(VertexId(0), VertexId(1)));
        assert!(
            one.can_insert_edge_loop(),
            "one edge is exactly what is wanted"
        );

        let two = scene_with_edges(&[
            EdgeId::between(VertexId(0), VertexId(1)),
            EdgeId::between(VertexId(2), VertexId(3)),
        ]);
        assert!(
            !two.can_insert_edge_loop(),
            "two edges can name the same ring, so neither is offered"
        );
    }

    #[test]
    fn an_edge_loop_splits_the_cube_and_hands_the_selection_to_the_ring() {
        let mut scene = scene_with_edges(&[EdgeId::between(VertexId(0), VertexId(1))]);

        let commit = scene
            .insert_edge_loop()
            .unwrap_or_else(|errors| panic!("one cube edge should seed a loop: {errors:?}"));

        assert_eq!(commit.faces_cut, 4);
        assert_eq!(commit.new_vertices, 4);
        assert_eq!(scene.mesh().faces.len(), 10);
        assert!(scene.take_dirty(), "the renderer must be told to redraw");

        let selected: Vec<EdgeId> = scene.selection.edges().iter().copied().collect();
        assert_eq!(selected.len(), 4, "the new ring is what stays highlighted");
        assert!(
            selected
                .iter()
                .all(|&edge| scene.mesh().edges().iter().any(|&(live, _)| live == edge)),
            "every edge left selected must exist on the mesh"
        );
    }

    #[test]
    fn an_edge_loop_is_one_undo_step() {
        let before = HalfEdgeMesh::create_cube();
        let mut scene = scene_with_edges(&[EdgeId::between(VertexId(0), VertexId(1))]);

        scene
            .insert_edge_loop()
            .expect("the loop should cut cleanly");
        assert_ne!(scene.mesh(), &before, "the cube changed");

        assert!(scene.undo(), "one step back");
        assert_eq!(
            scene.mesh(),
            &before,
            "undo returns the whole cube, not the mesh mid-cut"
        );
        assert!(!scene.can_undo(), "and no half-step is left to unwind");
    }

    #[test]
    fn an_edge_loop_refusal_changes_nothing_and_lands_no_history_step() {
        let mut scene = scene_with_edges(&[
            EdgeId::between(VertexId(0), VertexId(1)),
            EdgeId::between(VertexId(2), VertexId(3)),
        ]);

        let errors = scene
            .insert_edge_loop()
            .expect_err("two edges cannot seed one ring");
        expect_reason(&errors, "one edge at a time");

        assert_eq!(scene.mesh(), &HalfEdgeMesh::create_cube());
        assert!(
            !scene.can_undo(),
            "a refused command must not appear in the undo chain"
        );
        assert_eq!(
            scene.selection().edges().len(),
            2,
            "and must not move the selection"
        );
    }

    #[test]
    fn an_edge_loop_in_another_mode_says_which_mode_it_wants() {
        let mut scene = Scene::new();
        scene.set_mode(EditMode::Face);
        scene.selection.faces_mut().insert(FaceId(0));

        let errors = scene
            .insert_edge_loop()
            .expect_err("a face cannot seed an edge ring");
        expect_reason(&errors, "switch to Edge mode");
    }

    #[test]
    fn an_edge_loop_ignores_a_selection_left_stale_by_another_edit() {
        // An id from a mesh that no longer exists is what a selection looks like after an undo or a load.
        // Seeding a ring off it must be refused as "nothing selected", not answered with a cut the user
        // cannot see or a panic on a vertex that is not there.
        let mut scene = edge_mode_scene();
        scene
            .selection
            .edges_mut()
            .insert(EdgeId::between(VertexId(200), VertexId(201)));

        assert!(!scene.can_insert_edge_loop());
        let errors = scene
            .insert_edge_loop()
            .expect_err("a dead edge names nothing to cut");
        expect_reason(&errors, "needs one edge selected");
    }
}
