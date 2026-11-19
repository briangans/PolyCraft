//! The editable document: the mesh being modeled, the current edit mode, and what is selected.
//!
//! Anything that changes the mesh or the selection goes through here, which flags the document as
//! dirty. `State::render` checks that flag once per frame and re-uploads the GPU buffers.

use crate::history::HistoryStack;
use crate::mesh::HalfEdgeMesh;
use crate::picking::{self, Ray};
use crate::selection::Selection;

pub use crate::edit_mode::EditMode;
pub use crate::tool::Tool;

/// Click tolerance for components, in pixels. Measured on screen, so it feels the same at any zoom.
pub const VERTEX_PICK_RADIUS_PIXELS: f32 = 8.0;
pub const EDGE_PICK_RADIUS_PIXELS: f32 = 6.0;

/// How many actions `undo` can walk back before the oldest ones are dropped. Meshes here are small
/// (the index buffer caps out around 16k quads), so snapshots stay cheap enough to keep a deep stack.
pub const HISTORY_DEPTH: usize = 64;

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
    selection: Selection,
    history: HistoryStack<DocumentSnapshot>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            mesh: HalfEdgeMesh::create_cube(),
            dirty: false,
            mode: EditMode::Object,
            tool: Tool::default(),
            selection: Selection::new(),
            history: HistoryStack::new(HISTORY_DEPTH),
        }
    }

    pub fn mesh(&self) -> &HalfEdgeMesh {
        &self.mesh
    }

    /// Mutable access to the mesh. Flags the document dirty so the renderer re-uploads it next
    /// frame. Wrap calls in `begin_action` / `end_action` if the change should be undoable.
    #[allow(dead_code)] // First caller will be the upcoming edit operations (extrude, delete)
    pub fn mesh_mut(&mut self) -> &mut HalfEdgeMesh {
        self.dirty = true;
        &mut self.mesh
    }

    pub fn mode(&self) -> EditMode {
        self.mode
    }

    /// Switches modes, dropping the selections the new mode cannot hold (as Maya does).
    pub fn set_mode(&mut self, mode: EditMode) {
        if mode == self.mode {
            return;
        }

        self.mode = mode;
        self.selection.retain_mode(mode);
        self.dirty = true;
    }

    pub fn tool(&self) -> Tool {
        self.tool
    }

    /// Sets the active manipulation or selection tool. Does not alter the current selection.
    pub fn set_tool(&mut self, tool: Tool) {
        self.tool = tool;
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
    pub fn undo(&mut self) -> bool {
        let current = self.snapshot();
        match self.history.undo(&current) {
            Some(state) => {
                self.restore(state);
                true
            }
            None => false,
        }
    }

    /// Re-applies the last undone action. Returns false when there was nothing to redo.
    pub fn redo(&mut self) -> bool {
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
            let live: std::collections::HashSet<crate::mesh::EdgeId> = self
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
}
