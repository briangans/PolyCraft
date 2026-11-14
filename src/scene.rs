//! The editable document: the mesh being modeled, the current edit mode, and what is selected.
//!
//! Anything that changes the mesh or the selection goes through here, which flags the document as
//! dirty. `State::render` checks that flag once per frame and re-uploads the GPU buffers.

use crate::mesh::HalfEdgeMesh;
use crate::picking::{self, Ray};
use crate::selection::Selection;

pub use crate::edit_mode::EditMode;
pub use crate::tool::Tool;

/// Click tolerance for components, in pixels. Measured on screen, so it feels the same at any zoom.
pub const VERTEX_PICK_RADIUS_PIXELS: f32 = 8.0;
pub const EDGE_PICK_RADIUS_PIXELS: f32 = 6.0;

pub struct Scene {
    mesh: HalfEdgeMesh,
    dirty: bool,
    mode: EditMode,
    tool: Tool,
    selection: Selection,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            mesh: HalfEdgeMesh::create_cube(),
            dirty: false,
            mode: EditMode::Object,
            tool: Tool::default(),
            selection: Selection::new(),
        }
    }

    pub fn mesh(&self) -> &HalfEdgeMesh {
        &self.mesh
    }

    /// Mutable access to the mesh. Flags the document dirty so the renderer re-uploads it next
    /// frame.
    #[allow(dead_code)] // First caller will be the upcoming edit operations
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
}
