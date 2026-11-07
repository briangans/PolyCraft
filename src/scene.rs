//! The editable document: the mesh being modeled plus the current edit mode.
//!
//! Anything that changes the mesh must go through `Scene::mesh_mut`, which flags the mesh as
//! dirty. `State::render` checks that flag once per frame and re-uploads the GPU buffers.

use crate::mesh::HalfEdgeMesh;

/// Maya-style component mode, switched from the pie menu.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditMode {
    Object,
    Vertex,
    Edge,
    Face,
}

impl EditMode {
    pub fn label(self) -> &'static str {
        match self {
            EditMode::Object => "Object",
            EditMode::Vertex => "Vertex",
            EditMode::Edge => "Edge",
            EditMode::Face => "Face",
        }
    }
}

pub struct Scene {
    mesh: HalfEdgeMesh,
    mesh_dirty: bool,
    pub mode: EditMode,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            mesh: HalfEdgeMesh::create_cube(),
            mesh_dirty: false,
            mode: EditMode::Object,
        }
    }

    pub fn mesh(&self) -> &HalfEdgeMesh {
        &self.mesh
    }

    /// Mutable access to the mesh. Flags it dirty so the renderer re-uploads it next frame.
    #[allow(dead_code)] // First caller will be the upcoming edit operations
    pub fn mesh_mut(&mut self) -> &mut HalfEdgeMesh {
        self.mesh_dirty = true;
        &mut self.mesh
    }

    /// Returns true if the mesh changed since the last call, and clears the flag.
    pub fn take_mesh_dirty(&mut self) -> bool {
        std::mem::take(&mut self.mesh_dirty)
    }
}
