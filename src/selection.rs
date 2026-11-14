//! Which components are selected. One set per component kind, because switching modes in Maya
//! starts a fresh selection of that kind.

use crate::edit_mode::EditMode;
use crate::mesh::{EdgeId, FaceId, VertexId};
use std::collections::HashSet;

#[derive(Default)]
pub struct Selection {
    vertices: HashSet<VertexId>,
    edges: HashSet<EdgeId>,
    faces: HashSet<FaceId>,
    object: bool,
}

impl Selection {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_vertex_selected(&self, vertex: VertexId) -> bool {
        self.vertices.contains(&vertex)
    }

    pub fn is_edge_selected(&self, edge: EdgeId) -> bool {
        self.edges.contains(&edge)
    }

    pub fn is_face_selected(&self, face: FaceId) -> bool {
        self.faces.contains(&face)
    }

    pub fn is_object_selected(&self) -> bool {
        self.object
    }

    /// Number of selected components of the kind `mode` works on.
    pub fn count(&self, mode: EditMode) -> usize {
        match mode {
            EditMode::Object => usize::from(self.object),
            EditMode::Vertex => self.vertices.len(),
            EditMode::Edge => self.edges.len(),
            EditMode::Face => self.faces.len(),
        }
    }

    /// Drops every selection that `mode` cannot have selected. Switching modes in Maya works the
    /// same way: the components of the mode you left are deselected.
    pub fn retain_mode(&mut self, mode: EditMode) {
        if mode != EditMode::Vertex {
            self.vertices.clear();
        }
        if mode != EditMode::Edge {
            self.edges.clear();
        }
        if mode != EditMode::Face {
            self.faces.clear();
        }
        if mode != EditMode::Object {
            self.object = false;
        }
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.edges.clear();
        self.faces.clear();
        self.object = false;
    }

    /// Selects the object, or deselects it when `additive` is false and it was already selected.
    /// Returns true when the selection changed.
    pub fn apply_object_click(&mut self, hit: bool, additive: bool) -> bool {
        if !hit {
            if additive || !self.object {
                return false;
            }
            self.object = false;
            return true;
        }

        // A plain click on the object selects it; Shift+click toggles it back off.
        let next = if additive { !self.object } else { true };
        if next == self.object {
            return false;
        }
        self.object = next;
        true
    }

    /// Applies a click on a component: replaces the selection, or adds/removes it when `additive`.
    /// Returns true when the selection changed.
    pub fn apply_click<C>(
        &mut self,
        hit: Option<C>,
        additive: bool,
        set: fn(&mut Self) -> &mut HashSet<C>,
    ) -> bool
    where
        C: Eq + std::hash::Hash + Copy,
    {
        let target = set(self);

        match hit {
            None if additive => false,
            None => {
                if target.is_empty() {
                    false
                } else {
                    target.clear();
                    true
                }
            }
            Some(component) if additive => {
                if target.contains(&component) {
                    target.remove(&component);
                    true
                } else {
                    target.insert(component);
                    true
                }
            }
            Some(component) => {
                let already_only = target.len() == 1 && target.contains(&component);
                target.clear();
                target.insert(component);
                !already_only
            }
        }
    }

    pub fn vertices(&self) -> &HashSet<VertexId> {
        &self.vertices
    }

    pub fn vertices_mut(&mut self) -> &mut HashSet<VertexId> {
        &mut self.vertices
    }

    pub fn edges(&self) -> &HashSet<EdgeId> {
        &self.edges
    }

    pub fn edges_mut(&mut self) -> &mut HashSet<EdgeId> {
        &mut self.edges
    }

    pub fn faces(&self) -> &HashSet<FaceId> {
        &self.faces
    }

    pub fn faces_mut(&mut self) -> &mut HashSet<FaceId> {
        &mut self.faces
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_click_replaces_the_selection() {
        let mut selection = Selection::new();

        assert!(selection.apply_click(Some(VertexId(1)), false, Selection::vertices_mut));
        assert!(selection.is_vertex_selected(VertexId(1)));

        assert!(selection.apply_click(Some(VertexId(4)), false, Selection::vertices_mut));
        assert!(!selection.is_vertex_selected(VertexId(1)));
        assert_eq!(selection.count(EditMode::Vertex), 1);
    }

    #[test]
    fn clicking_the_same_component_again_changes_nothing() {
        let mut selection = Selection::new();
        selection.apply_click(Some(VertexId(1)), false, Selection::vertices_mut);
        assert!(!selection.apply_click(Some(VertexId(1)), false, Selection::vertices_mut));
        assert_eq!(selection.count(EditMode::Vertex), 1);
    }

    #[test]
    fn additive_click_toggles_components() {
        let mut selection = Selection::new();

        selection.apply_click(Some(VertexId(1)), true, Selection::vertices_mut);
        selection.apply_click(Some(VertexId(2)), true, Selection::vertices_mut);
        assert_eq!(selection.count(EditMode::Vertex), 2);

        assert!(selection.apply_click(Some(VertexId(1)), true, Selection::vertices_mut));
        assert!(!selection.is_vertex_selected(VertexId(1)));
        assert_eq!(selection.count(EditMode::Vertex), 1);
    }

    #[test]
    fn clicking_empty_space_clears_but_shift_clicking_it_keeps() {
        let mut selection = Selection::new();
        selection.apply_click(Some(FaceId(3)), false, Selection::faces_mut);

        assert!(!selection.apply_click(None, true, Selection::faces_mut));
        assert_eq!(selection.count(EditMode::Face), 1);

        assert!(selection.apply_click(None, false, Selection::faces_mut));
        assert_eq!(selection.count(EditMode::Face), 0);
    }

    #[test]
    fn switching_modes_drops_the_other_kinds() {
        let mut selection = Selection::new();
        selection.apply_click(Some(VertexId(0)), false, Selection::vertices_mut);
        selection.apply_click(Some(EdgeId(0, 1)), false, Selection::edges_mut);
        selection.apply_click(Some(FaceId(2)), false, Selection::faces_mut);
        selection.apply_object_click(true, false);

        selection.retain_mode(EditMode::Edge);
        assert_eq!(selection.count(EditMode::Vertex), 0);
        assert_eq!(selection.count(EditMode::Edge), 1);
        assert_eq!(selection.count(EditMode::Face), 0);
        assert_eq!(selection.count(EditMode::Object), 0);
    }

    #[test]
    fn object_click_selects_then_shift_click_deselects() {
        let mut selection = Selection::new();
        assert!(selection.apply_object_click(true, false));
        assert!(selection.is_object_selected());
        assert!(
            !selection.apply_object_click(true, false),
            "already selected"
        );

        assert!(selection.apply_object_click(true, true));
        assert!(!selection.is_object_selected());
    }

    #[test]
    fn clicking_empty_space_deselects_the_object() {
        let mut selection = Selection::new();
        selection.apply_object_click(true, false);
        assert!(selection.apply_object_click(false, false));
        assert!(!selection.is_object_selected());
    }
}
