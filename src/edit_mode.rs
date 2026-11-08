//! Maya-style component mode, switched from the pie menu.

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

    /// True for the modes that work on mesh components rather than the whole object.
    pub fn is_component_mode(self) -> bool {
        !matches!(self, EditMode::Object)
    }
}
