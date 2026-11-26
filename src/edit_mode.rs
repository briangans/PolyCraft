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

    /// The mode's name as a plural, for command echoes: "Selected 12 edges".
    ///
    /// Kept beside [`EditMode::label`] rather than spelled out at each call site so a mode added later
    /// cannot arrive with a notice that still says the old thing, and so the `s` is not hand-appended
    /// where it reads as "Vertexs".
    pub fn plural_label(self) -> &'static str {
        match self {
            EditMode::Object => "Objects",
            EditMode::Vertex => "Vertices",
            EditMode::Edge => "Edges",
            EditMode::Face => "Faces",
        }
    }

    /// True for the modes that work on mesh components rather than the whole object.
    pub fn is_component_mode(self) -> bool {
        !matches!(self, EditMode::Object)
    }
}
