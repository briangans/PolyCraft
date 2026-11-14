//! Maya-style active tool (Select, Move, Rotate, Scale), bound to Q / W / E / R.
//!
//! Tools are distinct from `EditMode`:
//! - `EditMode` defines *what* components can be selected (Object, Vertex, Edge, Face).
//! - `Tool` defines *how* the active selection is manipulated (Select, Translate, Rotate, Scale).
//!
//! Switching tools does not alter the current selection or edit mode, matching Maya's interaction model.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Tool {
    /// Pure selection tool (Q key). No transform gizmo is drawn.
    Select,
    /// Move / Translate tool (W key). Draws the 3D translate gizmo at the selection pivot.
    #[default]
    Translate,
    /// Rotate tool (E key). Reserved for future rotation manipulator.
    Rotate,
    /// Scale tool (R key). Reserved for future scale manipulator.
    Scale,
}

impl Tool {
    /// Human-readable name for UI display.
    pub fn label(self) -> &'static str {
        match self {
            Tool::Select => "Select",
            Tool::Translate => "Move",
            Tool::Rotate => "Rotate",
            Tool::Scale => "Scale",
        }
    }

    /// Hotkey shortcut associated with this tool in Maya.
    pub fn shortcut(self) -> &'static str {
        match self {
            Tool::Select => "Q",
            Tool::Translate => "W",
            Tool::Rotate => "E",
            Tool::Scale => "R",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tool_is_translate() {
        assert_eq!(Tool::default(), Tool::Translate);
    }

    #[test]
    fn labels_and_shortcuts() {
        assert_eq!(Tool::Select.label(), "Select");
        assert_eq!(Tool::Select.shortcut(), "Q");
        assert_eq!(Tool::Translate.label(), "Move");
        assert_eq!(Tool::Translate.shortcut(), "W");
        assert_eq!(Tool::Rotate.label(), "Rotate");
        assert_eq!(Tool::Rotate.shortcut(), "E");
        assert_eq!(Tool::Scale.label(), "Scale");
        assert_eq!(Tool::Scale.shortcut(), "R");
    }
}
