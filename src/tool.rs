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
    /// Rotate tool (E key). Turns the selection about the axis of the arm that is dragged.
    Rotate,
    /// Scale tool (R key). Stretches the selection along the axis of the arm that is dragged.
    Scale,
}

impl Tool {
    /// Whether this tool draws and drags a transform handle at the selection.
    ///
    /// `Select` is the only one that does not: it is the tool for saying *which* components, and a handle
    /// sitting over the picking would take the click the user meant for the geometry. The three transform
    /// tools all grab an arm of the same [`crate::gizmo::Frame`] and differ only in what the cursor's
    /// movement means once it is held, which is why they share one gizmo rather than having three.
    pub fn has_gizmo(self) -> bool {
        !matches!(self, Tool::Select)
    }

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
    fn every_transform_tool_offers_a_handle_and_select_does_not() {
        // The gizmo is drawn, picked and dragged on this predicate, so a tool that answers wrong either
        // loses its manipulator or sits over the geometry the Select tool is trying to pick.
        assert!(!Tool::Select.has_gizmo());
        assert!(Tool::Translate.has_gizmo());
        assert!(Tool::Rotate.has_gizmo());
        assert!(Tool::Scale.has_gizmo());
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
