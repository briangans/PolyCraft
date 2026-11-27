//! The main menu bar, expressed as data rather than widgets.
//!
//! Two reasons it is its own module. First, the menu bar is the project's honest map of what Maya can do
//! and this cannot: every Maya command in these menus is listed, the ones with no implementation behind
//! them arrive greyed out, and each greyed entry carries a note saying what is missing. A modelling
//! checklist that lives in prose goes stale; one that renders next to the viewport gets noticed. Second,
//! it keeps that list out of `ui.rs`, which is otherwise about drawing.
//!
//! Nothing here touches `egui`, `Scene` or `State`, so the whole table is unit-testable without a window,
//! and `ui.rs` maps a [`Command`] onto the same `State` methods a panel button, a pie flick or a hotkey
//! reaches. That is deliberate: those four routes cannot disagree about what a command does when they all
//! call the same function.
//!
//! # What the two label columns mean
//!
//! Titles and shortcuts are Maya's, so the menu reads as the thing being compared against. The shortcut
//! column means one thing on a live row and another on a greyed one, and both are stated rather than
//! implied:
//!
//! - on a **live** row it is the chord this app binds. Those follow Maya where Maya binds the command
//!   itself - `Q`/`W`/`E`/`R`, `Ctrl+Z`, `Ctrl+Y`, `F8` to `F11` - and are ours where Maya leaves the
//!   command on the shelf with no key, which is the case for *Extrude*, *Insert Edge Loop* and *Bevel*.
//! - on a **greyed** row it is the chord Maya uses, kept so a future binding lands where a Maya user
//!   already expects it rather than wherever a free letter turned up.
//!
//! `Ctrl+A` is the one row where the two meanings genuinely clash: it is Maya's Attribute Editor, which
//! this app has no equivalent of, so the key is free and is spent on Select All. If an attribute system
//! ever lands, that is the chord to renegotiate.

use crate::edit_mode::EditMode;
use crate::tool::Tool;

/// A command the app can actually carry out.
///
/// A variant may only be listed once a `Scene` or `State` method exists that performs it, and the menu
/// never performs anything itself. That is weaker than "must already have another route" and deliberately
/// so: Select All, Invert and the reference-system cycle arrived with this menu, and a rule forbidding
/// them would make the menu bar a report on the past instead of a way to use the app. What the rule does
/// buy is the anti-drift property - one command, one method, whichever route the user took to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Command {
    ObjectMode,
    VertexMode,
    EdgeMode,
    FaceMode,
    SelectTool,
    MoveTool,
    /// Maya's `E`. Turns the selection about the arm of the handle that is dragged.
    RotateTool,
    /// Maya's `R`. Stretches the selection along the arm of the handle that is dragged.
    ScaleTool,
    Undo,
    Redo,
    Extrude,
    EdgeLoop,
    Bevel,
    /// Maya's `Del`. Removes the selected faces, or the faces a selected corner bounds; the row keeps
    /// Maya's own `Remove` label because that is where Maya puts it.
    Delete,
    /// Maya's *Fill Hole*: caps every opening in the mesh. Not driven by the selection, which is why it
    /// follows `Delete` rather than the selection-driven commands above it.
    FillHoles,
    /// Maya's `X`, which this app borrows for the reference system - see the digest's keymap note.
    CycleReferenceSystem,
    SelectAll,
    SelectNone,
    InvertSelection,
}

/// One menu row: `(Maya's own label, Maya's shortcut, what we can do, the note)`.
///
/// The shortcut is `""` when Maya binds none, and the command is `None` when we have not built it - in
/// which case the row draws disabled and the note becomes the tooltip explaining the gap. The note is
/// empty for live rows that need no explanation.
pub type Row = (&'static str, &'static str, Option<Command>, &'static str);

pub struct Menu {
    pub title: &'static str,
    pub rows: &'static [Row],
}

/// Shorthand for a row the app cannot carry out yet.
const fn gap(label: &'static str, shortcut: &'static str, note: &'static str) -> Row {
    (label, shortcut, None, note)
}

/// Shorthand for a row that works.
const fn live(label: &'static str, shortcut: &'static str, command: Command) -> Row {
    (label, shortcut, Some(command), "")
}

/// Maya's modelling menu set, in Maya's order, with Maya's labels.
pub const MENUS: &[Menu] = &[
    Menu {
        title: "Select",
        rows: &[
            live("All", "Ctrl+A", Command::SelectAll),
            live("None", "Ctrl+Shift+A", Command::SelectNone),
            live("Invert", "Ctrl+Shift+I", Command::InvertSelection),
            gap(
                "Border",
                "",
                "the rim of an opening; boundary_edges() already finds these, nothing selects them",
            ),
            gap(
                "Edge Loop",
                "",
                "selects a ring here rather than cutting one, and no interactive ring picking exists \
                 yet - Shift+L cuts",
            ),
            gap("Edge Ring", "", "needs a ring walk across quads"),
            live("Vertex", "F9", Command::VertexMode),
            live("Edge", "F10", Command::EdgeMode),
            live("Face", "F11", Command::FaceMode),
            live("Object", "F8", Command::ObjectMode),
            // Maya keeps these on the Tool Box, the column of buttons at the left of its window, which
            // this app has no equivalent of - so the menu is the only place they can be reached. All four
            // are live: the three transform tools draw and drag one handle each, differing only in what the
            // cursor's movement means once an arm is held. `Show Manipulator` stays greyed because there is
            // nothing to show or hide - the handle follows the tool and the selection, so it has no state of
            // its own to toggle.
            live("Select Tool", "Q", Command::SelectTool),
            live("Move Tool", "W", Command::MoveTool),
            live("Rotate Tool", "E", Command::RotateTool),
            live("Scale Tool", "R", Command::ScaleTool),
            gap(
                "Show Manipulator Tool",
                "T",
                "there is no show/hide state - the gizmo follows the tool and the selection",
            ),
            gap("Hidden", "", "nothing can be hidden yet"),
            gap(
                "Grow",
                ">",
                "widens the selection by one ring; needs the neighbours of a component, which the mesh can already give",
            ),
            gap(
                "Shrink",
                "<",
                "narrows the selection by one ring, and no neighbourhood walk exists to narrow",
            ),
            gap(
                "Soft Select",
                "",
                "attenuates a drag by distance from the cursor; no per-vertex weight is stored",
            ),
            gap("Paint Selection Tool", "", "needs a falloff brush and a painted weight per vertex"),
        ],
    },
    Menu {
        title: "Edit",
        rows: &[
            live("Undo", "Ctrl+Z", Command::Undo),
            live("Redo", "Ctrl+Shift+Z", Command::Redo),
            gap(
                "Repeat Last Tool",
                "Shift+R",
                "commands are not recorded, so there is nothing to repeat",
            ),
            gap("Undo History", "", "history exists but only two steps are reachable, not the stack"),
            gap("Preferences", "Ctrl+,", "no settings are stored anywhere"),
        ],
    },
    // The third of Maya's seven always-present menus (File, Edit, Create, Select, Modify, Display,
    // Windows). Maya's entries here are pivots and transforms, and every one of them is greyed for the
    // same structural reason: a component modeller writes vertex positions outright, so there is no
    // pivot to move and no transform to bake, match or align. The one live row is the setting that
    // decides which axes a drag travels along.
    Menu {
        title: "Modify",
        rows: &[
            live("Reference System", "X", Command::CycleReferenceSystem),
            gap(
                "Center Pivot",
                "",
                "no pivot is stored; the gizmo sits on the average of the selection",
            ),
            gap(
                "Zero Pivot",
                "",
                "no pivot is stored, so there is nothing to drop on the world origin",
            ),
            gap(
                "Freeze Transformations",
                "",
                "components are moved outright, so there is no transform to bake",
            ),
            gap("Match Transformations", "", "one object per document, nothing to match against"),
            gap("Align Tool", "", "needs a transform to write and a second object to align to"),
            gap("Snap Together Tool", "", "needs snapping, a transform, and a second object"),
        ],
    },
    Menu {
        title: "Mesh",
        rows: &[
            live("Insert Edge Loop Tool", "Shift+L", Command::EdgeLoop),
            live("Extrude", "Shift+E", Command::Extrude),
            live("Bevel", "Shift+B", Command::Bevel),
            gap(
                "Multi-Cut Tool",
                "",
                "freehand cuts across a face; needs a face-splitting primitive",
            ),
            gap("Insert Edge Tool", "", "needs the same face-splitting primitive"),
            live("Delete Component", "Del", Command::Delete),
            gap(
                "Remove",
                "",
                "rebuilds the surface over a component instead of opening a hole; the same merge that \
                 Del on a selected edge would need",
            ),
            gap(
                "Merge",
                "",
                "welds coincident corners into one, which needs a merge primitive rather than a removal",
            ),
            gap("Split", "", "unwelds a corner into one per face, which needs corners that can be shared"),
            gap(
                "Extract Faces",
                "",
                "lifts faces into a second object, and a document holds no second object",
            ),
            gap("Duplicate Faces", "", "would need an object, or a second mesh"),
            live("Fill Hole", "", Command::FillHoles),
            gap(
                "Reverse",
                "",
                "flips the loop order normals are read from, which needs every twin of the face rewired",
            ),
            gap(
                "Normals > Set to Soft",
                "",
                "needs a normal per vertex; generate_render_buffers writes one flat normal per face",
            ),
            gap("Harden Edge", "", "needs per-edge soft/hard state the mesh does not hold"),
            gap("Booleans", "", "union/difference need CSG, and a second mesh to cut against"),
            gap("Combine", "", "no scene graph, so no second mesh to combine into"),
            gap("Separate", "", "no scene graph"),
            gap(
                "Cleanup...",
                "",
                "no operation leaves violations to clean: each one validates its result before installing it",
            ),
            gap("Add Divisions", "3", "needs subdivision surfaces, a subsystem of its own"),
            gap("Reduce", "", "needs edge collapse"),
            gap(
                "Triangulate",
                "",
                "renders triangles already, so nothing would change on screen; the mesh keeps quads",
            ),
            gap("Mirror", "", "needs mirroring plus a weld along the seam"),
            gap("Circularize", "", "solves a ring onto a circle; needs the ring walk first"),
        ],
    },
    Menu {
        title: "Create",
        rows: &[
            gap(
                "Polygon Cube",
                "",
                "a cube is what the document opens on; nothing can add a second",
            ),
            gap("Polygon Sphere", "", "needs triangulation and one object per primitive"),
            gap("Polygon Cylinder", "", "needs a cap fan and one object per primitive"),
            gap("Polygon Cone", "", "needs a cap fan"),
            gap(
                "Polygon Plane",
                "",
                "needs one object per primitive, though it would be the cheapest to add",
            ),
            gap("Polygon Torus", "", "needs one object per primitive"),
            gap("Polygon Prism / Pyramid / Pipe", "", "needs one object per primitive"),
            gap(
                "CV Curve Tool",
                "",
                "curves are a second geometry type, which needs its own selection and rendering",
            ),
            gap("Text", "", "needs a font mesh"),
            gap("Camera / Light", "", "one fixed orbit camera and no lighting model"),
            gap("Joint", "", "no deformation system"),
        ],
    },
    Menu {
        title: "Windows",
        rows: &[
            gap(
                "Channel Box / Layer Editor",
                "",
                "no keyed attributes and no layers to list",
            ),
            gap(
                "Attribute Editor",
                "Ctrl+A",
                "no attribute system; note this collides with Select All",
            ),
            gap("Outliner", "", "exactly one object, so nothing to list"),
            gap("Hypershade", "", "no materials - faces are tinted by selection state only"),
            gap(
                "UV Editor",
                "",
                "the mesh carries no texture coordinates",
            ),
            gap(
                "Modeling Toolkit",
                "",
                "needs the attribute panels this app has none of; the Tools panel is the stand-in",
            ),
            gap(
                "Script Editor",
                "",
                "no scripting layer; every command is compiled in",
            ),
            gap(
                "Viewport Layout: Single / 4 Panel",
                "Space",
                "one viewport, and no second layout to switch to; Maya toggles these with the spacebar",
            ),
        ],
    },
];

/// How many rows Maya has here that this app can carry out, and how many it cannot.
///
/// The pair is the project's size estimate in Maya's own terms, and the test over it is what catches a
/// command landing without its menu row being switched to live.
pub fn tally() -> (usize, usize) {
    let live = MENUS
        .iter()
        .flat_map(|menu| menu.rows.iter())
        .filter(|(_, _, command, _)| command.is_some())
        .count();
    let total = MENUS.iter().map(|menu| menu.rows.len()).sum::<usize>();
    (live, total - live)
}

/// The selection mode a row switches to, if it switches to one.
///
/// Used to tick the row that is live, the way Maya's Select menu marks the current component type. Only
/// the mode rows answer here; the tool rows answer [`tool_for`] instead, and a row that ticks is a state
/// rather than an action.
pub fn mode_for(command: Command) -> Option<EditMode> {
    match command {
        Command::ObjectMode => Some(EditMode::Object),
        Command::VertexMode => Some(EditMode::Vertex),
        Command::EdgeMode => Some(EditMode::Edge),
        Command::FaceMode => Some(EditMode::Face),
        _ => None,
    }
}

/// The transform tool a row switches to, if it switches to one.
///
/// Maya shows the active tool by lighting its Tool Box button; with no Tool Box, the tick on these rows is
/// where that state is visible. All four tool rows answer here, including the two that draw the same handle
/// Move does - a row that ticks a tool it does not switch to would be the worse bug, because the tick is how
/// the user checks which gesture the arms are about to perform.
pub fn tool_for(command: Command) -> Option<Tool> {
    match command {
        Command::SelectTool => Some(Tool::Select),
        Command::MoveTool => Some(Tool::Translate),
        Command::RotateTool => Some(Tool::Rotate),
        Command::ScaleTool => Some(Tool::Scale),
        _ => None,
    }
}

/// The tooltip for a row: what is missing when it is greyed, or the chord when it works.
///
/// A live row repeats its chord instead of carrying a sentence, because the chord is the thing a Maya user
/// checks and there is nothing to explain about something that already works.
pub fn tooltip_for(row: &Row) -> &'static str {
    let (_label, shortcut, command, note) = row;
    match command {
        Some(_) if !shortcut.is_empty() => shortcut,
        Some(_) => "This works.",
        None => note,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Words that make a greyed row's tooltip a claim about the project rather than a shrug.
    const GAP_WORDS: [&str; 8] = [
        "not", "no", "none", "nothing", "need", "needs", "lacks", "without",
    ];

    /// Below this many characters a greyed row's note is a shrug, not an explanation: "not yet" is 7 and
    /// "coming soon" is 11, while a real one names a missing thing and so runs longer - "needs a cap fan".
    const PLACEHOLDER_LIMIT: usize = 12;

    /// Every command the enum can name, spelled out by hand on purpose. A list generated from the enum
    /// could not catch the failure these tests exist for: a variant added to `Command` with no menu row,
    /// which is exactly how `SelectTool`, `MoveTool` and `CycleReferenceSystem` sat unreachable until the
    /// compiler complained about them being never constructed.
    const ALL_COMMANDS: &[Command] = &[
        Command::ObjectMode,
        Command::VertexMode,
        Command::EdgeMode,
        Command::FaceMode,
        Command::SelectTool,
        Command::MoveTool,
        Command::RotateTool,
        Command::ScaleTool,
        Command::Undo,
        Command::Redo,
        Command::Extrude,
        Command::EdgeLoop,
        Command::Bevel,
        Command::Delete,
        Command::FillHoles,
        Command::CycleReferenceSystem,
        Command::SelectAll,
        Command::SelectNone,
        Command::InvertSelection,
    ];

    fn rows() -> Vec<&'static Row> {
        MENUS.iter().flat_map(|menu| menu.rows.iter()).collect()
    }

    fn commands() -> Vec<Command> {
        rows()
            .iter()
            .filter_map(|(_, _, command, _)| *command)
            .collect()
    }

    #[test]
    fn every_command_the_enum_names_reaches_a_menu_row() {
        let listed = commands();
        let unreachable: Vec<String> = ALL_COMMANDS
            .iter()
            .filter(|command| !listed.contains(command))
            .map(|command| format!("{command:?}"))
            .collect();
        assert!(
            unreachable.is_empty(),
            "{unreachable:?} exist but no menu row offers them, so a viewer of the menu cannot tell \
             they are there; add the row or drop the variant"
        );
    }

    #[test]
    fn no_menu_row_offers_a_command_the_enum_no_longer_has() {
        // The other direction: a row kept after its variant is folded away would be a live-looking
        // nothing, which is worse to debug than a compile error because nothing complains.
        let listed = commands();
        let stray: Vec<String> = listed
            .iter()
            .filter(|command| !ALL_COMMANDS.contains(command))
            .map(|command| format!("{command:?}"))
            .collect();
        assert!(
            stray.is_empty(),
            "{stray:?} are offered by no variant that exists"
        );
    }

    #[test]
    fn a_row_either_ticks_a_state_or_fires_an_action_never_both() {
        for command in ALL_COMMANDS {
            let ticks_state = mode_for(*command).is_some() || tool_for(*command).is_some();
            let is_state = matches!(
                command,
                Command::ObjectMode
                    | Command::VertexMode
                    | Command::EdgeMode
                    | Command::FaceMode
                    | Command::SelectTool
                    | Command::MoveTool
                    | Command::RotateTool
                    | Command::ScaleTool
            );
            assert_eq!(
                ticks_state, is_state,
                "{command:?} is described as {is_state} but answered {ticks_state} about ticking itself"
            );
        }
    }

    #[test]
    fn the_tally_counts_its_own_table_rather_than_a_remembered_number() {
        let (live, missing) = tally();
        let counted = rows().len();
        assert_eq!(
            live + missing,
            counted,
            "the two halves of the tally must add up to the table"
        );
        assert_eq!(
            live,
            commands().len(),
            "live rows are exactly the rows carrying a command"
        );
        assert!(
            live > 0 && missing > 0,
            "both halves matter: a table with no gaps is not Maya, and one with nothing live is not a menu"
        );
    }

    #[test]
    fn a_live_row_offers_its_chord_and_a_greyed_one_says_what_is_missing() {
        for row in rows() {
            let (label, shortcut, command, note) = row;
            let tip = tooltip_for(row);
            if command.is_some() {
                assert!(
                    !tip.is_empty(),
                    "{label} is clickable but shows nothing, so its chord is undiscoverable"
                );
                if !shortcut.is_empty() {
                    assert_eq!(tip, *shortcut, "{label} should advertise its own chord");
                }
            } else {
                assert!(
                    note.len() > PLACEHOLDER_LIMIT,
                    "{label} is greyed with the {note:?} note, which is short enough to be a \
                     placeholder rather than an explanation"
                );
            }
        }
    }

    #[test]
    fn every_greyed_row_blames_a_real_gap_rather_than_saying_nothing() {
        // The notes are the roadmap in Maya's words, and they are also what tells a user the button is
        // not broken - so each must name the capability it is waiting on. Anything that passes while
        // reading like a shrug ("not yet", "coming soon") is the failure this guards against, which is
        // why the words are matched whole rather than as a prefix: "notation" is not a gap.
        //
        // Offenders are collected and reported together rather than asserted one at a time, because the
        // table is long and a first-failure test makes fixing it a round trip per row.
        let offenders: Vec<String> = rows()
            .into_iter()
            .filter(|(_, _, command, _)| command.is_none())
            .filter(|(_, _, _, note)| {
                !GAP_WORDS.iter().any(|word| {
                    note.split_whitespace().any(|token| {
                        token
                            .trim_end_matches(|c: char| !c.is_alphanumeric())
                            .eq(*word)
                    })
                })
            })
            .map(|(label, _, _, note)| format!("{label}: {note}"))
            .collect();

        assert!(
            offenders.is_empty(),
            "greyed rows are named here without naming a missing capability: {offenders:?}"
        );
    }

    #[test]
    fn no_menu_repeats_a_row_label() {
        for menu in MENUS {
            let mut seen: Vec<&str> = menu.rows.iter().map(|row| row.0).collect();
            let before = seen.len();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(
                seen.len(),
                before,
                "{} lists one label twice, which reads as a bug in the menu rather than in the table",
                menu.title
            );
        }
    }
}
