//! UI layer (egui). It never mutates application state directly: `UiState::draw` returns what
//! the user asked for (for example a new `EditMode`) and the caller applies it.

use crate::scene::EditMode;
use crate::tool::Tool;
use egui::{Color32, Pos2, Rect, Vec2};

/// The user intents returned by `UiState::draw`.
///
/// Following the project architecture, `ui.rs` never mutates application state directly; it returns
/// the user's intent to `main.rs`, which applies it to `Scene`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UiResponse {
    pub mode: Option<EditMode>,
    pub tool: Option<Tool>,
    /// The Undo button was clicked; the caller performs the undo.
    pub undo: bool,
    /// The Redo button was clicked; the caller performs the redo.
    pub redo: bool,
}

/// Everything the UI displays for this frame, handed to [`UiState::draw`].
///
/// A struct rather than a long positional argument list: the two booleans and two option labels
/// beside each other would be far too easy to pass in the wrong order to notice in a call site.
#[derive(Clone, Copy, Debug)]
pub struct UiContext<'a> {
    pub mode: EditMode,
    pub tool: Tool,
    pub selected_count: usize,
    /// Whether there is an action to step back / forward, so the buttons can be greyed out.
    pub can_undo: bool,
    pub can_redo: bool,
    /// Name of the action Undo would revert ("Move Vertex"), shown on the button Maya-style.
    pub undo_label: Option<&'a str>,
    pub redo_label: Option<&'a str>,
}

/// Button text Maya-style: plain "Undo" when there is nothing to step back to, "Undo - Move Vertex"
/// when there is, so the user can see what the hotkey will take away before pressing it.
fn history_button_text(prefix: &str, label: Option<&str>) -> String {
    match label {
        Some(label) => format!("{prefix} - {label}"),
        None => prefix.to_owned(),
    }
}

const PIE_RADIUS: f32 = 70.0;

/// How far the pointer must travel from the menu center before a slot counts as flicked toward.
/// Releasing Space inside this radius closes the menu without changing anything.
const PIE_DEAD_ZONE: f32 = 25.0;

/// The pie slots with their offsets from the menu center (Top, Right, Bottom, Left).
fn pie_slots() -> [(EditMode, Vec2); 4] {
    [
        (EditMode::Edge, Vec2::new(0.0, -PIE_RADIUS)),
        (EditMode::Object, Vec2::new(PIE_RADIUS, 0.0)),
        (EditMode::Face, Vec2::new(0.0, PIE_RADIUS)),
        (EditMode::Vertex, Vec2::new(-PIE_RADIUS, 0.0)),
    ]
}

/// The slot that a pointer offset (relative to the menu center) points toward, or `None` while
/// it is still inside the dead zone. Only the direction matters, not the distance, so a quick
/// flick works even if the pointer never reaches the button.
fn slot_toward(offset: Vec2) -> Option<EditMode> {
    if offset.length() < PIE_DEAD_ZONE {
        return None;
    }
    pie_slots()
        .into_iter()
        .max_by(|a, b| {
            let score = |slot: &(EditMode, Vec2)| slot.1.normalized().dot(offset);
            score(a).total_cmp(&score(b))
        })
        .map(|(mode, _)| mode)
}

pub struct UiState {
    pub pie_menu_active: bool,
    pub pie_menu_center: Pos2,
}

impl UiState {
    pub fn new() -> Self {
        Self {
            pie_menu_active: false,
            pie_menu_center: Pos2::ZERO,
        }
    }

    /// Draws the UI for this frame. Returns the requested intents (mode/tool change, undo/redo) if any.
    pub fn draw(&mut self, ctx: &egui::Context, current: UiContext<'_>) -> UiResponse {
        let UiContext {
            mode: current_mode,
            tool: current_tool,
            selected_count,
            can_undo,
            can_redo,
            undo_label,
            redo_label,
        } = current;
        let mut response = UiResponse::default();

        // 1. Check if Right Mouse Button (without Alt, Maya-style) or Spacebar is held
        let rmb_down = ctx
            .input(|i| i.pointer.button_down(egui::PointerButton::Secondary) && !i.modifiers.alt);
        let space_down = ctx.input(|i| i.key_down(egui::Key::Space));
        let menu_down = rmb_down || space_down;

        // 2. Get the exact screen coordinates of the mouse (None if it has left the window)
        let pointer = ctx.input(|i| i.pointer.latest_pos());

        // 3. Lock the center position the moment RMB or Space is pressed. With no known pointer position
        // there is nowhere to center the menu, so it stays closed until the pointer is back.
        if menu_down && !self.pie_menu_active {
            if let Some(pos) = pointer {
                self.pie_menu_active = true;
                self.pie_menu_center = pos;
            }
        } else if !menu_down {
            // Releasing RMB or Space commits the slot the pointer was flicked toward (Maya-style).
            // Releasing inside the dead zone, with the pointer outside the window, or after the
            // window lost focus (Alt-Tab makes winit synthesize a key release) cancels.
            if self.pie_menu_active && ctx.input(|i| i.focused) {
                response.mode = pointer.and_then(|p| slot_toward(p - self.pie_menu_center));
            }
            self.pie_menu_active = false;
        }

        if self.pie_menu_active {
            // An Area allows us to place elements using absolute screen coordinates
            egui::Area::new(egui::Id::new("pie_menu_area"))
                .fixed_pos(Pos2::ZERO)
                .order(egui::Order::Foreground)
                .show(ctx, |ui| {
                    let center = self.pie_menu_center;

                    // Draw the Maya-style dark translucent background circle
                    ui.painter().circle_filled(
                        center,
                        PIE_RADIUS + 30.0,
                        Color32::from_black_alpha(200),
                    );

                    // Highlight the slot being flicked toward, or the current mode if none
                    let highlighted = pointer
                        .and_then(|p| slot_toward(p - center))
                        .unwrap_or(current_mode);

                    for (mode, offset) in pie_slots() {
                        let btn_center = center + offset;
                        let btn_size = Vec2::new(60.0, 25.0);
                        let btn_rect = Rect::from_center_size(btn_center, btn_size);

                        // Force the button to render at the exact mathematical coordinate
                        let button = egui::Button::new(mode.label()).selected(mode == highlighted);
                        if ui.put(btn_rect, button).clicked() {
                            response.mode = Some(mode);
                        }
                    }

                    // Draw a swipe-tracking line from the menu center to the current mouse position.
                    // Hidden while the mouse is outside the window: there is no position to point at.
                    if let Some(pos) = pointer {
                        ui.painter().line_segment(
                            [center, pos],
                            egui::Stroke::new(2.0_f32, Color32::from_white_alpha(100)),
                        );
                    }
                });
        } else {
            // A persistent hint window when the menu is closed
            egui::Window::new("PolyCraft Tools").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Tool:");
                    if ui
                        .selectable_label(current_tool == Tool::Select, "Select (Q)")
                        .clicked()
                    {
                        response.tool = Some(Tool::Select);
                    }
                    if ui
                        .selectable_label(current_tool == Tool::Translate, "Move (W)")
                        .clicked()
                    {
                        response.tool = Some(Tool::Translate);
                    }
                    if ui
                        .selectable_label(current_tool == Tool::Rotate, "Rotate (E)")
                        .clicked()
                    {
                        response.tool = Some(Tool::Rotate);
                    }
                    if ui
                        .selectable_label(current_tool == Tool::Scale, "Scale (R)")
                        .clicked()
                    {
                        response.tool = Some(Tool::Scale);
                    }
                });
                ui.separator();
                ui.horizontal(|ui| {
                    // The buttons report intent only; the caller owns the history and applies it.
                    let undo_button = egui::Button::new(history_button_text("Undo", undo_label));
                    if ui.add_enabled(can_undo, undo_button).clicked() {
                        response.undo = true;
                    }
                    let redo_button = egui::Button::new(history_button_text("Redo", redo_label));
                    if ui.add_enabled(can_redo, redo_button).clicked() {
                        response.redo = true;
                    }
                });
                ui.separator();
                ui.label(format!("Mode: {}", current_mode.label()));
                let count_str = match current_mode {
                    EditMode::Object => {
                        if selected_count > 0 {
                            "Selected: 1 object"
                        } else {
                            "Selected: None"
                        }
                    }
                    EditMode::Vertex => &format!("Selected: {} vertices", selected_count),
                    EditMode::Edge => &format!("Selected: {} edges", selected_count),
                    EditMode::Face => &format!("Selected: {} faces", selected_count),
                };
                ui.label(count_str);
                ui.separator();
                ui.label("Hold RMB or SPACE to open the Pie Menu.");
                ui.label("Hotkeys: Q: Select | W: Move | E: Rotate | R: Scale");
                ui.label("Ctrl+Z: Undo | Ctrl+Y / Ctrl+Shift+Z: Redo");
            });
        }

        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inside_dead_zone_cancels() {
        assert_eq!(slot_toward(Vec2::ZERO), None);
        assert_eq!(slot_toward(Vec2::new(PIE_DEAD_ZONE - 1.0, 0.0)), None);
    }

    #[test]
    fn flick_picks_slot_by_direction() {
        // Screen space: +x is right, +y is down.
        assert_eq!(slot_toward(Vec2::new(0.0, -30.0)), Some(EditMode::Edge));
        assert_eq!(slot_toward(Vec2::new(30.0, 0.0)), Some(EditMode::Object));
        assert_eq!(slot_toward(Vec2::new(0.0, 30.0)), Some(EditMode::Face));
        assert_eq!(slot_toward(Vec2::new(-30.0, 0.0)), Some(EditMode::Vertex));
    }

    #[test]
    fn distance_and_diagonals() {
        // Overshooting far past the button still picks the same slot.
        assert_eq!(slot_toward(Vec2::new(-400.0, 20.0)), Some(EditMode::Vertex));
        // Diagonals go to whichever axis dominates.
        assert_eq!(slot_toward(Vec2::new(40.0, -60.0)), Some(EditMode::Edge));
        assert_eq!(slot_toward(Vec2::new(60.0, 40.0)), Some(EditMode::Object));
    }

    // Frame-level tests: drive `UiState::draw` through egui with synthetic input, no window needed.

    fn space_event(pressed: bool) -> egui::Event {
        egui::Event::Key {
            key: egui::Key::Space,
            physical_key: Some(egui::Key::Space),
            pressed,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }
    }

    /// Frame state every pie-menu test draws with: nothing to undo, so the history buttons stay inert
    /// and cannot interfere with the mode intents the tests assert on.
    fn test_context() -> UiContext<'static> {
        UiContext {
            mode: EditMode::Object,
            tool: Tool::Translate,
            selected_count: 0,
            can_undo: false,
            can_redo: false,
            undo_label: None,
            redo_label: None,
        }
    }

    /// Runs one egui frame; returns what `draw` reported and the shapes it painted.
    fn run_frame_full(
        ctx: &egui::Context,
        ui_state: &mut UiState,
        events: Vec<egui::Event>,
        focused: bool,
    ) -> (UiResponse, Vec<egui::epaint::ClippedShape>) {
        run_frame_with(ctx, ui_state, events, focused, test_context())
    }

    /// `run_frame_full` with the displayed state supplied by the caller, for the tests that need the
    /// history buttons live.
    fn run_frame_with(
        ctx: &egui::Context,
        ui_state: &mut UiState,
        events: Vec<egui::Event>,
        focused: bool,
        context: UiContext<'_>,
    ) -> (UiResponse, Vec<egui::epaint::ClippedShape>) {
        let mut picked = UiResponse::default();
        let mut modifiers = egui::Modifiers::NONE;
        for ev in &events {
            match ev {
                egui::Event::PointerButton { modifiers: m, .. }
                | egui::Event::Key { modifiers: m, .. } => {
                    modifiers = *m;
                }
                _ => {}
            }
        }
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0))),
            focused,
            events,
            modifiers,
            ..Default::default()
        };
        let output = ctx.run(input, |ctx| {
            picked = ui_state.draw(ctx, context);
        });
        (picked, output.shapes)
    }

    /// Runs one egui frame with the given events and returns what `draw` reported.
    fn run_frame(
        ctx: &egui::Context,
        ui_state: &mut UiState,
        events: Vec<egui::Event>,
        focused: bool,
    ) -> Option<EditMode> {
        run_frame_full(ctx, ui_state, events, focused).0.mode
    }

    /// Opens the menu with the pointer at `center`.
    fn open_menu(ctx: &egui::Context, ui_state: &mut UiState, center: Pos2) {
        let open = vec![egui::Event::PointerMoved(center), space_event(true)];
        run_frame_full(ctx, ui_state, open, true);
    }

    /// Endpoints of the first line segment among the painted shapes, if any. The swipe-tracking
    /// line is the only line segment the pie menu paints.
    fn tracking_line(shapes: &[egui::epaint::ClippedShape]) -> Option<[Pos2; 2]> {
        fn find(shape: &egui::epaint::Shape) -> Option<[Pos2; 2]> {
            match shape {
                egui::epaint::Shape::LineSegment { points, .. } => Some(*points),
                egui::epaint::Shape::Vec(inner) => inner.iter().find_map(find),
                _ => None,
            }
        }
        shapes.iter().find_map(|clipped| find(&clipped.shape))
    }

    /// Opens the menu at (300, 300), moves the pointer by `flick`, then releases Space.
    fn flick_and_release(flick: Vec2, focused_on_release: bool) -> Option<EditMode> {
        let ctx = egui::Context::default();
        let mut ui_state = UiState::new();
        let center = Pos2::new(300.0, 300.0);

        let open = vec![egui::Event::PointerMoved(center), space_event(true)];
        assert_eq!(run_frame(&ctx, &mut ui_state, open, true), None);

        let flick_to = vec![egui::Event::PointerMoved(center + flick)];
        let flicked = run_frame(&ctx, &mut ui_state, flick_to, true);
        assert_eq!(
            flicked, None,
            "nothing is committed until Space is released"
        );

        run_frame(
            &ctx,
            &mut ui_state,
            vec![space_event(false)],
            focused_on_release,
        )
    }

    #[test]
    fn releasing_space_after_a_flick_selects_the_slot() {
        assert_eq!(
            flick_and_release(Vec2::new(40.0, 0.0), true),
            Some(EditMode::Object)
        );
        assert_eq!(
            flick_and_release(Vec2::new(-40.0, 0.0), true),
            Some(EditMode::Vertex)
        );
        assert_eq!(
            flick_and_release(Vec2::new(0.0, -40.0), true),
            Some(EditMode::Edge)
        );
        assert_eq!(
            flick_and_release(Vec2::new(0.0, 40.0), true),
            Some(EditMode::Face)
        );
    }

    #[test]
    fn releasing_space_in_the_dead_zone_cancels() {
        assert_eq!(flick_and_release(Vec2::new(10.0, 0.0), true), None);
    }

    #[test]
    fn losing_focus_with_the_menu_open_does_not_select() {
        // Alt-Tab: winit synthesizes a key release and egui reports `focused == false`.
        assert_eq!(flick_and_release(Vec2::new(40.0, 0.0), false), None);
    }

    #[test]
    fn tracking_line_follows_the_pointer() {
        let ctx = egui::Context::default();
        let mut ui_state = UiState::new();
        let center = Pos2::new(300.0, 300.0);
        open_menu(&ctx, &mut ui_state, center);

        let pointer = Pos2::new(340.0, 300.0);
        let moved = vec![egui::Event::PointerMoved(pointer)];
        let (_, shapes) = run_frame_full(&ctx, &mut ui_state, moved, true);
        assert_eq!(tracking_line(&shapes), Some([center, pointer]));
    }

    #[test]
    fn tracking_line_is_hidden_when_the_pointer_leaves_the_window() {
        let ctx = egui::Context::default();
        let mut ui_state = UiState::new();
        open_menu(&ctx, &mut ui_state, Pos2::new(300.0, 300.0));

        let gone = vec![egui::Event::PointerGone];
        let (_, shapes) = run_frame_full(&ctx, &mut ui_state, gone, true);
        assert_eq!(tracking_line(&shapes), None);
    }

    #[test]
    fn menu_stays_closed_while_the_pointer_position_is_unknown() {
        let ctx = egui::Context::default();
        let mut ui_state = UiState::new();
        run_frame_full(&ctx, &mut ui_state, vec![space_event(true)], true);
        assert!(!ui_state.pie_menu_active);
    }

    #[test]
    fn releasing_space_with_the_pointer_outside_the_window_cancels() {
        let ctx = egui::Context::default();
        let mut ui_state = UiState::new();
        let center = Pos2::new(300.0, 300.0);
        open_menu(&ctx, &mut ui_state, center);

        let flick = vec![egui::Event::PointerMoved(center + Vec2::new(40.0, 0.0))];
        run_frame(&ctx, &mut ui_state, flick, true);

        // The pointer leaves the window while Space is still held, then Space is released.
        run_frame(&ctx, &mut ui_state, vec![egui::Event::PointerGone], true);
        let released = run_frame(&ctx, &mut ui_state, vec![space_event(false)], true);
        assert_eq!(released, None);
    }

    fn rmb_event(pos: Pos2, pressed: bool, alt: bool) -> egui::Event {
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Secondary,
            pressed,
            modifiers: if alt {
                egui::Modifiers::ALT
            } else {
                egui::Modifiers::NONE
            },
        }
    }

    #[test]
    fn rmb_flick_and_release_selects_slot() {
        let ctx = egui::Context::default();
        let mut ui_state = UiState::new();
        let center = Pos2::new(300.0, 300.0);

        // Press RMB at center
        let press = vec![rmb_event(center, true, false)];
        assert_eq!(run_frame(&ctx, &mut ui_state, press, true), None);
        assert!(ui_state.pie_menu_active);

        // Move to the right (+X is Object mode)
        let flick_pos = center + Vec2::new(40.0, 0.0);
        let flick = vec![egui::Event::PointerMoved(flick_pos)];
        assert_eq!(run_frame(&ctx, &mut ui_state, flick, true), None);

        // Release RMB
        let release = vec![rmb_event(flick_pos, false, false)];
        let picked = run_frame(&ctx, &mut ui_state, release, true);
        assert_eq!(picked, Some(EditMode::Object));
        assert!(!ui_state.pie_menu_active);
    }

    #[test]
    fn alt_rmb_does_not_open_pie_menu() {
        let ctx = egui::Context::default();
        let mut ui_state = UiState::new();
        let center = Pos2::new(300.0, 300.0);

        // Press RMB with Alt held (camera navigation)
        let press = vec![rmb_event(center, true, true)];
        assert_eq!(run_frame(&ctx, &mut ui_state, press, true), None);
        assert!(
            !ui_state.pie_menu_active,
            "Alt+RMB is camera zoom, must not open pie menu"
        );
    }

    #[test]
    fn tools_window_displays_and_returns_tool_choice() {
        let ctx = egui::Context::default();
        let mut ui_state = UiState::new();

        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0))),
            focused: true,
            ..Default::default()
        };

        let mut response = UiResponse::default();
        let _ = ctx.run(input, |ctx| {
            response = ui_state.draw(ctx, test_context());
        });

        // When not clicked, returns None for mode and tool
        assert_eq!(response.mode, None);
        assert_eq!(response.tool, None);
    }

    #[test]
    fn history_buttons_are_inert_until_the_action_exists() {
        // Nothing to step back to: the response carries no history intent even on an empty frame.
        let ctx = egui::Context::default();
        let mut ui_state = UiState::new();
        let (response, _) = run_frame_full(&ctx, &mut ui_state, Vec::new(), true);
        assert!(!response.undo);
        assert!(!response.redo);
    }

    #[test]
    fn button_labels_name_the_action_they_would_revert() {
        assert_eq!(history_button_text("Undo", None), "Undo");
        assert_eq!(
            history_button_text("Undo", Some("Move Vertex")),
            "Undo - Move Vertex",
            "the user should be able to read what Ctrl+Z will take away before pressing it"
        );
        assert_eq!(
            history_button_text("Redo", Some("Move Edge")),
            "Redo - Move Edge"
        );
    }

    /// Where `wanted` was painted this frame, as a clickable point inside it.
    fn painted_text_center(shapes: &[egui::epaint::ClippedShape], wanted: &str) -> Option<Pos2> {
        fn find(shape: &egui::epaint::Shape, wanted: &str) -> Option<Pos2> {
            match shape {
                egui::epaint::Shape::Text(text) => {
                    let galley = &text.galley;
                    // `galley.rect` is in galley-local space (`rect.top()` is always 0), so the
                    // on-screen point is the shape's position plus that local offset.
                    (galley.job.text == wanted).then(|| text.pos + galley.rect.center().to_vec2())
                }
                egui::epaint::Shape::Vec(inner) => inner.iter().find_map(|s| find(s, wanted)),
                _ => None,
            }
        }
        shapes
            .iter()
            .find_map(|clipped| find(&clipped.shape, wanted))
    }

    /// Presses and releases the primary button on the button labelled `wanted`, one egui frame apart
    /// like real input, and reports what the frame that took the release returned. The click point is
    /// read back out of the painted label rather than hardcoded, so the test survives the panel being
    /// moved or the buttons changing size. Panics if no such label was painted.
    fn click_button_labelled(context: UiContext<'_>, wanted: &str) -> UiResponse {
        let ctx = egui::Context::default();
        let mut ui_state = UiState::new();

        // egui cannot lay out text on the very first frame: it hands the backend the font texture in
        // `textures_delta` and only measures glyphs once the backend reports that texture back, so
        // every label on frame one paints as `Shape::Noop`. Draw a throwaway frame first to get the
        // fonts in, or there would be no label to find.
        run_frame_with(&ctx, &mut ui_state, Vec::new(), true, context);

        let (_, shapes) = run_frame_with(&ctx, &mut ui_state, Vec::new(), true, context);
        let center = painted_text_center(&shapes, wanted)
            .unwrap_or_else(|| panic!("no button labelled {wanted:?} was painted"));

        let hover = vec![egui::Event::PointerMoved(center)];
        run_frame_with(&ctx, &mut ui_state, hover, true, context);

        let press = vec![egui::Event::PointerButton {
            pos: center,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        }];
        let pressed = run_frame_with(&ctx, &mut ui_state, press, true, context).0;
        assert!(
            !pressed.undo && !pressed.redo,
            "pressing is not enough; the action belongs on release"
        );

        let release = vec![
            egui::Event::PointerMoved(center),
            egui::Event::PointerButton {
                pos: center,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ];
        run_frame_with(&ctx, &mut ui_state, release, true, context).0
    }

    /// A frame state with one "Move Vertex" action waiting to be undone, and nothing to redo.
    fn undoable_context() -> UiContext<'static> {
        UiContext {
            mode: EditMode::Vertex,
            tool: Tool::Translate,
            selected_count: 1,
            can_undo: true,
            can_redo: false,
            undo_label: Some("Move Vertex"),
            redo_label: None,
        }
    }

    /// The same with the branches swapped: nothing to undo, one action to redo.
    fn redoable_context() -> UiContext<'static> {
        UiContext {
            can_undo: false,
            can_redo: true,
            undo_label: None,
            redo_label: Some("Move Vertex"),
            ..undoable_context()
        }
    }

    #[test]
    fn clicking_undo_reports_only_the_undo_intent() {
        let response = click_button_labelled(undoable_context(), "Undo - Move Vertex");
        assert!(response.undo, "the enabled Undo button must report intent");
        assert!(!response.redo, "the greyed Redo button must stay inert");
    }

    #[test]
    fn clicking_redo_reports_only_the_redo_intent() {
        let response = click_button_labelled(redoable_context(), "Redo - Move Vertex");
        assert!(response.redo);
        assert!(!response.undo);
    }

    #[test]
    fn a_greyed_out_history_button_does_not_report_intent() {
        // A disabled widget still paints its label, so this click lands on the button and must be
        // swallowed by `add_enabled` rather than reaching the caller's history.
        let response = click_button_labelled(test_context(), "Undo");
        assert!(
            !response.undo,
            "clicking a disabled Undo must not step the document back"
        );
        assert!(!response.redo);
    }

    #[test]
    fn clicking_a_history_button_does_not_change_the_mode_or_tool() {
        let response = click_button_labelled(undoable_context(), "Undo - Move Vertex");
        assert_eq!(response.mode, None);
        assert_eq!(response.tool, None);
    }
}
