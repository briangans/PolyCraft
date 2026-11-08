//! UI layer (egui). It never mutates application state directly: `UiState::draw` returns what
//! the user asked for (for example a new `EditMode`) and the caller applies it.

use crate::scene::EditMode;
use egui::{Color32, Pos2, Rect, Vec2};

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

    /// Draws the UI for this frame. Returns the new edit mode if the user picked one.
    pub fn draw(&mut self, ctx: &egui::Context, current_mode: EditMode) -> Option<EditMode> {
        let mut picked_mode = None;

        // 1. Check if Spacebar is held
        let space_down = ctx.input(|i| i.key_down(egui::Key::Space));

        // 2. Get the exact screen coordinates of the mouse (None if it has left the window)
        let pointer = ctx.input(|i| i.pointer.latest_pos());

        // 3. Lock the center position the moment Space is pressed. With no known pointer position
        // there is nowhere to center the menu, so it stays closed until the pointer is back.
        if space_down && !self.pie_menu_active {
            if let Some(pos) = pointer {
                self.pie_menu_active = true;
                self.pie_menu_center = pos;
            }
        } else if !space_down {
            // Releasing Space commits the slot the pointer was flicked toward (Maya-style).
            // Releasing inside the dead zone, with the pointer outside the window, or after the
            // window lost focus (Alt-Tab makes winit synthesize a key release) cancels.
            if self.pie_menu_active && ctx.input(|i| i.focused) {
                picked_mode = pointer.and_then(|p| slot_toward(p - self.pie_menu_center));
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
                            picked_mode = Some(mode);
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
                ui.label(format!("Mode: {}", current_mode.label()));
                ui.label("Hold SPACE to open the Pie Menu.");
            });
        }

        picked_mode
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

    /// Runs one egui frame; returns what `draw` reported and the shapes it painted.
    fn run_frame_full(
        ctx: &egui::Context,
        ui_state: &mut UiState,
        events: Vec<egui::Event>,
        focused: bool,
    ) -> (Option<EditMode>, Vec<egui::epaint::ClippedShape>) {
        let mut picked = None;
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0))),
            focused,
            events,
            ..Default::default()
        };
        let output = ctx.run(input, |ctx| {
            picked = ui_state.draw(ctx, EditMode::Object);
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
        run_frame_full(ctx, ui_state, events, focused).0
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
}
