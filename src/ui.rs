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
        let pointer_pos = pointer.unwrap_or(Pos2::ZERO);

        // 3. Lock the center position the moment Space is pressed
        if space_down && !self.pie_menu_active {
            self.pie_menu_active = true;
            self.pie_menu_center = pointer_pos;
        } else if !space_down {
            // Releasing Space commits the slot the pointer was flicked toward (Maya-style).
            // Releasing inside the dead zone, or with the pointer outside the window, cancels.
            if self.pie_menu_active {
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

                    // Draw a swipe-tracking line from the menu center to the current mouse position
                    ui.painter().line_segment(
                        [center, pointer_pos],
                        egui::Stroke::new(2.0, Color32::from_white_alpha(100)),
                    );
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
}
