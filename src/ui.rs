use egui::{Color32, Pos2, Rect, Vec2};

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

    pub fn draw(&mut self, ctx: &egui::Context) {
        // 1. Check if Spacebar is held
        let space_down = ctx.input(|i| i.key_down(egui::Key::Space));

        // 2. Get the exact screen coordinates of the mouse
        let pointer_pos = ctx.input(|i| i.pointer.latest_pos()).unwrap_or(Pos2::ZERO);

        // 3. Lock the center position the moment Space is pressed
        if space_down && !self.pie_menu_active {
            self.pie_menu_active = true;
            self.pie_menu_center = pointer_pos;
        } else if !space_down {
            self.pie_menu_active = false;
        }

        if self.pie_menu_active {
            // An Area allows us to place elements using absolute screen coordinates
            egui::Area::new(egui::Id::new("pie_menu_area"))
                .fixed_pos(Pos2::ZERO)
                .order(egui::Order::Foreground)
                .show(ctx, |ui| {
                    let center = self.pie_menu_center;
                    let radius = 70.0;

                    // Draw the Maya-style dark translucent background circle
                    ui.painter().circle_filled(
                        center,
                        radius + 30.0,
                        Color32::from_black_alpha(200),
                    );

                    // Map our Maya component modes to coordinates (Top, Right, Bottom, Left)
                    let buttons = [
                        ("Edge", Vec2::new(0.0, -radius)),
                        ("Object", Vec2::new(radius, 0.0)),
                        ("Face", Vec2::new(0.0, radius)),
                        ("Vertex", Vec2::new(-radius, 0.0)),
                    ];

                    for (label, offset) in buttons {
                        let btn_center = center + offset;
                        let btn_size = Vec2::new(60.0, 25.0);
                        let btn_rect = Rect::from_center_size(btn_center, btn_size);

                        // Force the button to render at the exact mathematical coordinate
                        if ui.put(btn_rect, egui::Button::new(label)).clicked() {
                            println!("Switched to {} mode!", label);
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
                ui.label("Hold SPACE to open the Pie Menu.");
            });
        }
    }
}
