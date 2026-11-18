mod camera;
pub mod edit_mode;
mod gizmo;
mod mesh;
mod overlay;
pub mod picking;
pub mod render_types;
mod renderer;
mod scene;
pub mod selection;
pub mod tool;
mod ui;

use renderer::State;
use std::sync::Arc;
use tool::Tool;
use winit::{
    event::{DeviceEvent, Event, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, ModifiersState, PhysicalKey},
    window::WindowBuilder,
};

async fn run() {
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("PolyCraft Viewport")
            .build(&event_loop)
            .unwrap(),
    );
    let mut state = State::new(window.clone()).await;

    let _ = event_loop.run(move |event, elwt| {
        match event {
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                state.process_mouse_motion(delta.0, delta.1);
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window().id() => {
                let response = state.on_window_event(event);

                match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(physical_size) => state.resize(*physical_size),
                    WindowEvent::CursorMoved { position, .. } => {
                        state.handle_cursor_moved(*position);
                    }
                    WindowEvent::ModifiersChanged(modifiers) => {
                        state.camera_controller_mut().alt_pressed =
                            modifiers.state().contains(ModifiersState::ALT);
                        state.set_shift_pressed(modifiers.state().contains(ModifiersState::SHIFT));
                    }
                    WindowEvent::KeyboardInput {
                        event:
                            winit::event::KeyEvent {
                                state: winit::event::ElementState::Pressed,
                                physical_key: PhysicalKey::Code(key_code),
                                repeat: false,
                                ..
                            },
                        ..
                    } => {
                        // Maya hotkeys: Q = Select, W = Move, E = Rotate, R = Scale, Esc = Cancel drag.
                        if !response.consumed && !state.wants_keyboard_input() {
                            match key_code {
                                KeyCode::KeyQ => state.scene_mut().set_tool(Tool::Select),
                                KeyCode::KeyW => state.scene_mut().set_tool(Tool::Translate),
                                KeyCode::KeyE => state.scene_mut().set_tool(Tool::Rotate),
                                KeyCode::KeyR => state.scene_mut().set_tool(Tool::Scale),
                                KeyCode::Escape => state.cancel_active_drag(),
                                _ => {}
                            }
                        }
                    }
                    WindowEvent::MouseInput {
                        state: button_state,
                        button,
                        ..
                    } => {
                        if !response.consumed {
                            state
                                .camera_controller_mut()
                                .process_mouse_input(*button_state, *button);
                            state.handle_mouse_input(*button_state, *button);
                        }
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        if !response.consumed {
                            state.process_scroll(delta);
                        }
                    }
                    WindowEvent::RedrawRequested => match state.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.size()),
                        Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                        Err(e) => eprintln!("{:?}", e),
                    },
                    _ => {}
                }
            }
            Event::AboutToWait => {
                state.window().request_redraw();
            }
            _ => {}
        }
    });
}

fn main() {
    pollster::block_on(run());
}
