//! GPU renderer, pipeline management, and viewport state.
//!
//! Encapsulates `wgpu` initialization, render passes (3D mesh, wireframe, vertex dots, transform gizmo),
//! egui immediate-mode UI rendering, and viewport interaction state.

use crate::camera::{Camera, CameraController};
use crate::edit_mode::EditMode;
use crate::gizmo::{self, GizmoAxis};
use crate::mesh::VertexId;
use crate::overlay;
use crate::picking::Ray;
use crate::render_types::{CameraUniform, GpuVertex};
use crate::scene::{Scene, EXTRUDE_FACE_DISTANCE};
use crate::tool::Tool;
use crate::ui::{UiContext, UiState};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

/// Two gizmo line vertices per axis, three axes. Allocated once and updated in-place every frame.
const GIZMO_VERTEX_COUNT: usize = GizmoAxis::ALL.len() * 2;

/// Builds the GPU vertex/index buffers for the 3D shaded mesh.
fn create_mesh_buffers(device: &wgpu::Device, scene: &Scene) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    let render_data = scene.mesh().generate_render_buffers();
    let raw_verts = render_data.vertices;
    let raw_indices = render_data.indices;
    let selection = scene.selection();
    let mode = scene.mode();

    let wgpu_vertices: Vec<GpuVertex> = raw_verts
        .iter()
        .enumerate()
        .map(|(i, render_vert)| {
            let face_id = render_data.vertex_faces[i];
            let is_selected = match mode {
                EditMode::Object => selection.is_object_selected(),
                EditMode::Face => selection.is_face_selected(face_id),
                _ => false,
            };
            let color = if is_selected {
                if mode == EditMode::Object {
                    [0.32, 0.65, 0.40] // Maya lead green for selected object
                } else {
                    [0.88, 0.48, 0.18] // Maya warm orange for selected face
                }
            } else {
                [0.6, 0.6, 0.6] // Default surface grey
            };

            GpuVertex::new_with_normal(render_vert.position, render_vert.normal, color)
        })
        .collect();

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(&wgpu_vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Index Buffer"),
        contents: bytemuck::cast_slice(&raw_indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    (vertex_buffer, index_buffer, raw_indices.len() as u32)
}

/// Builds an unlit `LineList` render pipeline (used for wireframe overlay and transform gizmo).
fn create_line_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
    label: &str,
    depth_stencil: Option<wgpu::DepthStencilState>,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: "vs_overlay",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[GpuVertex::desc()],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: "fs_overlay",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil,
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    })
}

/// Builds GPU line buffer for wireframe edges and triangle buffer for square vertex dots.
fn create_overlay_buffers(
    device: &wgpu::Device,
    scene: &Scene,
) -> (wgpu::Buffer, u32, wgpu::Buffer, u32) {
    let buffers = overlay::build_overlay_buffers(scene.mesh(), scene.mode(), scene.selection());

    let line_slice: &[GpuVertex] = if buffers.lines.is_empty() {
        &[GpuVertex::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])]
    } else {
        &buffers.lines
    };
    let line_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Overlay Line Buffer"),
        contents: bytemuck::cast_slice(line_slice),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let dot_slice: &[GpuVertex] = if buffers.dots.is_empty() {
        &[GpuVertex::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])]
    } else {
        &buffers.dots
    };
    let dot_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Overlay Dot Buffer"),
        contents: bytemuck::cast_slice(dot_slice),
        usage: wgpu::BufferUsages::VERTEX,
    });

    (
        line_buffer,
        buffers.lines.len() as u32,
        dot_buffer,
        buffers.dots.len() as u32,
    )
}

struct DragState {
    axis: GizmoAxis,
    pivot: cgmath::Point3<f32>,
    initial_param: f32,
    initial_positions: Vec<(VertexId, cgmath::Point3<f32>)>,
}

pub struct State<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Arc<Window>,
    render_pipeline: wgpu::RenderPipeline,
    overlay_line_pipeline: wgpu::RenderPipeline,
    overlay_dot_pipeline: wgpu::RenderPipeline,
    gizmo_pipeline: wgpu::RenderPipeline,
    scene: Scene,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    overlay_line_buffer: wgpu::Buffer,
    num_line_vertices: u32,
    overlay_dot_buffer: wgpu::Buffer,
    num_dot_vertices: u32,
    gizmo_buffer: wgpu::Buffer,
    num_gizmo_vertices: u32,
    depth_texture_view: wgpu::TextureView,
    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    ui_state: UiState,
    cursor_pos: winit::dpi::PhysicalPosition<f64>,
    shift_pressed: bool,
    ctrl_pressed: bool,
    drag_state: Option<DragState>,
    hovered_axis: Option<GizmoAxis>,
    /// One-line outcome of the last command, shown in the Tools panel. Maya's command echo is what tells
    /// a user a hotkey was received; without it a refused extrude is indistinguishable from a dead key.
    notice: Option<String>,
}

/// The name a gizmo drag records into history, Maya-style: the tool plus the component being moved
/// ("Move Vertex", "Move Edge"). In Object mode Maya names the chunk after the tool alone.
fn drag_action_label(mode: EditMode, tool: Tool) -> String {
    if mode == EditMode::Object {
        tool.label().to_owned()
    } else {
        format!("{} {}", tool.label(), mode.label())
    }
}

/// A keyboard action the viewport responds to. Produced by [`hotkey_for`], carried out by
/// `State::handle_hotkey`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Hotkey {
    SetTool(Tool),
    CancelDrag,
    Undo,
    Redo,
    Extrude,
}

/// The Maya keymap, as a pure function of the physical key and the Ctrl / Shift modifiers.
///
/// A free function rather than inline event-loop matching so the bindings - including the modifier
/// rules below - are unit-testable without a window or a GPU context.
///
/// The tool keys only fire with Ctrl released: `Ctrl+Z` sends `KeyZ`, and without that guard a
/// `Ctrl+E` intended as a menu mnemonic would silently switch to Rotate. They also require Shift
/// released, which is what frees `Shift+E` for extrude - Maya binds extrude to a shelf button rather
/// than a key, so this follows the `Shift` + letter convention the rest of the map already uses.
fn hotkey_for(key: winit::keyboard::KeyCode, ctrl: bool, shift: bool) -> Option<Hotkey> {
    use winit::keyboard::KeyCode as Key;

    if ctrl {
        return match key {
            Key::KeyZ if shift => Some(Hotkey::Redo),
            Key::KeyZ => Some(Hotkey::Undo),
            Key::KeyY => Some(Hotkey::Redo),
            _ => None,
        };
    }

    if shift {
        return match key {
            Key::KeyE => Some(Hotkey::Extrude),
            Key::Escape => Some(Hotkey::CancelDrag),
            _ => None,
        };
    }

    match key {
        Key::KeyQ => Some(Hotkey::SetTool(Tool::Select)),
        Key::KeyW => Some(Hotkey::SetTool(Tool::Translate)),
        Key::KeyE => Some(Hotkey::SetTool(Tool::Rotate)),
        Key::KeyR => Some(Hotkey::SetTool(Tool::Scale)),
        Key::Escape => Some(Hotkey::CancelDrag),
        _ => None,
    }
}

impl<'a> State<'a> {
    pub async fn new(window: Arc<Window>) -> State<'a> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .unwrap();
        let config = surface
            .get_default_config(&adapter, size.width, size.height)
            .unwrap();
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let camera = Camera {
            target: cgmath::Point3::new(0.0, 0.0, 0.0),
            distance: 3.0,
            yaw: 0.785,
            pitch: 0.5,
        };
        let camera_controller = CameraController::new();

        let aspect_ratio = config.width as f32 / config.height as f32;
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(
            &camera,
            aspect_ratio,
            config.width as f32,
            config.height as f32,
        );

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            label: Some("Depth Texture"),
            view_formats: &[],
        });
        let depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[GpuVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let overlay_line_pipeline = create_line_pipeline(
            &device,
            &shader,
            &render_pipeline_layout,
            config.format,
            "Overlay Line Pipeline",
            Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: -2,
                    slope_scale: -2.0,
                    clamp: 0.0,
                },
            }),
        );

        let gizmo_pipeline = create_line_pipeline(
            &device,
            &shader,
            &render_pipeline_layout,
            config.format,
            "Gizmo Line Pipeline",
            Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
        );

        let overlay_dot_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Overlay Dot Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_dot",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[GpuVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_overlay",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: -3,
                    slope_scale: -2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let scene = Scene::new();
        let (vertex_buffer, index_buffer, num_indices) = create_mesh_buffers(&device, &scene);
        let (overlay_line_buffer, num_line_vertices, overlay_dot_buffer, num_dot_vertices) =
            create_overlay_buffers(&device, &scene);

        let gizmo_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gizmo Line Buffer"),
            size: (GIZMO_VERTEX_COUNT * std::mem::size_of::<GpuVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::viewport::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, config.format, None, 1);
        let ui_state = UiState::new();

        Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            render_pipeline,
            overlay_line_pipeline,
            overlay_dot_pipeline,
            gizmo_pipeline,
            scene,
            vertex_buffer,
            index_buffer,
            num_indices,
            overlay_line_buffer,
            num_line_vertices,
            overlay_dot_buffer,
            num_dot_vertices,
            gizmo_buffer,
            num_gizmo_vertices: 0,
            depth_texture_view,
            camera,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            egui_ctx,
            egui_state,
            egui_renderer,
            ui_state,
            cursor_pos: winit::dpi::PhysicalPosition::new(0.0, 0.0),
            shift_pressed: false,
            ctrl_pressed: false,
            drag_state: None,
            hovered_axis: None,
            notice: None,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn size(&self) -> winit::dpi::PhysicalSize<u32> {
        self.size
    }

    pub fn camera_controller_mut(&mut self) -> &mut CameraController {
        &mut self.camera_controller
    }

    pub fn wants_keyboard_input(&self) -> bool {
        self.egui_ctx.wants_keyboard_input()
    }

    pub fn on_window_event(
        &mut self,
        event: &winit::event::WindowEvent,
    ) -> egui_winit::EventResponse {
        self.egui_state.on_window_event(&self.window, event)
    }

    pub fn set_shift_pressed(&mut self, pressed: bool) {
        self.shift_pressed = pressed;
    }

    pub fn set_ctrl_pressed(&mut self, pressed: bool) {
        self.ctrl_pressed = pressed;
    }

    /// Applies the Maya hotkey bound to `key`, returning true when the key was handled.
    ///
    /// The keymap itself is the pure [`hotkey_for`]; this only carries out what it asks for, so the
    /// bindings stay unit-testable without a window or a GPU.
    pub fn handle_hotkey(&mut self, key: winit::keyboard::KeyCode) -> bool {
        match hotkey_for(key, self.ctrl_pressed, self.shift_pressed) {
            Some(Hotkey::SetTool(tool)) => {
                self.scene.set_tool(tool);
                true
            }
            Some(Hotkey::CancelDrag) => {
                self.cancel_active_drag();
                true
            }
            Some(Hotkey::Undo) => {
                self.undo();
                true
            }
            Some(Hotkey::Redo) => {
                self.redo();
                true
            }
            Some(Hotkey::Extrude) => {
                self.extrude_selected_faces();
                true
            }
            None => false,
        }
    }

    /// Pushes the selected faces out by the fixed step, reporting the outcome in the Tools panel.
    ///
    /// A seam rather than a `main.rs` call into `Scene` for the same reason `undo` is: a drag in flight
    /// holds vertex ids and initial positions from the state an extrude invalidates, so it has to be
    /// abandoned first. A refusal leaves the mesh alone and is shown instead, because a hotkey that
    /// silently did nothing reads as a dead key.
    pub fn extrude_selected_faces(&mut self) {
        self.cancel_active_drag();
        self.notice = Some(
            match self.scene.extrude_selected_faces(EXTRUDE_FACE_DISTANCE) {
                Ok(shape) => format!(
                    "Extruded {} {} into {} wall{}",
                    shape.source_faces.len(),
                    if shape.source_faces.len() == 1 {
                        "face"
                    } else {
                        "faces"
                    },
                    shape.side_faces.len(),
                    if shape.side_faces.len() == 1 { "" } else { "s" }
                ),
                // The reasons already read as sentences naming the components involved, so they go up
                // as given rather than being paraphrased into something less specific.
                Err(errors) => errors.join(" "),
            },
        );
    }

    /// Steps the document one action back. A seam rather than a `main.rs` call into `Scene`, because
    /// an in-flight drag has to be abandoned first: undoing underneath an active drag would leave the
    /// drag holding vertex ids and initial positions from the state that was just rolled back.
    pub fn undo(&mut self) {
        self.cancel_active_drag();
        self.scene.undo();
    }

    /// Re-applies the last undone action, abandoning an in-flight drag for the same reason.
    pub fn redo(&mut self) {
        self.cancel_active_drag();
        self.scene.redo();
    }

    pub fn set_alt_pressed(&mut self, pressed: bool) {
        self.camera_controller.alt_pressed = pressed;
    }

    pub fn process_mouse_motion(&mut self, dx: f64, dy: f64) {
        if !self.egui_ctx.wants_pointer_input() {
            self.camera_controller
                .process_mouse_motion(dx, dy, &mut self.camera);
            self.update_camera_matrix();
        }
    }

    pub fn process_scroll(&mut self, delta: &winit::event::MouseScrollDelta) {
        self.camera_controller
            .process_scroll(delta, &mut self.camera);
        self.update_camera_matrix();
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                size: wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                label: Some("Depth Texture"),
                view_formats: &[],
            });
            self.depth_texture_view =
                depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

            self.update_camera_matrix();
        }
    }

    pub fn update_camera_matrix(&mut self) {
        let width = self.config.width as f32;
        let height = self.config.height as f32;
        let aspect = width / height;
        self.camera_uniform
            .update_view_proj(&self.camera, aspect, width, height);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }

    fn upload_mesh(&mut self) {
        let (vertex_buffer, index_buffer, num_indices) =
            create_mesh_buffers(&self.device, &self.scene);
        let (overlay_line_buffer, num_line_vertices, overlay_dot_buffer, num_dot_vertices) =
            create_overlay_buffers(&self.device, &self.scene);
        self.vertex_buffer = vertex_buffer;
        self.index_buffer = index_buffer;
        self.num_indices = num_indices;
        self.overlay_line_buffer = overlay_line_buffer;
        self.num_line_vertices = num_line_vertices;
        self.overlay_dot_buffer = overlay_dot_buffer;
        self.num_dot_vertices = num_dot_vertices;
    }

    fn upload_gizmo(&mut self) {
        if self.scene.tool() != Tool::Translate {
            self.num_gizmo_vertices = 0;
            return;
        }

        let active_axis = self
            .drag_state
            .as_ref()
            .map(|d| d.axis)
            .or(self.hovered_axis);

        let arms = gizmo::translate_arms(
            self.scene.mesh(),
            self.scene.mode(),
            self.scene.selection(),
            self.camera.eye(),
            self.config.height as f32,
            active_axis,
        );
        let vertices = overlay::gizmo_line_vertices(&arms);
        self.num_gizmo_vertices = vertices.len() as u32;
        if self.num_gizmo_vertices > 0 {
            self.queue
                .write_buffer(&self.gizmo_buffer, 0, bytemuck::cast_slice(&vertices));
        }
    }

    pub fn cursor_ray(&self) -> Option<Ray> {
        Ray::from_screen(
            &self.camera,
            self.cursor_pos.x as f32,
            self.cursor_pos.y as f32,
            self.config.width as f32,
            self.config.height as f32,
        )
    }

    pub fn handle_cursor_moved(&mut self, position: winit::dpi::PhysicalPosition<f64>) {
        self.cursor_pos = position;
        let height = self.config.height as f32;

        if let Some(ray) = self.cursor_ray() {
            if let Some(ref drag) = self.drag_state {
                let axis_dir = drag.axis.unit_vector();
                if let Some(current_param) =
                    gizmo::project_ray_onto_axis(drag.pivot, axis_dir, &ray)
                {
                    let delta_t = current_param - drag.initial_param;
                    let delta_vec = axis_dir * delta_t;
                    self.scene
                        .translate_vertices(&drag.initial_positions, delta_vec);
                }
            } else if self.scene.tool() == Tool::Translate
                && !self.camera_controller.alt_pressed
                && !self.ui_state.pie_menu_active
                && !self.egui_ctx.wants_pointer_input()
            {
                if let Some(pivot) =
                    gizmo::pivot(self.scene.mesh(), self.scene.mode(), self.scene.selection())
                {
                    self.hovered_axis = gizmo::pick_gizmo_axis(
                        pivot,
                        self.camera.eye(),
                        height,
                        &ray,
                        gizmo::GIZMO_PICK_TOLERANCE_PIXELS,
                    );
                } else {
                    self.hovered_axis = None;
                }
            } else {
                self.hovered_axis = None;
            }
        }
    }

    pub fn handle_mouse_input(
        &mut self,
        button_state: winit::event::ElementState,
        button: winit::event::MouseButton,
    ) {
        if button != winit::event::MouseButton::Left {
            return;
        }

        if button_state == winit::event::ElementState::Pressed {
            if !self.camera_controller.alt_pressed
                && !self.ui_state.pie_menu_active
                && !self.egui_ctx.wants_pointer_input()
            {
                let height = self.config.height as f32;
                if let Some(ray) = self.cursor_ray() {
                    let hit_axis = if self.scene.tool() == Tool::Translate {
                        gizmo::pivot(self.scene.mesh(), self.scene.mode(), self.scene.selection())
                            .and_then(|pivot| {
                                gizmo::pick_gizmo_axis(
                                    pivot,
                                    self.camera.eye(),
                                    height,
                                    &ray,
                                    gizmo::GIZMO_PICK_TOLERANCE_PIXELS,
                                )
                            })
                    } else {
                        None
                    };

                    if let Some(axis) = hit_axis {
                        let pivot = gizmo::pivot(
                            self.scene.mesh(),
                            self.scene.mode(),
                            self.scene.selection(),
                        )
                        .unwrap();
                        let initial_param =
                            gizmo::project_ray_onto_axis(pivot, axis.unit_vector(), &ray)
                                .unwrap_or(0.0);
                        let initial_positions: Vec<(VertexId, cgmath::Point3<f32>)> = self
                            .scene
                            .selected_vertices()
                            .into_iter()
                            .map(|v| (v, self.scene.mesh().vertex_position(v)))
                            .collect();

                        self.drag_state = Some(DragState {
                            axis,
                            pivot,
                            initial_param,
                            initial_positions,
                        });
                        // Opens the history step for this drag. Nothing has been moved yet, so this
                        // snapshot is the state `undo` returns to. `end_action` on release records it,
                        // which is what makes a whole drag one undo step instead of one per frame.
                        let label = drag_action_label(self.scene.mode(), self.scene.tool());
                        self.scene.begin_action(&label);
                    } else {
                        self.scene.select_at(&ray, self.shift_pressed, height);
                    }
                }
            }
        } else if button_state == winit::event::ElementState::Released
            && self.drag_state.take().is_some()
        {
            // Releases the translation, then records it. A press-and-release with no movement
            // compares equal to the captured state, so it leaves no step in history.
            self.scene.end_action();
        }
    }

    pub fn cancel_active_drag(&mut self) {
        if let Some(drag) = self.drag_state.take() {
            self.scene
                .translate_vertices(&drag.initial_positions, cgmath::Vector3::new(0.0, 0.0, 0.0));
            self.scene.cancel_action();
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // 1. Prepare UI data
        let raw_input = self.egui_state.take_egui_input(&self.window);
        self.egui_ctx.begin_frame(raw_input);

        let ui_response = self.ui_state.draw(
            &self.egui_ctx,
            UiContext {
                mode: self.scene.mode(),
                tool: self.scene.tool(),
                selected_count: self.scene.selected_count(),
                can_undo: self.scene.can_undo(),
                can_redo: self.scene.can_redo(),
                undo_label: self.scene.undo_label(),
                redo_label: self.scene.redo_label(),
                can_extrude: self.scene.can_extrude(),
                notice: self.notice.as_deref(),
            },
        );
        if let Some(mode) = ui_response.mode {
            self.scene.set_mode(mode);
        }
        if let Some(tool) = ui_response.tool {
            self.scene.set_tool(tool);
        }
        if ui_response.undo {
            self.undo();
        }
        if ui_response.redo {
            self.redo();
        }
        if ui_response.extrude {
            self.extrude_selected_faces();
        }

        if self.scene.take_dirty() {
            self.upload_mesh();
        }

        self.upload_gizmo();

        let full_output = self.egui_ctx.end_frame();
        let paint_jobs = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);

        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, image_delta);
        }

        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &paint_jobs,
            &egui_wgpu::ScreenDescriptor {
                size_in_pixels: [self.config.width, self.config.height],
                pixels_per_point: self.window.scale_factor() as f32,
            },
        );

        // 2. Render 3D Scene
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("3D Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.196,
                            g: 0.196,
                            b: 0.196,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);

            if self.num_line_vertices > 0 {
                render_pass.set_pipeline(&self.overlay_line_pipeline);
                render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.overlay_line_buffer.slice(..));
                render_pass.draw(0..self.num_line_vertices, 0..1);
            }

            if self.num_dot_vertices > 0 {
                render_pass.set_pipeline(&self.overlay_dot_pipeline);
                render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.overlay_dot_buffer.slice(..));
                render_pass.draw(0..self.num_dot_vertices, 0..1);
            }

            if self.num_gizmo_vertices > 0 {
                render_pass.set_pipeline(&self.gizmo_pipeline);
                render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.gizmo_buffer.slice(..));
                render_pass.draw(0..self.num_gizmo_vertices, 0..1);
            }
        }

        // 3. Render UI pass on top
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("UI Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            self.egui_renderer.render(
                &mut render_pass,
                &paint_jobs,
                &egui_wgpu::ScreenDescriptor {
                    size_in_pixels: [self.config.width, self.config.height],
                    pixels_per_point: self.window.scale_factor() as f32,
                },
            );
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::keyboard::KeyCode as Key;

    #[test]
    fn plain_keys_switch_tools_as_in_maya() {
        assert_eq!(
            hotkey_for(Key::KeyQ, false, false),
            Some(Hotkey::SetTool(Tool::Select))
        );
        assert_eq!(
            hotkey_for(Key::KeyW, false, false),
            Some(Hotkey::SetTool(Tool::Translate))
        );
        assert_eq!(
            hotkey_for(Key::KeyE, false, false),
            Some(Hotkey::SetTool(Tool::Rotate))
        );
        assert_eq!(
            hotkey_for(Key::KeyR, false, false),
            Some(Hotkey::SetTool(Tool::Scale))
        );
    }

    #[test]
    fn escape_cancels_the_active_drag() {
        assert_eq!(
            hotkey_for(Key::Escape, false, false),
            Some(Hotkey::CancelDrag)
        );
    }

    #[test]
    fn ctrl_z_undoes_and_the_redo_chords_redo() {
        assert_eq!(hotkey_for(Key::KeyZ, true, false), Some(Hotkey::Undo));
        assert_eq!(hotkey_for(Key::KeyZ, true, true), Some(Hotkey::Redo));
        assert_eq!(hotkey_for(Key::KeyY, true, false), Some(Hotkey::Redo));
    }

    #[test]
    fn tool_keys_do_not_fire_while_ctrl_is_held() {
        // Otherwise `Ctrl+E` would quietly switch to Rotate, and `Ctrl+W` would close nothing but
        // change the tool.
        assert_eq!(hotkey_for(Key::KeyE, true, false), None);
        assert_eq!(hotkey_for(Key::KeyW, true, false), None);
        assert_eq!(hotkey_for(Key::KeyQ, true, false), None);
    }

    #[test]
    fn unbound_keys_are_left_alone() {
        assert_eq!(
            hotkey_for(Key::KeyZ, false, false),
            None,
            "Z alone is not an undo key"
        );
        assert_eq!(hotkey_for(Key::KeyY, false, false), None);
        assert_eq!(
            hotkey_for(Key::Space, false, false),
            None,
            "Space belongs to the pie menu"
        );
    }

    #[test]
    fn shift_alone_does_not_reach_the_history() {
        // Shift+click is additive selection, handled by picking, and Shift+Z must not undo.
        assert_eq!(hotkey_for(Key::KeyZ, false, true), None);
        assert_eq!(
            hotkey_for(Key::Escape, false, true),
            Some(Hotkey::CancelDrag)
        );
    }

    #[test]
    fn drag_labels_name_the_component_being_moved() {
        assert_eq!(
            drag_action_label(EditMode::Vertex, Tool::Translate),
            "Move Vertex"
        );
        assert_eq!(
            drag_action_label(EditMode::Edge, Tool::Translate),
            "Move Edge"
        );
        assert_eq!(
            drag_action_label(EditMode::Face, Tool::Translate),
            "Move Face"
        );
        assert_eq!(drag_action_label(EditMode::Object, Tool::Translate), "Move");
    }
}
