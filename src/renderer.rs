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
use crate::scene::{ExtrudeCommit, Scene};
use crate::tool::Tool;
use crate::ui::{UiContext, UiResponse, UiState};
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

/// Cursor bookkeeping a drag shares, so moving the view mid-gesture does not fling the geometry.
///
/// Projecting a cursor ray onto an axis says something about position only while the eye stays put, and
/// navigation does not end a tool drag: Alt+drag orbits, and the wheel zooms with no modifier at all.
/// Measured from a new sight line, the next projection reports a parameter that has nothing to do with
/// where the user's hand moved, so the lump lurches by however far the camera travelled.
///
/// `needs_reanchor` absorbs that. A camera swing marks the anchor, and the next event re-baselines
/// instead of moving anything - the drag continues from the distance it had already reached. Until a swing
/// happens the arithmetic is exactly the cursor's travel along the axis, so ordinary drags are unaffected.
struct DragAnchor {
    /// Where the cursor projected onto the drag axis at the previous event.
    last_param: f32,
    /// Total travel along the axis since the drag began, which is what the geometry is offset by.
    applied: f32,
    /// Set when a camera change swallowed events, so the next projection re-baselines.
    needs_reanchor: bool,
}

impl DragAnchor {
    fn new(first_param: f32) -> Self {
        Self {
            last_param: first_param,
            applied: 0.0,
            needs_reanchor: false,
        }
    }

    /// Starts an anchor from whatever projection the cursor offers when the drag is armed, if any.
    ///
    /// A cursor outside the window, or a sight line running down the axis, projects to nothing. Anchoring
    /// on an arbitrary zero in that case would read the first mouse move as travel, so the anchor is
    /// flagged to re-baseline on the next event instead.
    fn at(first_param: Option<f32>) -> Self {
        let mut anchor = Self::new(first_param.unwrap_or(0.0));
        anchor.needs_reanchor = first_param.is_none();
        anchor
    }

    /// Folds in one cursor event. The drag's geometry is then offset by [`DragAnchor::applied`] along its
    /// axis, relative to what was captured when the drag began.
    ///
    /// `limit` caps a single step as a distance in model units. Extrude passes one because a sight line
    /// running nearly parallel to the travel axis turns a one-pixel move into a wild projection; translate
    /// has no such problem and passes [`f32::MAX`].
    fn advance(&mut self, param: f32, limit: f32) {
        let travel = if self.needs_reanchor {
            self.needs_reanchor = false;
            0.0
        } else {
            param - self.last_param
        };
        self.last_param = param;
        self.applied += travel.clamp(-limit, limit);
    }

    /// Notes that the view moved, so the next projection cannot be compared with `last_param`.
    fn mark_swung(&mut self) {
        self.needs_reanchor = true;
    }
}

struct DragState {
    axis: GizmoAxis,
    /// The orientation this drag was grabbed in, kept for its whole life rather than re-read each event.
    ///
    /// Re-deriving it from the selection would make the drag follow the *current* space while the arms the
    /// user is pulling were drawn in another, and would also let a selection change mid-drag swing the
    /// travel axis under the cursor. Both draw and pick read the same value back out of here.
    frame: gizmo::Frame,
    anchor: DragAnchor,
    initial_positions: Vec<(VertexId, cgmath::Point3<f32>)>,
}

/// An extrude awaiting its depth from the mouse, armed by [`Hotkey::Extrude`] and settled by a click.
///
/// The `frame` is [`gizmo::Frame::from_axis`] about the direction the faces will travel, so the handle
/// the user drags is oriented to the geometry rather than to the world - the same arrangement Maya calls
/// component space, and the reason this holds a frame where [`DragState`] holds a [`GizmoAxis`].
struct ExtrudeState {
    frame: gizmo::Frame,
    /// World length of the drawn arm, remembered so the step limit scales with the handle rather than
    /// with an absolute distance the user never chose.
    arm_length: f32,
    /// How far the faces have travelled along the frame's Z, accumulated by [`DragAnchor`].
    anchor: DragAnchor,
}

/// The most depth an extrude drag may gain or lose in one cursor event, as a multiple of the handle's
/// on-screen length.
///
/// Projecting the cursor onto an axis only says something while the sight line differs from that axis.
/// Extruding a face that points straight at the camera is the limit case: every cursor ray is nearly
/// parallel to the travel axis, so the closest-approach point - and with it the depth - swings wildly for
/// a one-pixel move. Without a limit the lump leaps across the scene; with one it moves at a pace the user
/// can follow, catching up over the next few events, which is what a handle at that angle can honestly
/// offer.
const MAX_EXTRUDE_STEP_ARMS: f32 = 1.0;

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
    extrude_state: Option<ExtrudeState>,
    hovered_axis: Option<GizmoAxis>,
    /// One-line outcome of the last command, shown in the Tools panel. Maya's command echo is what tells
    /// a user a hotkey was received; without it a refused extrude is indistinguishable from a dead key.
    notice: Option<String>,
}

/// The command echo for a settled extrude, worded so a drag that never left the surface reads as the
/// no-op it was rather than as a growth of zero walls.
fn describe_extrude(commit: ExtrudeCommit) -> String {
    if commit.walls == 0 {
        return "Extrude cancelled - the faces never left the surface".to_string();
    }
    format!(
        "Extruded {} {} into {} wall{}",
        commit.faces,
        if commit.faces == 1 { "face" } else { "faces" },
        commit.walls,
        if commit.walls == 1 { "" } else { "s" }
    )
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
    EdgeLoop,
    CycleSpace,
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
///
/// `Shift+L` carries the same reasoning for Maya's *Insert Edge Loop Tool*, which is toolbox-only there
/// as well: `L` for loop, sitting beside `Shift+E` so the two modelling commands stay together instead of
/// being scattered across the map by whichever letter happened to be free.
///
/// `X` is a deliberate divergence rather than a Maya keymap copy: bare `x` is Maya's press-and-release
/// "snap to grids", and this app has no snapping of any kind yet, so there is nothing here for it to
/// conflict with. It sits with the tool keys rather than outside them so the same modifier guard applies
/// (`Ctrl+X` is Cut everywhere, and must not reorient the handle). When snapping lands, it will need this
/// key back or a different home for the reference system - hence the note.
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
            Key::KeyL => Some(Hotkey::EdgeLoop),
            Key::Escape => Some(Hotkey::CancelDrag),
            _ => None,
        };
    }

    match key {
        Key::KeyQ => Some(Hotkey::SetTool(Tool::Select)),
        Key::KeyW => Some(Hotkey::SetTool(Tool::Translate)),
        Key::KeyE => Some(Hotkey::SetTool(Tool::Rotate)),
        Key::KeyR => Some(Hotkey::SetTool(Tool::Scale)),
        Key::KeyX => Some(Hotkey::CycleSpace),
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
            extrude_state: None,
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
                self.begin_extrude_drag();
                true
            }
            Some(Hotkey::EdgeLoop) => {
                self.insert_edge_loop();
                true
            }
            Some(Hotkey::CycleSpace) => {
                self.scene.cycle_space();
                // Echoed because the only other signal is the handle reorienting, which is easy to miss on
                // a face whose normal already runs along a world axis.
                self.notice = Some(format!("Gizmo space: {}", self.scene.space().label()));
                true
            }
            None => false,
        }
    }

    /// Arms an extrude of the selected faces and waits for the mouse to give it a depth.
    ///
    /// A seam rather than a `main.rs` call into `Scene` for the same reason `undo` is: a translate drag in
    /// flight holds vertex ids and initial positions from the state an extrude invalidates, so it has to be
    /// abandoned first. A refusal leaves the document untouched and is echoed instead, because a hotkey that
    /// silently did nothing reads as a dead key.
    ///
    /// The handle is drawn in the frame the faces define - [`gizmo::Frame::from_axis`] about the direction
    /// they will travel - so the user drags along the geometry's own axis, which is what Maya calls
    /// component space, rather than guessing which world axis points out of a tilted face.
    pub fn begin_extrude_drag(&mut self) {
        self.cancel_active_drag();

        let guide = match self.scene.begin_extrude_drag() {
            Ok(guide) => guide,
            Err(errors) => {
                self.notice = Some(errors.join(" "));
                return;
            }
        };

        let height = self.config.height as f32;
        let arm_length = gizmo::axis_world_length(guide.pivot, self.camera.eye(), height);
        let Some(frame) = gizmo::Frame::from_axis(guide.pivot, guide.direction) else {
            // Unreachable while `Scene` hands back a unit normal, but an axis with no direction would draw
            // a handle pointing nowhere, so the drag is abandoned rather than invented.
            self.scene.abandon_extrude();
            self.notice = Some("the selected faces have no direction to extrude along".to_string());
            return;
        };

        let travel = frame.direction(GizmoAxis::Z);
        let anchor = DragAnchor::at(
            self.cursor_ray()
                .and_then(|ray| gizmo::project_ray_onto_axis(frame.origin, travel, &ray)),
        );

        self.extrude_state = Some(ExtrudeState {
            frame,
            arm_length,
            anchor,
        });
        self.notice = Some(
            "Extruding - move the mouse to set the depth, click to set it, Escape to cancel"
                .to_string(),
        );
    }

    /// Runs an edge loop through the single selected edge and echoes what it cut.
    ///
    /// The gesture in flight is closed first for the reason `cancel_active_drag` documents: an armed extrude
    /// previews from the snapshot taken the instant it was armed, so cutting a ring underneath it and
    /// leaving that extrude open would have its click delete the ring the user just made. Unlike extrude
    /// there is no depth to drag here - a loop lands on the press - so this needs no follow-up state of its
    /// own, and a refusal arrives as a notice because a hotkey that silently did nothing reads as a dead key.
    pub fn insert_edge_loop(&mut self) {
        self.cancel_active_drag();

        match self.scene.insert_edge_loop() {
            Ok(cut) => {
                self.notice = Some(format!(
                    "Inserted an edge loop through {} faces, adding {} vertices",
                    cut.faces_cut, cut.new_vertices
                ));
            }
            Err(errors) => self.notice = Some(errors.join(" ")),
        }
    }

    /// Moves the cursor along the armed extrude's travel axis, re-previewing the lump where it now stands.
    fn track_extrude_drag(&mut self, ray: &Ray) {
        // `Scene` is the authority on whether a drag is open. A mode or tool switch arriving from the
        // panel closes one inside `Scene` without passing through this state, and previewing on against
        // a snapshot it has already rolled back would resurrect geometry the user has let go of.
        if !self.scene.is_extruding() {
            self.extrude_state = None;
            return;
        }

        let Some(extrude) = self.extrude_state.as_mut() else {
            return;
        };
        let travel = extrude.frame.direction(GizmoAxis::Z);

        // A cursor ray running parallel to the travel axis says nothing about depth, so the drag holds the
        // last depth it knew rather than inventing one.
        if let Some(param) = gizmo::project_ray_onto_axis(extrude.frame.origin, travel, ray) {
            let limit = extrude.arm_length * MAX_EXTRUDE_STEP_ARMS;
            extrude.anchor.advance(param, limit);
        }

        let depth = extrude.anchor.applied;
        let _ = self.scene.preview_extrude(depth);
    }

    /// Marks whichever drag is in flight that the view has moved, so its next cursor event re-baselines
    /// rather than reading a projection measured from a different sight line.
    fn mark_drag_swung(&mut self) {
        if let Some(extrude) = self.extrude_state.as_mut() {
            extrude.anchor.mark_swung();
        }
        if let Some(drag) = self.drag_state.as_mut() {
            drag.anchor.mark_swung();
        }
    }

    /// Settles an armed extrude wherever the mouse left it, echoing what landed. Returns false when no
    /// extrude was armed, so the caller knows the click was the user selecting rather than committing.
    fn commit_extrude_drag(&mut self) -> bool {
        if self.extrude_state.take().is_none() {
            return false;
        }
        // `Scene` may have closed the drag first - a mode or tool switch on the same frame - and there is
        // nothing to settle then. Reporting false lets the click be read as the selection it also is,
        // rather than vanishing into a commit of nothing.
        let Some(commit) = self.scene.commit_extrude() else {
            return false;
        };
        self.notice = Some(describe_extrude(commit));
        true
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
        // The wheel needs no modifier, so it is the one camera change a user can make without lifting a
        // drag or releasing a key - zooming to see a wall better mid-extrude would otherwise be read as
        // depth the moment the cursor moves again.
        self.mark_drag_swung();
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            // A different window height changes what the same cursor position projects to, so a drag in
            // flight has to re-baseline before its next event.
            self.mark_drag_swung();
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

    /// The frame the transform gizmo is drawn and picked in this frame, or `None` when there is nothing to
    /// anchor it to.
    ///
    /// One accessor feeding the draw path and both pick paths, because a handle drawn from one frame and
    /// clicked against another puts the click target on a different line than the one the user aimed at.
    /// That was invisible while every gizmo was world-aligned; with [`gizmo::Space::Component`] tilting the
    /// arms it becomes a handle that picks nothing.
    ///
    /// A drag in flight reports the frame it was grabbed with rather than the current one, so pressing the
    /// space toggle mid-pull cannot redraw the arms somewhere else while the geometry keeps travelling along
    /// the arm actually held.
    fn gizmo_frame(&self) -> Option<gizmo::Frame> {
        if let Some(drag) = self.drag_state.as_ref() {
            return Some(drag.frame);
        }
        gizmo::frame_for(
            self.scene.mesh(),
            self.scene.mode(),
            self.scene.selection(),
            self.scene.space(),
        )
    }

    /// Which arm of `frame` the cursor is over, or `None` for none of them.
    ///
    /// The arm length has to come from the frame's own depth rather than the selection's, so this computes
    /// it the way [`gizmo::frame_arms`] does; sharing one pick helper across hover and press is what keeps
    /// the two from disagreeing about where an arm ends.
    fn gizmo_axis_under(
        &self,
        frame: gizmo::Frame,
        ray: &Ray,
        viewport_height: f32,
    ) -> Option<GizmoAxis> {
        let length = gizmo::axis_world_length(frame.origin, self.camera.eye(), viewport_height);
        gizmo::pick_frame_axis(
            &frame,
            length,
            ray,
            gizmo::GIZMO_PICK_TOLERANCE_PIXELS,
            viewport_height,
        )
    }

    fn upload_gizmo(&mut self) {
        // An armed extrude owns the handle: the axis it will travel along is the depth control, so it
        // replaces the translate gizmo for as long as the drag is open, whatever tool is selected. Only
        // that one arm is drawn, because the drag has one degree of freedom - showing the two axes across
        // it would offer handles that do nothing.
        if let Some(extrude) = &self.extrude_state {
            let travel = extrude.frame.direction(GizmoAxis::Z);
            let arms = vec![gizmo::Arm {
                start: extrude.frame.origin,
                end: extrude.frame.origin + travel * extrude.arm_length,
                color: gizmo::COLOR_AXIS_ACTIVE,
            }];
            let vertices = overlay::gizmo_line_vertices(&arms);
            self.num_gizmo_vertices = vertices.len() as u32;
            if self.num_gizmo_vertices > 0 {
                self.queue
                    .write_buffer(&self.gizmo_buffer, 0, bytemuck::cast_slice(&vertices));
            }
            return;
        }

        if self.scene.tool() != Tool::Translate {
            self.num_gizmo_vertices = 0;
            return;
        }

        let active_axis = self
            .drag_state
            .as_ref()
            .map(|d| d.axis)
            .or(self.hovered_axis);

        let arms = match self.gizmo_frame() {
            Some(frame) => gizmo::frame_arms(
                &frame,
                self.camera.eye(),
                self.config.height as f32,
                active_axis,
            ),
            None => Vec::new(),
        };
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

        // Alt with a button down is the camera's gesture, and `process_mouse_motion` has already spent
        // this event orbiting. Every branch below reads the cursor as tool input instead, so orbiting for
        // a better look at the face being extruded would be banked as depth, and orbiting mid vertex-drag
        // would move geometry as a side effect of moving the view. Hover is cleared rather than left
        // standing, which is what an Alt-held cursor used to do.
        if self.camera_controller.is_navigating() {
            self.mark_drag_swung();
            self.hovered_axis = None;
            return;
        }

        if let Some(ray) = self.cursor_ray() {
            if self.extrude_state.is_some() {
                self.track_extrude_drag(&ray);
            } else if let Some(drag) = self.drag_state.as_mut() {
                let travel = drag.frame.direction(drag.axis);
                if let Some(current_param) =
                    gizmo::project_ray_onto_axis(drag.frame.origin, travel, &ray)
                {
                    drag.anchor.advance(current_param, f32::MAX);
                    let delta_vec = travel * drag.anchor.applied;
                    self.scene
                        .translate_vertices(&drag.initial_positions, delta_vec);
                }
            } else if self.scene.tool() == Tool::Translate
                && !self.camera_controller.alt_pressed
                && !self.ui_state.pie_menu_active
                && !self.egui_ctx.wants_pointer_input()
            {
                self.hovered_axis = self
                    .gizmo_frame()
                    .and_then(|frame| self.gizmo_axis_under(frame, &ray, height));
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
            // An armed extrude claims this click as its depth, so it has to be settled before the click
            // can be read as a selection or as the start of a gizmo drag.
            if self.commit_extrude_drag() {
                return;
            }

            if !self.camera_controller.alt_pressed
                && !self.ui_state.pie_menu_active
                && !self.egui_ctx.wants_pointer_input()
            {
                let height = self.config.height as f32;
                if let Some(ray) = self.cursor_ray() {
                    let frame = if self.scene.tool() == Tool::Translate {
                        self.gizmo_frame()
                    } else {
                        None
                    };
                    let hit_axis =
                        frame.and_then(|frame| self.gizmo_axis_under(frame, &ray, height));

                    if let Some((frame, axis)) = frame.zip(hit_axis) {
                        let anchor = DragAnchor::at(gizmo::project_ray_onto_axis(
                            frame.origin,
                            frame.direction(axis),
                            &ray,
                        ));
                        let initial_positions: Vec<(VertexId, cgmath::Point3<f32>)> = self
                            .scene
                            .selected_vertices()
                            .into_iter()
                            .map(|v| (v, self.scene.mesh().vertex_position(v)))
                            .collect();

                        self.drag_state = Some(DragState {
                            axis,
                            frame,
                            anchor,
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

    /// Gives up whatever gesture is in flight: an extrude awaiting its depth, then a translate drag.
    ///
    /// Both live here rather than at the call sites because every path that moves the document underneath
    /// a gesture has to close it first - Escape, yes, but also undo, redo and arming an extrude while a
    /// translate drag is open. Leaving an extrude armed across one of those would have it preview from a
    /// snapshot that no longer matches, and its click would land on unrelated geometry.
    pub fn cancel_active_drag(&mut self) {
        if self.extrude_state.take().is_some() {
            self.scene.abandon_extrude();
        }
        if let Some(drag) = self.drag_state.take() {
            self.scene
                .translate_vertices(&drag.initial_positions, cgmath::Vector3::new(0.0, 0.0, 0.0));
            self.scene.cancel_action();
        }
    }

    /// Carries out what the panel asked for this frame.
    ///
    /// A seam of its own because `ui.rs` reports intent only: nothing in the panel can reach `Scene`, which
    /// is what keeps the history rules - abandoning an armed extrude before switching mode or tool - in one
    /// place whether the command arrived from a button, the pie menu or a hotkey.
    fn apply_ui_response(&mut self, response: UiResponse) {
        if let Some(mode) = response.mode {
            self.scene.set_mode(mode);
        }
        if let Some(tool) = response.tool {
            self.scene.set_tool(tool);
        }
        if let Some(space) = response.space {
            self.scene.set_space(space);
        }
        if response.undo {
            self.undo();
        }
        if response.redo {
            self.redo();
        }
        if response.extrude {
            // The panel button means two different things depending on whether a drag is open, which is
            // what its label says: arm a new extrude, or settle the one already travelling at the depth
            // the mouse reached. Routing both through here keeps the viewport click and the button in
            // agreement rather than each holding their own idea of the gesture's second half.
            if self.scene.is_extruding() {
                self.commit_extrude_drag();
            } else {
                self.begin_extrude_drag();
            }
        }
        if response.edge_loop {
            self.insert_edge_loop();
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
                extruding: self.scene.is_extruding(),
                can_insert_edge_loop: self.scene.can_insert_edge_loop(),
                space: self.scene.space(),
                notice: self.notice.as_deref(),
            },
        );
        self.apply_ui_response(ui_response);

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
    fn x_cycles_the_gizmo_reference_system() {
        assert_eq!(
            hotkey_for(Key::KeyX, false, false),
            Some(Hotkey::CycleSpace)
        );
        // Held with a modifier it must stay quiet, or a `Ctrl+X` cut would silently reorient the handle
        // instead of doing nothing at all.
        assert_eq!(hotkey_for(Key::KeyX, true, false), None);
        assert_eq!(hotkey_for(Key::KeyX, false, true), None);
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

    #[test]
    fn an_undisturbed_drag_tracks_the_cursor_exactly() {
        // The re-anchoring must not cost anything ordinary: with the camera still, travel along the axis is
        // simply the cursor's movement, so the handle sits where the hand put it. The parameters below are
        // all exactly representable in f32, which is what lets these assert equality rather than a range.
        let mut anchor = DragAnchor::at(Some(10.5));
        anchor.advance(11.0, f32::MAX);
        anchor.advance(12.0, f32::MAX);
        anchor.advance(10.0, f32::MAX);
        assert_eq!(anchor.applied, -0.5);
    }

    #[test]
    fn swinging_the_camera_mid_drag_absorbs_the_jump() {
        // Alt+drag orbits without ending the tool drag, and the wheel zooms with no modifier at all. The
        // projection measured from the new sight line has nothing to do with the user's hand, so without a
        // re-baseline the next cursor event moves the geometry by however far the camera travelled - the
        // lump leaps, and on an extrude that depth is committed to the mesh.
        let mut anchor = DragAnchor::at(Some(10.5));
        anchor.advance(10.75, f32::MAX);
        anchor.mark_swung();
        anchor.advance(80.0, f32::MAX);
        assert_eq!(
            anchor.applied, 0.25,
            "the swing itself must contribute nothing"
        );

        anchor.advance(80.25, f32::MAX);
        assert_eq!(
            anchor.applied, 0.5,
            "and the drag resumes from the depth it had already reached"
        );
    }

    #[test]
    fn a_drag_armed_without_a_projection_waits_for_the_cursor() {
        // A cursor outside the window, or a sight line running down the travel axis, projects to nothing.
        // Anchoring that on zero would read the first move as travel and jump the moment the mouse stirred.
        let mut anchor = DragAnchor::at(None);
        anchor.advance(37.0, f32::MAX);
        assert_eq!(anchor.applied, 0.0);

        anchor.advance(37.5, f32::MAX);
        assert_eq!(anchor.applied, 0.5);
    }

    #[test]
    fn a_single_step_is_limited_but_the_drag_catches_up() {
        // Nearly parallel to the travel axis, a one-pixel move projects wildly. The limit is what keeps the
        // lump moving at a pace the user can follow; it is per event rather than total, so several steady
        // events cover the distance.
        let mut anchor = DragAnchor::at(Some(0.0));
        anchor.advance(100.0, 1.0);
        assert_eq!(anchor.applied, 1.0, "one event cannot outrun the handle");

        anchor.advance(101.0, 1.0);
        assert_eq!(anchor.applied, 2.0, "steady tracking keeps gaining");

        anchor.advance(-50.0, 1.0);
        assert_eq!(
            anchor.applied, 1.0,
            "and the same limit holds on the way back"
        );
    }
}
