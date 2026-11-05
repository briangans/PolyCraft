use cgmath::SquareMatrix;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    event::{DeviceEvent, ElementState, Event, MouseButton, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

const VERTICES: &[Vertex] = &[
    Vertex { position: [0.0, 0.5, 0.0], color: [1.0, 0.0, 0.0] },
    Vertex { position: [-0.5, -0.5, 0.0], color: [0.0, 1.0, 0.0] },
    Vertex { position: [0.5, -0.5, 0.0], color: [0.0, 0.0, 1.0] },
];

// --- NEW CAMERA SYSTEM ---

struct Camera {
    target: cgmath::Point3<f32>,
    distance: f32,
    yaw: f32,
    pitch: f32,
}

impl Camera {
    fn build_view_projection_matrix(&self, aspect: f32) -> cgmath::Matrix4<f32> {
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();

        // Calculate the camera's physical position in space based on orbit angles
        let eye = cgmath::Point3::new(
            self.target.x + self.distance * cos_pitch * sin_yaw,
            self.target.y + self.distance * sin_pitch,
            self.target.z + self.distance * cos_pitch * cos_yaw,
        );

        let view = cgmath::Matrix4::look_at_rh(eye, self.target, cgmath::Vector3::unit_y());
        let proj = cgmath::perspective(cgmath::Deg(45.0), aspect, 0.1, 100.0);
        
        let wgpu_correction = cgmath::Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.0,
            0.0, 0.0, 0.5, 1.0,
        );
        
        wgpu_correction * proj * view
    }
}

struct CameraController {
    is_drag_active: bool,
    speed: f32,
}

impl CameraController {
    fn new(speed: f32) -> Self {
        Self {
            is_drag_active: false,
            speed,
        }
    }

    fn process_mouse_input(&mut self, state: ElementState, button: MouseButton) -> bool {
        if button == MouseButton::Left {
            self.is_drag_active = state == ElementState::Pressed;
            true
        } else {
            false
        }
    }

    fn process_mouse_motion(&mut self, dx: f64, dy: f64, camera: &mut Camera) {
        if self.is_drag_active {
            camera.yaw -= (dx as f32) * self.speed;
            camera.pitch += (dy as f32) * self.speed;

            // Clamp pitch to prevent the camera from flipping upside down at the poles
            let safe_angle = std::f32::consts::FRAC_PI_2 - 0.01;
            camera.pitch = camera.pitch.clamp(-safe_angle, safe_angle);
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self { view_proj: cgmath::Matrix4::identity().into() }
    }
    fn update_view_proj(&mut self, camera: &Camera, aspect: f32) {
        self.view_proj = camera.build_view_projection_matrix(aspect).into();
    }
}

// --- STATE INTEGRATION ---

struct State<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    window: Arc<Window>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    
    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
}

impl<'a> State<'a> {
    async fn new(window: Arc<Window>) -> State<'a> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.unwrap();

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.unwrap();
        let config = surface.get_default_config(&adapter, size.width, size.height).unwrap();
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // Initialize Camera
        let camera = Camera {
            target: cgmath::Point3::new(0.0, 0.0, 0.0),
            distance: 2.0,
            yaw: 0.0,
            pitch: 0.2, // Start slightly above the triangle
        };
        let camera_controller = CameraController::new(0.005);
        
        let aspect_ratio = config.width as f32 / config.height as f32;
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, aspect_ratio);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() }],
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
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
                buffers: &[Vertex::desc()],
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            surface, device, queue, config, size, window, render_pipeline, vertex_buffer,
            camera, camera_controller, camera_uniform, camera_buffer, camera_bind_group,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.update_camera_matrix();
        }
    }

    // Explicit function to push matrix changes to the GPU
    fn update_camera_matrix(&mut self) {
        let aspect = self.config.width as f32 / self.config.height as f32;
        self.camera_uniform.update_view_proj(&self.camera, aspect);
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera_uniform]));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.196, g: 0.196, b: 0.196, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}

async fn run() {
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(WindowBuilder::new().with_title("PolyCraft Viewport").build(&event_loop).unwrap());
    
    let mut state = State::new(window.clone()).await;

    let _ = event_loop.run(move |event, elwt| {
        match event {
            // 1. Capture unbounded mouse delta for rotation
            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
                state.camera_controller.process_mouse_motion(delta.0, delta.1, &mut state.camera);
                state.update_camera_matrix();
            }
            Event::WindowEvent { ref event, window_id } if window_id == state.window.id() => {
                match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(physical_size) => state.resize(*physical_size),
                    // 2. Track left-click hold state
                    WindowEvent::MouseInput { state: button_state, button, .. } => {
                        state.camera_controller.process_mouse_input(*button_state, *button);
                    }
                    WindowEvent::RedrawRequested => {
                        match state.render() {
                            Ok(_) => {}
                            Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                            Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                            Err(e) => eprintln!("{:?}", e),
                        }
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
                state.window.request_redraw();
            }
            _ => {}
        }
    });
}

fn main() {
    pollster::block_on(run());
}