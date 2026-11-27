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

/// The most gizmo line vertices the handle can ever need: three arms, plus the three rotate rings when the
/// Rotate tool is up, each of those drawn as [`gizmo::RING_SEGMENTS`] chords of two vertices.
///
/// Sized for the largest handle rather than the common one, because `upload_gizmo` writes the frame it built
/// straight into a buffer allocated here and wgpu treats running past the end as a validation error, not a
/// warning. A frame is rebuilt every mouse event, so this is one allocation held for the life of the window.
const GIZMO_VERTEX_COUNT: usize =
    GizmoAxis::ALL.len() * 2 + GizmoAxis::ALL.len() * gizmo::RING_SEGMENTS * 2;

/// The most solid arrowhead vertices the handle can ever need: three tips, each [`gizmo::TIP_SEGMENTS`]
/// side faces plus the same number of cap faces, at three unshared vertices per flat-shaded face.
///
/// Sized the same way as [`GIZMO_VERTEX_COUNT`] — for the largest handle, because the tip buffer is
/// written whole and running past its end is a wgpu validation error rather than a warning.
const GIZMO_TIP_VERTEX_COUNT: usize = GizmoAxis::ALL.len() * gizmo::TIP_SEGMENTS * 2 * 3;

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

/// Coverage samples per pixel for the viewport's colour and depth buffers.
///
/// The wireframe is a `LineList`, so a 3D edge lands on one row of pixel centres. Face-on that reads as a
/// clean line; tilt the model and a long edge climbs through the grid in a staircase, and wherever a
/// segment misses every centre in a column the line goes away for a pixel or two - which is what makes the
/// same edge look solid from one angle and dotted from another. Four samples per pixel replace that
/// on/off decision with a partial blend, and because the depth buffer is sampled too, the edges keep
/// their own antialiasing instead of gaining a hard jagged silhouette.
///
/// Every pipeline that draws into these buffers has to ask for the same count, so the number lives here
/// rather than being spelled out at each `MultisampleState`.
pub const VIEWPORT_SAMPLE_COUNT: u32 = 4;

/// The colour buffer the scene is drawn into, before it is resolved onto the surface.
///
/// A surface texture cannot be multisampled itself, so the scene renders here and the render pass resolves
/// it into the surface in the same pass - which is why the pipelines name `VIEWPORT_SAMPLE_COUNT` while the
/// UI pass, drawing straight to the surface afterwards, stays at one.
fn create_msaa_color_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: config.width.max(1),
            height: config.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: VIEWPORT_SAMPLE_COUNT,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        label: Some("Viewport MSAA Texture"),
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// The depth buffer, sized and sampled to match the colour buffer it is paired with.
///
/// wgpu requires both attachments of a render pass to agree on their sample count, so the two are created
/// from one place and `resize` refreshes them together - a depth buffer left at one sample while the
/// colour buffer moves to four fails at the first draw call rather than at the call that set it up.
///
/// `RENDER_ATTACHMENT` alone, not `TEXTURE_BINDING`: picking is done by ray-triangles against the mesh on
/// the CPU (`picking.rs`), so nothing ever samples depth, and a multisampled texture can only be sampled
/// with an explicit per-sample load anyway.
fn create_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: VIEWPORT_SAMPLE_COUNT,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        label: Some("Depth Texture"),
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
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
            count: VIEWPORT_SAMPLE_COUNT,
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

/// The most a rotate drag may turn in one cursor event, in radians.
///
/// The sibling of [`MAX_EXTRUDE_STEP_ARMS`] for the same hazard: sighting nearly edge-on to the plane a ring
/// is measured in, the projected bearing leaps for a pixel of cursor travel. A cap this size still lets a
/// brisk drag reach a full turn in a handful of events, while a wild projection turns the model at the pace
/// the cursor can be followed rather than the pace the arithmetic ran at.
const MAX_ROTATE_STEP_RADIANS: f32 = 0.35;

/// Cursor bookkeeping for a rotate drag, which measures a *sweep* rather than a distance.
///
/// [`DragAnchor`] differences a scalar that runs along one line, which is what a move or a scale needs. A
/// turn has no such scalar, and the one way of forcing it is worse than the small steps kept here: an
/// absolute bearing from `atan2` jumps from +pi to -pi as the handle passes behind the pivot, and a
/// difference taken across that jump reads a whole turn the wrong way round.
struct SweepAnchor {
    /// The cursor's bearing at the previous event, as a unit vector across the plane of the ring.
    ///
    /// A bearing rather than a position, because only a point *with* a bearing may become the reference for
    /// the next step: the cursor crosses the line of rotation mid-drag, and storing that would freeze the
    /// turn from then on, since every later step would be measured from somewhere with nothing to measure
    /// from. It also drops the cursor's distance from the pivot, which is not part of a turn.
    previous: Option<cgmath::Vector3<f32>>,
    /// Total turn since the grab, in radians.
    ///
    /// Absolute, because [`Scene::rotate_vertices`] recomputes every vertex from the grab snapshot and so
    /// expects the angle the drag has reached rather than the next step toward it.
    applied: f32,
    /// Set when a camera change swallowed events, so the next projection only re-baselines.
    needs_reanchor: bool,
}

impl SweepAnchor {
    /// Starts a sweep from the cursor's bearing when the handle was grabbed, if the view could say one.
    fn at(
        hit: Option<cgmath::Point3<f32>>,
        pivot: cgmath::Point3<f32>,
        axis: cgmath::Vector3<f32>,
    ) -> Self {
        let bearing = hit.and_then(|hit| gizmo::bearing_across(hit - pivot, axis));
        Self {
            previous: bearing,
            applied: 0.0,
            needs_reanchor: bearing.is_none(),
        }
    }

    /// Folds one cursor event into the sweep, capping the step at `limit` radians.
    fn advance(
        &mut self,
        hit: cgmath::Point3<f32>,
        pivot: cgmath::Point3<f32>,
        axis: cgmath::Vector3<f32>,
        limit: f32,
    ) {
        // No bearing means no information, and the angle already banked stays. Retuning the instrument to
        // the model it came from rather than to the reading it failed to take.
        let Some(bearing) = gizmo::bearing_across(hit - pivot, axis) else {
            return;
        };

        let previous = self.previous.replace(bearing);
        let reanchored = std::mem::take(&mut self.needs_reanchor);

        // Either the first bearing the drag has been given, or the view moved since the last and the two
        // were measured from different sight lines. Both mean note the bearing, turn nothing.
        let Some(previous) = previous.filter(|_| !reanchored) else {
            return;
        };
        if let Some(step) = gizmo::signed_angle(previous, bearing, axis) {
            self.applied += step.clamp(-limit, limit);
        }
    }

    /// Notes that the view moved, so the next bearing cannot be compared with the last.
    fn mark_swung(&mut self) {
        self.needs_reanchor = true;
    }
}

/// How far a grabbed handle has carried its gesture, in the unit that tool reads.
enum Progress {
    /// Travel along the drag axis, in model units: what Move offsets by, and what Scale divides by the
    /// selection's own width to become a proportion.
    Linear(DragAnchor),
    /// Sweep about the drag axis, in radians.
    Angular(SweepAnchor),
}

impl Progress {
    fn mark_swung(&mut self) {
        match self {
            Progress::Linear(anchor) => anchor.mark_swung(),
            Progress::Angular(anchor) => anchor.mark_swung(),
        }
    }
}

struct DragState {
    /// The tool this drag was grabbed with, which decides what the cursor's movement means.
    ///
    /// Held for the drag's whole life like `frame`, for the same reason: `E` arriving mid-pull must not
    /// reinterpret travel that has already been banked as a turn of however far the mouse had gone.
    tool: Tool,
    axis: GizmoAxis,
    /// The orientation this drag was grabbed in, kept for its whole life rather than re-read each event.
    ///
    /// Re-deriving it from the selection would make the drag follow the *current* space while the arms the
    /// user is pulling were drawn in another, and would also let a selection change mid-drag swing the
    /// travel axis under the cursor. Both draw and pick read the same value back out of here.
    frame: gizmo::Frame,
    progress: Progress,
    /// How wide the selection is along the drag axis, captured at the grab.
    ///
    /// The rate a scale drag works at. Re-measuring it as the drag went would shrink that rate as the
    /// geometry grew, so the handle would run away from the cursor - and growing a wall is exactly the drag
    /// where that matters most.
    span: f32,
    initial_positions: Vec<(VertexId, cgmath::Point3<f32>)>,
}

impl DragState {
    /// Grabs `axis` of `frame` with the cursor where it is, snapshotting the geometry the gesture works
    /// from and reading off the two numbers the drag will need for its whole life.
    ///
    /// `initial_positions` is the anchor for every write, which is what lets a drag be corrected by a later
    /// event instead of accumulating, and what makes [`DragState::restore`] an identity rather than an
    /// inversion. `span` is measured from the same snapshot, before the first event can move anything.
    fn begin(tool: Tool, frame: gizmo::Frame, axis: GizmoAxis, ray: &Ray) -> Self {
        let direction = frame.direction(axis);
        let initial_positions: Vec<(VertexId, cgmath::Point3<f32>)> = Vec::new();
        let progress = match tool {
            Tool::Rotate => Progress::Angular(SweepAnchor::at(
                gizmo::project_ray_onto_plane(frame.origin, direction, ray),
                frame.origin,
                direction,
            )),
            _ => Progress::Linear(DragAnchor::at(gizmo::project_ray_onto_axis(
                frame.origin,
                direction,
                ray,
            ))),
        };
        Self {
            tool,
            axis,
            frame,
            progress,
            span: 0.0,
            initial_positions,
        }
    }

    /// Fills in the selection snapshot the caller gathered, and the width of that selection along the drag
    /// axis, which is the rate a scale drag works at.
    fn with_selection(mut self, initial_positions: Vec<(VertexId, cgmath::Point3<f32>)>) -> Self {
        // Measured here rather than at `begin` so there is one place the snapshot is taken, and one place
        // that reads it. A selection spanning nothing along the axis gives a zero span, which
        // `gizmo::scale_factor` turns into a rate of 1.0 - a stuck handle rather than an infinite one.
        let span = gizmo::span_along(
            &initial_positions
                .iter()
                .map(|&(_, p)| p)
                .collect::<Vec<_>>(),
            self.direction(),
        );
        self.initial_positions = initial_positions;
        self.span = span;
        self
    }

    /// The direction this drag pulls along, from the frame it was grabbed with.
    fn direction(&self) -> cgmath::Vector3<f32> {
        self.frame.direction(self.axis)
    }

    /// Folds one cursor event into the gesture and writes the result over the document.
    ///
    /// Takes the `Scene` rather than reaching through `State`, because reading this anchor and writing the
    /// mesh are two mutable borrows that cannot overlap - and the split is worth keeping on its own terms: it
    /// puts "how far has the gesture gone" and "what does that do to the mesh" on either side of one line.
    fn advance(&mut self, scene: &mut Scene, ray: &Ray) {
        let pivot = self.frame.origin;
        let direction = self.direction();

        match &mut self.progress {
            Progress::Linear(anchor) => {
                // A sight line running down the axis says nothing about travel along it, so the drag holds
                // the distance it had already reached rather than inventing one.
                let Some(param) = gizmo::project_ray_onto_axis(pivot, direction, ray) else {
                    return;
                };
                anchor.advance(param, f32::MAX);
                match self.tool {
                    Tool::Scale => scene.scale_vertices(
                        &self.initial_positions,
                        pivot,
                        direction,
                        gizmo::scale_factor(anchor.applied, self.span),
                    ),
                    _ => scene
                        .translate_vertices(&self.initial_positions, direction * anchor.applied),
                }
            }
            Progress::Angular(anchor) => {
                let Some(hit) = gizmo::project_ray_onto_plane(pivot, direction, ray) else {
                    return;
                };
                anchor.advance(hit, pivot, direction, MAX_ROTATE_STEP_RADIANS);
                scene.rotate_vertices(&self.initial_positions, pivot, direction, anchor.applied);
            }
        }
    }

    /// Puts the document back where the grab found it, which is how a cancelled drag leaves nothing behind.
    ///
    /// Every tool writes positions recomputed from [`DragState::initial_positions`] by however far the cursor
    /// has reached, so cancelling is a matter of applying the tool's *identity* - no travel, no turn, unit
    /// stretch - rather than running the gesture's arithmetic backwards. The test
    /// `the_angle_is_absolute_so_zero_restores_and_negating_does_not` in `scene` is what pins that property
    /// down, and it is why Escape needs no memory of the steps a drag took.
    fn restore(&self, scene: &mut Scene) {
        let pivot = self.frame.origin;
        let direction = self.direction();
        match self.tool {
            Tool::Rotate => scene.rotate_vertices(&self.initial_positions, pivot, direction, 0.0),
            Tool::Scale => scene.scale_vertices(&self.initial_positions, pivot, direction, 1.0),
            _ => scene
                .translate_vertices(&self.initial_positions, cgmath::Vector3::new(0.0, 0.0, 0.0)),
        }
    }
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
    gizmo_tip_pipeline: wgpu::RenderPipeline,
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
    gizmo_tip_buffer: wgpu::Buffer,
    num_gizmo_tip_vertices: u32,
    depth_texture_view: wgpu::TextureView,
    /// The multisampled colour buffer the scene renders into, resolved onto the surface at the end of the
    /// same pass. Rebuilt with the depth buffer whenever the window changes size.
    msaa_view: wgpu::TextureView,
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

/// Which handle is capped with solid arrowheads: the Move handle alone, for now.
///
/// Rotate's rings are drawn at the radius the arms reach, so cones at those tips would sit under the
/// rings where the user aims to grab them; Scale's tips will be cubes rather than cones. Keeping the
/// choice in one pure function is what stops the drawn tips and the picked handle disagreeing about
/// which tool owns a solid head.
fn draws_gizmo_tips(tool: Tool) -> bool {
    tool == Tool::Translate
}

/// A keyboard action the viewport responds to. Produced by [`hotkey_for`], carried out by
/// `State::handle_hotkey`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Hotkey {
    SetTool(Tool),
    SetMode(EditMode),
    ToggleObjectComponent,
    CancelDrag,
    Undo,
    Redo,
    Extrude,
    EdgeLoop,
    Bevel,
    Delete,
    CycleSpace,
    SelectAll,
    SelectNone,
    InvertSelection,
}

/// The Maya keymap, as a pure function of the physical key and the Ctrl / Shift modifiers.
///
/// A free function rather than inline event-loop matching so the bindings - including the modifier
/// rules below - are unit-testable without a window or a GPU context.
///
/// # Bound exactly as Maya binds them
///
/// `Q` / `W` / `E` / `R` are Select / Move / Rotate / Scale, `Ctrl+Z` and `Ctrl+Y` (or
/// `Ctrl+Shift+Z`) are undo / redo, and the four selection-mode keys are Maya's own: **`F8`** toggles
/// object and component editing, **`F9` / `F10` / `F11`** pick Vertex / Edge / Face. `F12` is Maya's UV
/// key and is deliberately unbound, since there is no UV layer here to switch to.
///
/// The three select commands keep Maya's chords exactly: **`Ctrl+A`** All, **`Ctrl+Shift+A`** None,
/// **`Ctrl+Shift+I`** Invert. `Ctrl+A` is the one place this app knowingly overwrites Maya, whose `Ctrl+A`
/// opens the Attribute Editor - there is no attribute system here to displace, and `menu.rs` records the
/// claim so it is renegotiated rather than quietly inherited if one lands.
///
/// All three act on the mode in use, matching Maya: `Ctrl+A` in Face mode selects every face, and a
/// selection made in one mode is not carried into another.
///
/// **`Del`** is Maya's key for removing a component and is taken unchanged - the only modelling command
/// here that needs no modifier, which is why `Ctrl+Delete` and `Shift+Delete` deliberately do nothing
/// rather than reaching the same command by a second chord.
///
/// The tool keys only fire with Ctrl released: `Ctrl+Z` sends `KeyZ`, and without that guard a
/// `Ctrl` chord would silently switch tools. They also require Shift released, which frees the
/// `Shift` + letter block below.
///
/// # Ours, because Maya leaves these on the shelf
///
/// `Ctrl+E` matches Maya's *Edit Mesh > Extrude* and is the chord to prefer; `Shift+E` is kept as a
/// neighbour so the three modelling commands can be hit from one hand. Maya binds no default key to the
/// *Insert Edge Loop Tool* or to *Bevel*, so `Shift+L` (loop) and `Shift+B` are ours by the same
/// convention - chosen to sit together rather than scattered by whichever letter was free. Both collide
/// with Maya's animation hotkeys (`Shift+E` sets a rotate key), which is acceptable in a modelling-only
/// tool and worth revisiting if animation ever lands.
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
            Key::KeyE => Some(Hotkey::Extrude),
            Key::KeyA if shift => Some(Hotkey::SelectNone),
            Key::KeyA => Some(Hotkey::SelectAll),
            Key::KeyI if shift => Some(Hotkey::InvertSelection),
            _ => None,
        };
    }

    if shift {
        return match key {
            Key::KeyE => Some(Hotkey::Extrude),
            Key::KeyL => Some(Hotkey::EdgeLoop),
            Key::KeyB => Some(Hotkey::Bevel),
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
        Key::F8 => Some(Hotkey::ToggleObjectComponent),
        Key::F9 => Some(Hotkey::SetMode(EditMode::Vertex)),
        Key::F10 => Some(Hotkey::SetMode(EditMode::Edge)),
        Key::F11 => Some(Hotkey::SetMode(EditMode::Face)),
        Key::Delete => Some(Hotkey::Delete),
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

        let (_, depth_texture_view) = create_depth_texture(&device, config.width, config.height);
        // Dropping the texture handle here is deliberate, and follows the depth buffer beside it: the view
        // keeps its buffer alive, which is all the render pass needs.
        let (_, msaa_view) = create_msaa_color_texture(&device, &config);

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
                count: VIEWPORT_SAMPLE_COUNT,
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

        // The solid arrowheads of the Move handle. Lit by the mesh's own shaders so the head reads as a
        // piece of geometry rather than a decal, with back faces culled - the cone's winding is
        // counter-clockwise from outside exactly as the mesh's is, so culling is what makes it look solid.
        // Depth still always-passes, matching the arms: occluding the tips while the shafts stayed on top
        // would split one handle across two depth rules.
        let gizmo_tip_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Gizmo Tip Pipeline"),
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
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: VIEWPORT_SAMPLE_COUNT,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

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
                count: VIEWPORT_SAMPLE_COUNT,
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

        let gizmo_tip_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gizmo Tip Buffer"),
            size: (GIZMO_TIP_VERTEX_COUNT * std::mem::size_of::<GpuVertex>()) as u64,
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
            gizmo_tip_pipeline,
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
            gizmo_tip_buffer,
            num_gizmo_tip_vertices: 0,
            depth_texture_view,
            msaa_view,
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
            Some(Hotkey::SetMode(mode)) => {
                self.scene.set_mode(mode);
                // Echoed like the space cycle: the mode is shown in the panel title, which is easy to miss
                // mid-task, and a mode switch also drops the selection that no longer applies.
                self.notice = Some(format!("Selection mode: {}", mode.label()));
                true
            }
            Some(Hotkey::ToggleObjectComponent) => {
                self.scene.toggle_object_component();
                self.notice = Some(format!("Selection mode: {}", self.scene.mode().label()));
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
            Some(Hotkey::Bevel) => {
                self.bevel_selected_edge();
                true
            }
            Some(Hotkey::Delete) => {
                self.delete_selected();
                true
            }
            Some(Hotkey::CycleSpace) => {
                self.cycle_reference_system();
                true
            }
            Some(Hotkey::SelectAll) => {
                self.select_all();
                true
            }
            Some(Hotkey::SelectNone) => {
                self.select_none();
                true
            }
            Some(Hotkey::InvertSelection) => {
                self.invert_selection();
                true
            }
            None => false,
        }
    }

    /// Advances the transform gizmo to the next reference system and echoes it.
    ///
    /// Shared with the menu row because the only signal otherwise is the handle reorienting, which is easy
    /// to miss on a face whose normal already runs along a world axis - and a menu click that echoed
    /// something different from the key would read as two commands, not two routes to one.
    fn cycle_reference_system(&mut self) {
        self.scene.cycle_space();
        self.notice = Some(format!("Gizmo space: {}", self.scene.space().label()));
    }

    /// Selects every component of the mode in use and echoes how many, so the command is visible even when
    /// the whole mesh was already chosen and nothing on screen changes.
    fn select_all(&mut self) {
        let count = self.scene.select_all();
        self.notice = Some(format!(
            "Selected {} {}",
            count,
            self.scene.mode().plural_label().to_lowercase()
        ));
    }

    /// Drops the selection. Echoed unconditionally: with nothing selected the command changes nothing, and
    /// a key that answers nothing looks broken.
    fn select_none(&mut self) {
        self.scene.select_none();
        self.notice = Some("Selection cleared".to_string());
    }

    /// Swaps the selected and unselected components of the mode in use.
    fn invert_selection(&mut self) {
        let count = self.scene.invert_selection();
        self.notice = Some(format!(
            "Inverted: {} {} selected",
            count,
            self.scene.mode().plural_label().to_lowercase()
        ));
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

    /// Trims the selected edge, opening a facet where it ran, and echoes what the bevel did.
    ///
    /// The gesture in flight is closed first for the reason `insert_edge_loop` documents: a bevel moves
    /// corners, and an armed extrude previews from the snapshot taken the instant it was armed, so leaving
    /// one open underneath a bevel would have its click restore geometry the bevel has already cut away.
    /// Like the loop there is nothing to drag - a facet appears on the press - so no follow-up state of its
    /// own, and a refusal arrives as a notice rather than silence.
    pub fn bevel_selected_edge(&mut self) {
        self.cancel_active_drag();

        match self.scene.bevel_selected_edge() {
            Ok(bevel) => {
                self.notice = Some(format!(
                    "Beveled the edge, opening facet {} between the faces it joined",
                    bevel.facet.0
                ));
            }
            Err(errors) => self.notice = Some(errors.join(" ")),
        }
    }

    /// Retires the selected components, opening the mesh where they were, and echoes what left.
    ///
    /// The gesture in flight is closed first for the reason `insert_edge_loop` documents, and the reason is
    /// sharper here than it is there: an armed extrude previews from the snapshot taken when it was armed,
    /// so a click after a delete would write back the faces that have just been removed. Nothing of its own
    /// follows a delete - what goes is decided by the selection, not the cursor - so there is no drag state
    /// to hand over to, and a refusal arrives as a notice rather than silence.
    pub fn delete_selected(&mut self) {
        self.cancel_active_drag();

        match self.scene.delete_selected() {
            Ok(gone) => {
                let mut echo = format!(
                    "Deleted {} {}",
                    gone.faces,
                    if gone.faces == 1 { "face" } else { "faces" }
                );
                // Only worth saying when corners actually went: they leave no outline of their own, so a
                // count is the sole sign that the mesh shrank rather than merely opened.
                if gone.vertices > 0 {
                    echo.push_str(&format!(
                        ", dropping {} {} that no surviving face needed",
                        gone.vertices,
                        if gone.vertices == 1 {
                            "corner"
                        } else {
                            "corners"
                        }
                    ));
                }
                self.notice = Some(echo);
            }
            Err(errors) => self.notice = Some(errors.join(" ")),
        }
    }

    /// Grows a face across every opening in the mesh and echoes how many closed.
    ///
    /// The gesture in flight is closed first for the reason `delete_selected` documents, and the reason is
    /// as sharp here: an armed extrude previews from the snapshot taken when it was armed, so writing caps
    /// into the document and then letting that drag settle would paste the pre-fill faces back over the
    /// top of them. Nothing of its own follows a fill - the cursor decides nothing - so there is no state to
    /// hand over, and a refusal arrives as a notice rather than silence.
    pub fn fill_holes(&mut self) {
        self.cancel_active_drag();

        match self.scene.fill_holes() {
            Ok(closed) => {
                let mut echo = format!(
                    "Filled {} {}",
                    closed.faces,
                    if closed.faces == 1 {
                        "opening"
                    } else {
                        "openings"
                    }
                );
                // Worth spelling out only when the caps were not all the same shape, because the count is
                // what tells a user that the command reached openings they had not meant to close - and a
                // triangle in the list is the sign the rim was not the clean quad a delete leaves. Read off
                // `first` rather than indexed, so an echo can never be the thing that panics whatever the
                // primitive handed back.
                let mixed = closed
                    .corners
                    .first()
                    .is_some_and(|&first| closed.corners.iter().any(|&corners| corners != first));
                if mixed {
                    let shapes = closed
                        .corners
                        .iter()
                        .map(|corners| corners.to_string())
                        .collect::<Vec<String>>()
                        .join(", ");
                    echo.push_str(&format!(" ({shapes} corners each)"));
                }
                self.notice = Some(echo);
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
            drag.progress.mark_swung();
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

            // Both attachments are rebuilt at the new size and have to agree on their sample count, so
            // they come from the same helpers that created them. The texture handles are dropped on
            // purpose, as they are in `new`: a wgpu view holds its texture alive, and nothing here ever
            // needs the texture itself again.
            let (_, depth) = create_depth_texture(&self.device, new_size.width, new_size.height);
            let (_, msaa) = create_msaa_color_texture(&self.device, &self.config);
            self.depth_texture_view = depth;
            self.msaa_view = msaa;

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
    /// it the way [`gizmo::handle`] does; sharing one pick helper across hover and press is what keeps
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
            self.num_gizmo_tip_vertices = 0;
            return;
        }

        if !self.scene.tool().has_gizmo() {
            self.num_gizmo_vertices = 0;
            self.num_gizmo_tip_vertices = 0;
            return;
        }

        let active_axis = self
            .drag_state
            .as_ref()
            .map(|d| d.axis)
            .or(self.hovered_axis);

        let rotating = self.scene.tool() == Tool::Rotate;
        let tips = match self.gizmo_frame() {
            Some(frame) => {
                // The same length the arms are drawn at and the picker tests at, so a tip always caps the
                // end of the arm it belongs to rather than floating past it.
                let length = gizmo::axis_world_length(
                    frame.origin,
                    self.camera.eye(),
                    self.config.height as f32,
                );
                if draws_gizmo_tips(self.scene.tool()) {
                    gizmo::arrowheads(&frame, length, active_axis)
                } else {
                    Vec::new()
                }
            }
            None => Vec::new(),
        };
        let arms = match self.gizmo_frame() {
            Some(frame) => gizmo::handle(
                &frame,
                self.camera.eye(),
                self.config.height as f32,
                active_axis,
                rotating,
            ),
            None => Vec::new(),
        };
        let vertices = overlay::gizmo_line_vertices(&arms);
        self.num_gizmo_vertices = vertices.len() as u32;
        if self.num_gizmo_vertices > 0 {
            self.queue
                .write_buffer(&self.gizmo_buffer, 0, bytemuck::cast_slice(&vertices));
        }

        let tip_vertices = overlay::gizmo_tip_vertices(&tips);
        self.num_gizmo_tip_vertices = tip_vertices.len() as u32;
        if self.num_gizmo_tip_vertices > 0 {
            self.queue.write_buffer(
                &self.gizmo_tip_buffer,
                0,
                bytemuck::cast_slice(&tip_vertices),
            );
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
            } else if self.drag_state.is_some() {
                // Taken out rather than borrowed in place: `advance` needs `&mut Scene`, and `self.drag_state`
                // is a field of the same `self`, so the two borrows cannot overlap. Putting it back at the end
                // of the block keeps the gesture open across the next event.
                let mut drag = self.drag_state.take().expect("just tested above");
                drag.advance(&mut self.scene, &ray);
                self.drag_state = Some(drag);
            } else if self.scene.tool().has_gizmo()
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
                    let frame = if self.scene.tool().has_gizmo() {
                        self.gizmo_frame()
                    } else {
                        None
                    };
                    let hit_axis =
                        frame.and_then(|frame| self.gizmo_axis_under(frame, &ray, height));

                    if let Some((frame, axis)) = frame.zip(hit_axis) {
                        let initial_positions: Vec<(VertexId, cgmath::Point3<f32>)> = self
                            .scene
                            .selected_vertices()
                            .into_iter()
                            .map(|v| (v, self.scene.mesh().vertex_position(v)))
                            .collect();

                        let tool = self.scene.tool();
                        self.drag_state = Some(
                            DragState::begin(tool, frame, axis, &ray)
                                .with_selection(initial_positions),
                        );
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
            drag.restore(&mut self.scene);
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
        if response.bevel {
            self.bevel_selected_edge();
        }
        if response.delete {
            self.delete_selected();
        }
        if response.fill_holes {
            self.fill_holes();
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
                can_bevel: self.scene.can_bevel(),
                can_delete: self.scene.can_delete(),
                can_fill_holes: self.scene.can_fill_holes(),
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
            // Draws into the multisampled buffer and resolves onto the surface as the pass closes, in one
            // command - the surface itself cannot hold samples, so this is the only place the two views
            // meet. `Discard` rather than `Store`: the resolved surface is what survives, and keeping the
            // individual samples around would cost a full extra colour buffer per frame for nothing.
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("3D Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa_view,
                    resolve_target: Some(&view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.196,
                            g: 0.196,
                            b: 0.196,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        // Nothing reads depth once this pass closes - the UI pass has no depth attachment
                        // and picking is done on the CPU - so the four samples never need writing back to
                        // memory they would be cleared from next frame anyway.
                        store: wgpu::StoreOp::Discard,
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

            if self.num_gizmo_tip_vertices > 0 {
                render_pass.set_pipeline(&self.gizmo_tip_pipeline);
                render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.gizmo_tip_buffer.slice(..));
                render_pass.draw(0..self.num_gizmo_tip_vertices, 0..1);
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
        // `Ctrl+W` closes a window in every other application and `Ctrl+Q` quits it; neither must quietly
        // change the tool here. `E` is the exception, and deliberately so - Maya binds *Edit Mesh > Extrude*
        // to `Ctrl+E`, so the chord is taken by a real command rather than falling through to Rotate.
        assert_eq!(hotkey_for(Key::KeyE, true, false), Some(Hotkey::Extrude));
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
    fn the_modelling_chords_fire_and_bare_letters_do_not() {
        // The three commands that edit the mesh rather than the selection, all on the `Shift` + letter
        // convention. `Shift` is load-bearing twice over here: it keeps them off the tool keys (`E` alone is
        // Rotate), and a bare `B` must stay unbound so a stray press does not cut a facet into the mesh.
        assert_eq!(hotkey_for(Key::KeyE, false, true), Some(Hotkey::Extrude));
        assert_eq!(hotkey_for(Key::KeyL, false, true), Some(Hotkey::EdgeLoop));
        assert_eq!(hotkey_for(Key::KeyB, false, true), Some(Hotkey::Bevel));

        assert_eq!(
            hotkey_for(Key::KeyB, false, false),
            None,
            "bare B is not a bevel, and must not quietly trim an edge"
        );
        // Ctrl wins: `Ctrl+B` is bold in every other application, and here it must simply do nothing rather
        // than bevel behind a menu mnemonic.
        assert_eq!(hotkey_for(Key::KeyB, true, false), None);
        assert_eq!(hotkey_for(Key::KeyL, true, false), None);
    }

    #[test]
    fn the_selection_mode_keys_match_maya_exactly() {
        // Autodesk's own list puts component selection on F9/F10/F11 with F8 toggling object and component
        // editing - not on F2/F3/F4, which are its *menu set* keys. Getting this wrong is worse than not
        // binding it at all: a key that switches to the wrong mode is a bug a Maya user cannot name.
        assert_eq!(
            hotkey_for(Key::F8, false, false),
            Some(Hotkey::ToggleObjectComponent)
        );
        assert_eq!(
            hotkey_for(Key::F9, false, false),
            Some(Hotkey::SetMode(EditMode::Vertex))
        );
        assert_eq!(
            hotkey_for(Key::F10, false, false),
            Some(Hotkey::SetMode(EditMode::Edge))
        );
        assert_eq!(
            hotkey_for(Key::F11, false, false),
            Some(Hotkey::SetMode(EditMode::Face))
        );

        // F12 is Maya's UV key. There is no UV layer here, so it must stay unbound rather than be pressed
        // into service as something else - a borrowed key is indistinguishable from a broken one.
        assert_eq!(hotkey_for(Key::F12, false, false), None);
        assert_eq!(hotkey_for(Key::F1, false, false), None);
        // Held down, the mode keys do nothing: Maya has no Ctrl/Shift variant of them, and inventing one
        // would shadow a chord another application already owns.
        assert_eq!(hotkey_for(Key::F9, true, false), None);
        assert_eq!(hotkey_for(Key::F8, false, true), None);
    }

    #[test]
    fn delete_is_bound_to_the_delete_key_alone() {
        assert_eq!(hotkey_for(Key::Delete, false, false), Some(Hotkey::Delete));

        // Maya has no modifier variant of Delete. Inventing one would put an undiscoverable chord in front
        // of the operation that destroys geometry, and `Ctrl+Delete` is a text-editing key everywhere else.
        assert_eq!(hotkey_for(Key::Delete, true, false), None);
        assert_eq!(hotkey_for(Key::Delete, false, true), None);

        // Backspace is the key a user reaches for on a laptop, but it is Maya's text-editing key and this
        // app has no text field to protect yet - so a stray press would delete a face with no way to undo
        // the surprise. Held back deliberately rather than overlooked; see the digest's rule that a
        // borrowed key is indistinguishable from a broken one.
        assert_eq!(hotkey_for(Key::Backspace, false, false), None);
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
    fn only_the_translate_handle_is_capped_with_solid_tips() {
        assert!(draws_gizmo_tips(Tool::Translate));
        // Rotate's rings pass through the arm tips, and Scale's tips are cubes, so cones on either
        // would sit on top of the thing the user is aiming at.
        assert!(!draws_gizmo_tips(Tool::Rotate));
        assert!(!draws_gizmo_tips(Tool::Scale));
        assert!(!draws_gizmo_tips(Tool::Select));
    }

    #[test]
    fn a_full_set_of_tips_fits_the_tip_buffer() {
        // Running past the end of the fixed gizmo buffers is a wgpu validation error, not a warning.
        let frame = gizmo::Frame::world(cgmath::Point3::new(0.0, 0.0, 0.0));
        let faces = gizmo::arrowheads(&frame, 1.0, None);
        assert!(overlay::gizmo_tip_vertices(&faces).len() <= GIZMO_TIP_VERTEX_COUNT);
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

    /// A point on the unit circle about Z at `degrees`, for the sweep tests below.
    fn bearing(degrees: f32) -> cgmath::Point3<f32> {
        let radians = degrees.to_radians();
        cgmath::Point3::new(radians.cos(), radians.sin(), 0.0)
    }

    /// The Z axis through the origin, which every sweep test below turns about.
    const SPIN: cgmath::Vector3<f32> = cgmath::Vector3 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };
    const ORIGIN: cgmath::Point3<f32> = cgmath::Point3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    #[test]
    fn an_undisturbed_sweep_tracks_the_cursor_exactly() {
        // The rotation analogue of the translate test above: with the camera still, a quarter of the way
        // round the ring is a quarter of a turn, and nothing about the incremental bookkeeping leaks.
        let mut sweep = SweepAnchor::at(Some(bearing(0.0)), ORIGIN, SPIN);
        sweep.advance(bearing(45.0), ORIGIN, SPIN, f32::MAX);
        sweep.advance(bearing(90.0), ORIGIN, SPIN, f32::MAX);
        assert!(
            (sweep.applied - std::f32::consts::FRAC_PI_2).abs() < 1e-5,
            "half way round a quarter turn should be a quarter turn, got {}",
            sweep.applied
        );

        sweep.advance(bearing(0.0), ORIGIN, SPIN, f32::MAX);
        assert!(
            sweep.applied.abs() < 1e-5,
            "carrying the handle back to where the grab found it must land there, got {}",
            sweep.applied
        );
    }

    #[test]
    fn a_sweep_keeps_going_past_where_an_angle_would_flip() {
        // The reason this anchor exists rather than storing one absolute bearing. `atan2` returns
        // (-pi, pi], so a handle carried past the far side of the ring jumps from +pi to -pi; a difference
        // taken across that reads a whole turn the wrong way round, and the model snaps. Stepping in small
        // arcs and summing never sees the discontinuity, so a full circle plus ten degrees is one circle
        // plus ten degrees.
        let mut sweep = SweepAnchor::at(Some(bearing(0.0)), ORIGIN, SPIN);
        for degrees in (10..=370).step_by(10) {
            sweep.advance(bearing(degrees as f32), ORIGIN, SPIN, f32::MAX);
        }
        assert!(
            (sweep.applied - 370.0f32.to_radians()).abs() < 1e-4,
            "the seam at 180 degrees must pass unnoticed, got {} radians",
            sweep.applied
        );
    }

    #[test]
    fn swinging_the_camera_mid_sweep_absorbs_the_jump() {
        // Same hazard as the translate drag and the same cure: the two bearings were measured from different
        // sight lines, so the projection after the swing only re-baselines. Without this, orbiting for a
        // better look mid-turn would spin the geometry by however far the camera went.
        let mut sweep = SweepAnchor::at(Some(bearing(0.0)), ORIGIN, SPIN);
        sweep.advance(bearing(20.0), ORIGIN, SPIN, f32::MAX);
        sweep.mark_swung();
        sweep.advance(bearing(250.0), ORIGIN, SPIN, f32::MAX);
        assert!(
            (sweep.applied - 20.0f32.to_radians()).abs() < 1e-5,
            "the swing itself must contribute nothing, got {}",
            sweep.applied
        );

        sweep.advance(bearing(260.0), ORIGIN, SPIN, f32::MAX);
        assert!(
            (sweep.applied - 30.0f32.to_radians()).abs() < 1e-5,
            "and the sweep resumes from the angle it had already reached, got {}",
            sweep.applied
        );
    }

    #[test]
    fn a_sweep_armed_without_a_projection_waits_for_the_cursor() {
        // Grabbed with the cursor outside the window, or sighting the ring edge-on: there is no bearing to
        // start from, and inventing one would spin the model the instant the mouse stirred.
        let mut sweep = SweepAnchor::at(None, ORIGIN, SPIN);
        sweep.advance(bearing(200.0), ORIGIN, SPIN, f32::MAX);
        assert_eq!(
            sweep.applied, 0.0,
            "the first projection only sets the bearing"
        );

        sweep.advance(bearing(210.0), ORIGIN, SPIN, f32::MAX);
        assert!(
            (sweep.applied - 10.0f32.to_radians()).abs() < 1e-5,
            "and the second begins the turn, got {}",
            sweep.applied
        );
    }

    #[test]
    fn one_wild_sweep_step_is_capped_but_the_drag_catches_up() {
        // Nearly edge-on to the ring's plane, a pixel of cursor travel swings the projected bearing by
        // nearly a half turn. The cap holds one event to a pace the user can follow; being per event rather
        // than total, a steady drag still reaches anywhere it is pointed.
        let mut sweep = SweepAnchor::at(Some(bearing(0.0)), ORIGIN, SPIN);

        // A 170 degree leap asks for 2.97 radians; the cap is what stops the model snapping round.
        sweep.advance(bearing(170.0), ORIGIN, SPIN, 0.2);
        assert!(
            (sweep.applied - 0.2).abs() < 1e-6,
            "one event cannot outrun the handle, got {}",
            sweep.applied
        );

        // A small step is not capped, so the drag carries on from the angle it reached rather than being
        // permanently slowed by the one event that ran wild.
        sweep.advance(bearing(175.0), ORIGIN, SPIN, 0.2);
        assert!(
            (sweep.applied - (0.2 + 5.0f32.to_radians())).abs() < 1e-6,
            "steady tracking keeps gaining, got {}",
            sweep.applied
        );

        // And the same limit holds coming back, measured the way the cursor actually went: 175 degrees back
        // round is a leap of -3.05, capped to -0.2, not a short-cut the other way.
        sweep.advance(bearing(0.0), ORIGIN, SPIN, 0.2);
        assert!(
            (sweep.applied - 5.0f32.to_radians()).abs() < 1e-6,
            "the cap must bite in both directions, got {}",
            sweep.applied
        );
    }

    #[test]
    fn a_sweep_holds_its_angle_when_the_cursor_loses_its_bearing() {
        // Mid-drag the cursor can cross the line of rotation, where it has no bearing at all. The turn stops
        // rather than reading zero and writing the model back to where the grab began - which would look like
        // the handle had come loose and shaken the geometry off.
        let mut sweep = SweepAnchor::at(Some(bearing(0.0)), ORIGIN, SPIN);
        sweep.advance(bearing(40.0), ORIGIN, SPIN, f32::MAX);
        let held = sweep.applied;

        // Straight up the axis: on the line of rotation, so no bearing to measure from.
        sweep.advance(cgmath::Point3::new(0.0, 0.0, 5.0), ORIGIN, SPIN, f32::MAX);
        assert_eq!(
            sweep.applied, held,
            "the angle must survive the pass through the axis"
        );

        // And it carries on from there, not from the other side of the circle.
        sweep.advance(bearing(50.0), ORIGIN, SPIN, f32::MAX);
        assert!(
            (sweep.applied - (held + 10.0f32.to_radians())).abs() < 1e-5,
            "the drag resumes where it stopped, got {}",
            sweep.applied
        );
    }

    #[test]
    fn each_tool_banks_its_gesture_in_its_own_unit() {
        // The one place a rotate drag and a scale drag part company is which anchor they carry, and getting
        // it wrong is invisible until the model shears: a rotate reading travel as an angle, or a scale
        // dividing a distance by the wrong span. `Select` is here because it must not grab a handle at all,
        // which `Tool::has_gizmo` is what the draw and pick paths both ask.
        let frame = gizmo::Frame::world(cgmath::Point3::new(0.0, 0.5, 0.0));
        let ray = Ray::new(
            cgmath::Point3::new(4.0, 0.5, 4.0),
            cgmath::Vector3::new(-1.0, 0.0, -1.0),
        );
        for (tool, angular) in [
            (Tool::Translate, false),
            (Tool::Rotate, true),
            (Tool::Scale, false),
        ] {
            let drag = DragState::begin(tool, frame, GizmoAxis::Z, &ray);
            assert_eq!(
                matches!(drag.progress, Progress::Angular(_)),
                angular,
                "{tool:?} must {}measure a sweep",
                if angular { "" } else { "not " }
            );
            assert_eq!(
                drag.tool, tool,
                "a drag must remember what it was grabbed with"
            );
            assert_eq!(drag.frame, frame, "and the frame it was grabbed in");
        }

        assert!(!Tool::Select.has_gizmo());
    }

    #[test]
    fn every_tool_that_offers_a_handle_names_itself_in_history() {
        // The label is what the undo list shows, so a drag that records "Move Vertex" while it was turning
        // the geometry is a history the user cannot read after the fact.
        for tool in [Tool::Translate, Tool::Rotate, Tool::Scale] {
            for mode in [
                EditMode::Vertex,
                EditMode::Edge,
                EditMode::Face,
                EditMode::Object,
            ] {
                let label = drag_action_label(mode, tool);
                assert!(
                    label.starts_with(tool.label()),
                    "a {tool:?} drag logged itself as {label:?}, which does not name the tool"
                );
            }
        }
        assert_eq!(
            drag_action_label(EditMode::Vertex, Tool::Rotate),
            "Rotate Vertex"
        );
        assert_eq!(drag_action_label(EditMode::Face, Tool::Scale), "Scale Face");
        assert_eq!(drag_action_label(EditMode::Object, Tool::Scale), "Scale");
    }
}
