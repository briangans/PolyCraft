//! Shared GPU vertex structures and uniform layouts.
//!
//! Standardizes the memory layout across all render passes (3D shaded mesh, wireframe edges,
//! billboarded vertex square dots, and constant-screen-size transform gizmo arms).

use crate::camera::Camera;
use cgmath::SquareMatrix;

/// The primary GPU vertex struct, matching the attributes in `src/shader.wgsl`.
///
/// Attribute layout:
/// - `@location(0)`: `position` (`Float32x3`)
/// - `@location(1)`: `normal` (`Float32x3`) — holds surface normal for 3D faces, or screen-pixel offset for vertex dots
/// - `@location(2)`: `color` (`Float32x3`)
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

impl GpuVertex {
    pub const ATTRIBS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }

    /// Creates an unlit line/overlay vertex with zero normal.
    pub fn new(position: [f32; 3], color: [f32; 3]) -> Self {
        Self {
            position,
            normal: [0.0, 0.0, 0.0],
            color,
        }
    }

    /// Creates a fully specified 3D mesh vertex with surface normal.
    pub fn new_with_normal(position: [f32; 3], normal: [f32; 3], color: [f32; 3]) -> Self {
        Self {
            position,
            normal,
            color,
        }
    }

    /// Creates a vertex dot corner where `normal.xy` encodes the screen-pixel offset.
    pub fn new_dot_corner(position: [f32; 3], pixel_offset: [f32; 2], color: [f32; 3]) -> Self {
        Self {
            position,
            normal: [pixel_offset[0], pixel_offset[1], 0.0],
            color,
        }
    }
}

/// Camera uniform buffer matching `@group(0) @binding(0)` in `src/shader.wgsl`.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub viewport_size: [f32; 2],
    pub _padding: [f32; 2],
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self::new()
    }
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
            viewport_size: [800.0, 600.0],
            _padding: [0.0, 0.0],
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera, aspect: f32, width: f32, height: f32) {
        self.view_proj = camera.build_view_projection_matrix(aspect).into();
        self.viewport_size = [width.max(1.0), height.max(1.0)];
    }
}
