//! Viewport overlays: wireframe edges and component handles (vertices/edges/object).
//!
//! Generates line vertex buffers for `PrimitiveTopology::LineList` rendering, styled after
//! Maya's viewport conventions.

use crate::edit_mode::EditMode;
use crate::mesh::HalfEdgeMesh;
use crate::selection::Selection;

pub const COLOR_WIREFRAME_DEFAULT: [f32; 3] = [0.15, 0.15, 0.16];
pub const COLOR_OBJECT_SELECTED: [f32; 3] = [0.26, 0.82, 0.38];
pub const COLOR_EDGE_SELECTED: [f32; 3] = [0.20, 0.85, 0.95];
pub const COLOR_VERTEX_DEFAULT: [f32; 3] = [0.35, 0.50, 0.90];
pub const COLOR_VERTEX_SELECTED: [f32; 3] = [1.0, 0.90, 0.15];

pub const VERTEX_MARKER_RADIUS: f32 = 0.035;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OverlayVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

impl OverlayVertex {
    pub fn new(position: [f32; 3], color: [f32; 3]) -> Self {
        Self {
            position,
            normal: [0.0, 0.0, 0.0],
            color,
        }
    }
}

/// Builds line vertices (pairs of endpoints) for wireframe edges and vertex handles.
pub fn build_overlay_vertices(
    mesh: &HalfEdgeMesh,
    mode: EditMode,
    selection: &Selection,
) -> Vec<OverlayVertex> {
    let mut vertices = Vec::new();

    // 1. Wireframe edges
    let is_object_selected = mode == EditMode::Object && selection.is_object_selected();
    let default_wire_color = if is_object_selected {
        COLOR_OBJECT_SELECTED
    } else {
        COLOR_WIREFRAME_DEFAULT
    };

    for (edge_id, half_edge_id) in mesh.edges() {
        let v0 = mesh.half_edges[half_edge_id.0].origin;
        let v1 = mesh.half_edge_destination(half_edge_id);
        let p0 = mesh.vertex_position(v0);
        let p1 = mesh.vertex_position(v1);

        let color = if mode == EditMode::Edge && selection.is_edge_selected(edge_id) {
            COLOR_EDGE_SELECTED
        } else {
            default_wire_color
        };

        vertices.push(OverlayVertex::new([p0.x, p0.y, p0.z], color));
        vertices.push(OverlayVertex::new([p1.x, p1.y, p1.z], color));
    }

    // 2. Vertex markers (only in Vertex mode)
    if mode == EditMode::Vertex {
        let r = VERTEX_MARKER_RADIUS;
        for (v_idx, mesh_vert) in mesh.vertices.iter().enumerate() {
            let v_id = crate::mesh::VertexId(v_idx);
            let p = mesh_vert.position;
            let is_selected = selection.is_vertex_selected(v_id);
            let color = if is_selected {
                COLOR_VERTEX_SELECTED
            } else {
                COLOR_VERTEX_DEFAULT
            };

            // 3-axis cross
            vertices.push(OverlayVertex::new([p.x - r, p.y, p.z], color));
            vertices.push(OverlayVertex::new([p.x + r, p.y, p.z], color));

            vertices.push(OverlayVertex::new([p.x, p.y - r, p.z], color));
            vertices.push(OverlayVertex::new([p.x, p.y + r, p.z], color));

            vertices.push(OverlayVertex::new([p.x, p.y, p.z - r], color));
            vertices.push(OverlayVertex::new([p.x, p.y, p.z + r], color));

            // For selected vertices, also add a 3D diamond (octahedron) outline
            if is_selected {
                let px = [p.x + r, p.y, p.z];
                let nx = [p.x - r, p.y, p.z];
                let py = [p.x, p.y + r, p.z];
                let ny = [p.x, p.y - r, p.z];
                let pz = [p.x, p.y, p.z + r];
                let nz = [p.x, p.y, p.z - r];

                // Middle ring (XY plane)
                vertices.push(OverlayVertex::new(px, color));
                vertices.push(OverlayVertex::new(py, color));
                vertices.push(OverlayVertex::new(py, color));
                vertices.push(OverlayVertex::new(nx, color));
                vertices.push(OverlayVertex::new(nx, color));
                vertices.push(OverlayVertex::new(ny, color));
                vertices.push(OverlayVertex::new(ny, color));
                vertices.push(OverlayVertex::new(px, color));

                // Pyramids to +Z and -Z
                vertices.push(OverlayVertex::new(px, color));
                vertices.push(OverlayVertex::new(pz, color));
                vertices.push(OverlayVertex::new(nx, color));
                vertices.push(OverlayVertex::new(pz, color));
                vertices.push(OverlayVertex::new(py, color));
                vertices.push(OverlayVertex::new(pz, color));
                vertices.push(OverlayVertex::new(ny, color));
                vertices.push(OverlayVertex::new(pz, color));

                vertices.push(OverlayVertex::new(px, color));
                vertices.push(OverlayVertex::new(nz, color));
                vertices.push(OverlayVertex::new(nx, color));
                vertices.push(OverlayVertex::new(nz, color));
                vertices.push(OverlayVertex::new(py, color));
                vertices.push(OverlayVertex::new(nz, color));
                vertices.push(OverlayVertex::new(ny, color));
                vertices.push(OverlayVertex::new(nz, color));
            }
        }
    }

    vertices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{EdgeId, HalfEdgeMesh, VertexId};

    #[test]
    fn cube_wireframe_in_object_mode_unselected() {
        let mesh = HalfEdgeMesh::create_cube();
        let selection = Selection::new();
        let verts = build_overlay_vertices(&mesh, EditMode::Object, &selection);

        // 12 edges * 2 endpoints = 24 line vertices
        assert_eq!(verts.len(), 24);
        assert!(verts.iter().all(|v| v.color == COLOR_WIREFRAME_DEFAULT));
    }

    #[test]
    fn cube_wireframe_in_object_mode_selected() {
        let mesh = HalfEdgeMesh::create_cube();
        let mut selection = Selection::new();
        selection.apply_object_click(true, false);
        let verts = build_overlay_vertices(&mesh, EditMode::Object, &selection);

        assert_eq!(verts.len(), 24);
        assert!(verts.iter().all(|v| v.color == COLOR_OBJECT_SELECTED));
    }

    #[test]
    fn edge_selection_highlights_only_selected_edge() {
        let mesh = HalfEdgeMesh::create_cube();
        let mut selection = Selection::new();
        let target_edge = EdgeId(0, 1);
        selection.apply_click(Some(target_edge), false, Selection::edges_mut);

        let verts = build_overlay_vertices(&mesh, EditMode::Edge, &selection);
        assert_eq!(verts.len(), 24);

        let selected_count = verts
            .as_chunks::<2>()
            .0
            .iter()
            .filter(|chunk| {
                chunk[0].color == COLOR_EDGE_SELECTED && chunk[1].color == COLOR_EDGE_SELECTED
            })
            .count();
        assert_eq!(selected_count, 1, "exactly one edge should be highlighted");

        let default_count = verts
            .as_chunks::<2>()
            .0
            .iter()
            .filter(|chunk| {
                chunk[0].color == COLOR_WIREFRAME_DEFAULT
                    && chunk[1].color == COLOR_WIREFRAME_DEFAULT
            })
            .count();
        assert_eq!(
            default_count, 11,
            "other 11 edges should use default wireframe color"
        );
    }

    #[test]
    fn vertex_mode_generates_wireframe_and_vertex_markers() {
        let mesh = HalfEdgeMesh::create_cube();
        let mut selection = Selection::new();
        let target_vertex = VertexId(0);
        selection.apply_click(Some(target_vertex), false, Selection::vertices_mut);

        let verts = build_overlay_vertices(&mesh, EditMode::Vertex, &selection);

        // 24 edge vertices
        // + 7 unselected vertices * 6 vertices per cross = 42
        // + 1 selected vertex * (6 cross + 24 diamond = 30) = 30
        // Total = 24 + 42 + 30 = 96
        assert_eq!(verts.len(), 96);

        let selected_vertex_colored = verts
            .iter()
            .filter(|v| v.color == COLOR_VERTEX_SELECTED)
            .count();
        assert_eq!(selected_vertex_colored, 30);
    }
}
