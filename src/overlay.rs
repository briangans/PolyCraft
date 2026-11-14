//! Viewport overlays: wireframe edges and component handles (square vertex dots).
//!
//! Generates line vertex buffers for wireframe edges (`PrimitiveTopology::LineList`) and
//! billboarded square dot buffers for vertices (`PrimitiveTopology::TriangleList`), styled after
//! Maya's viewport conventions.

use crate::edit_mode::EditMode;
use crate::mesh::HalfEdgeMesh;
use crate::selection::Selection;

pub const COLOR_WIREFRAME_DEFAULT: [f32; 3] = [0.15, 0.15, 0.16];
pub const COLOR_OBJECT_SELECTED: [f32; 3] = [0.26, 0.82, 0.38];
pub const COLOR_EDGE_SELECTED: [f32; 3] = [0.20, 0.85, 0.95];
pub const COLOR_VERTEX_DEFAULT: [f32; 3] = [0.25, 0.45, 0.90];
pub const COLOR_VERTEX_SELECTED: [f32; 3] = [1.0, 0.88, 0.15];

/// Half-size in screen pixels for vertex square dots.
pub const UNSELECTED_DOT_RADIUS_PIXELS: f32 = 3.0; // 6x6 pixel square dot
pub const SELECTED_DOT_RADIUS_PIXELS: f32 = 4.5; // 9x9 pixel square dot

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

    pub fn new_dot_corner(position: [f32; 3], pixel_offset: [f32; 2], color: [f32; 3]) -> Self {
        Self {
            position,
            normal: [pixel_offset[0], pixel_offset[1], 0.0],
            color,
        }
    }
}

pub struct OverlayBuffers {
    pub lines: Vec<OverlayVertex>,
    pub dots: Vec<OverlayVertex>,
}

/// Builds wireframe line segments and billboarded square vertex dots based on mode and selection.
pub fn build_overlay_buffers(
    mesh: &HalfEdgeMesh,
    mode: EditMode,
    selection: &Selection,
) -> OverlayBuffers {
    let mut lines = Vec::new();
    let mut dots = Vec::new();

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

        lines.push(OverlayVertex::new([p0.x, p0.y, p0.z], color));
        lines.push(OverlayVertex::new([p1.x, p1.y, p1.z], color));
    }

    // 2. Vertex markers (only in Vertex mode): screen-aligned flat square dots
    if mode == EditMode::Vertex {
        for (v_idx, mesh_vert) in mesh.vertices.iter().enumerate() {
            let v_id = crate::mesh::VertexId(v_idx);
            let p = [
                mesh_vert.position.x,
                mesh_vert.position.y,
                mesh_vert.position.z,
            ];
            let is_selected = selection.is_vertex_selected(v_id);
            let color = if is_selected {
                COLOR_VERTEX_SELECTED
            } else {
                COLOR_VERTEX_DEFAULT
            };
            let r = if is_selected {
                SELECTED_DOT_RADIUS_PIXELS
            } else {
                UNSELECTED_DOT_RADIUS_PIXELS
            };

            // Two triangles forming an upright square dot centered at p
            dots.push(OverlayVertex::new_dot_corner(p, [-r, -r], color));
            dots.push(OverlayVertex::new_dot_corner(p, [r, -r], color));
            dots.push(OverlayVertex::new_dot_corner(p, [r, r], color));

            dots.push(OverlayVertex::new_dot_corner(p, [-r, -r], color));
            dots.push(OverlayVertex::new_dot_corner(p, [r, r], color));
            dots.push(OverlayVertex::new_dot_corner(p, [-r, r], color));
        }
    }

    OverlayBuffers { lines, dots }
}

/// Converts transform-gizmo arms into line-list vertices, two per arm, each pair sharing its
/// axis colour.
///
/// The gizmo keeps its own GPU buffer rather than joining the wireframe one: arm length depends on
/// camera distance, so it is rewritten every frame, while the wireframe is rebuilt only when the
/// mesh or selection changes. Reusing `OverlayVertex` means the gizmo needs no new vertex layout.
pub fn gizmo_line_vertices(arms: &[crate::gizmo::Arm]) -> Vec<OverlayVertex> {
    let mut lines = Vec::with_capacity(arms.len() * 2);
    for arm in arms {
        lines.push(OverlayVertex::new(
            [arm.start.x, arm.start.y, arm.start.z],
            arm.color,
        ));
        lines.push(OverlayVertex::new(
            [arm.end.x, arm.end.y, arm.end.z],
            arm.color,
        ));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{EdgeId, HalfEdgeMesh, VertexId};

    #[test]
    fn cube_wireframe_in_object_mode_unselected() {
        let mesh = HalfEdgeMesh::create_cube();
        let selection = Selection::new();
        let buffers = build_overlay_buffers(&mesh, EditMode::Object, &selection);

        // 12 edges * 2 endpoints = 24 line vertices
        assert_eq!(buffers.lines.len(), 24);
        assert!(buffers
            .lines
            .iter()
            .all(|v| v.color == COLOR_WIREFRAME_DEFAULT));
        assert!(buffers.dots.is_empty());
    }

    #[test]
    fn cube_wireframe_in_object_mode_selected() {
        let mesh = HalfEdgeMesh::create_cube();
        let mut selection = Selection::new();
        selection.apply_object_click(true, false);
        let buffers = build_overlay_buffers(&mesh, EditMode::Object, &selection);

        assert_eq!(buffers.lines.len(), 24);
        assert!(buffers
            .lines
            .iter()
            .all(|v| v.color == COLOR_OBJECT_SELECTED));
        assert!(buffers.dots.is_empty());
    }

    #[test]
    fn edge_selection_highlights_only_selected_edge() {
        let mesh = HalfEdgeMesh::create_cube();
        let mut selection = Selection::new();
        let target_edge = EdgeId(0, 1);
        selection.apply_click(Some(target_edge), false, Selection::edges_mut);

        let buffers = build_overlay_buffers(&mesh, EditMode::Edge, &selection);
        assert_eq!(buffers.lines.len(), 24);
        assert!(buffers.dots.is_empty());

        let selected_count = buffers
            .lines
            .as_chunks::<2>()
            .0
            .iter()
            .filter(|chunk| {
                chunk[0].color == COLOR_EDGE_SELECTED && chunk[1].color == COLOR_EDGE_SELECTED
            })
            .count();
        assert_eq!(selected_count, 1, "exactly one edge should be highlighted");

        let default_count = buffers
            .lines
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
    fn vertex_mode_generates_wireframe_and_square_dots() {
        let mesh = HalfEdgeMesh::create_cube();
        let mut selection = Selection::new();
        let target_vertex = VertexId(0);
        selection.apply_click(Some(target_vertex), false, Selection::vertices_mut);

        let buffers = build_overlay_buffers(&mesh, EditMode::Vertex, &selection);

        // 24 edge line vertices
        assert_eq!(buffers.lines.len(), 24);

        // 8 cube vertices * 6 vertices per square dot = 48 dot vertices
        assert_eq!(buffers.dots.len(), 48);

        // 1 selected vertex = 6 vertices with COLOR_VERTEX_SELECTED
        let selected_dots = buffers
            .dots
            .iter()
            .filter(|v| v.color == COLOR_VERTEX_SELECTED)
            .count();
        assert_eq!(selected_dots, 6);

        // 7 unselected vertices = 42 vertices with COLOR_VERTEX_DEFAULT
        let unselected_dots = buffers
            .dots
            .iter()
            .filter(|v| v.color == COLOR_VERTEX_DEFAULT)
            .count();
        assert_eq!(unselected_dots, 42);
    }

    #[test]
    fn gizmo_line_vertices_converts_arms_into_unlit_line_pairs() {
        use cgmath::Point3;
        let arms = vec![
            crate::gizmo::Arm {
                start: Point3::new(0.0, 1.0, 2.0),
                end: Point3::new(3.0, 1.0, 2.0),
                color: [1.0, 0.0, 0.0],
            },
            crate::gizmo::Arm {
                start: Point3::new(0.0, 1.0, 2.0),
                end: Point3::new(0.0, 4.0, 2.0),
                color: [0.0, 1.0, 0.0],
            },
        ];

        let line_verts = gizmo_line_vertices(&arms);
        assert_eq!(line_verts.len(), 4, "2 vertices per arm");
        assert_eq!(line_verts[0].position, [0.0, 1.0, 2.0]);
        assert_eq!(line_verts[0].color, [1.0, 0.0, 0.0]);
        assert_eq!(line_verts[1].position, [3.0, 1.0, 2.0]);
        assert_eq!(line_verts[1].color, [1.0, 0.0, 0.0]);

        assert_eq!(line_verts[2].position, [0.0, 1.0, 2.0]);
        assert_eq!(line_verts[2].color, [0.0, 1.0, 0.0]);
        assert_eq!(line_verts[3].position, [0.0, 4.0, 2.0]);
        assert_eq!(line_verts[3].color, [0.0, 1.0, 0.0]);
    }
}
