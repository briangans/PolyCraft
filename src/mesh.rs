use cgmath::{InnerSpace, Point3}; // <-- Ensure InnerSpace is imported
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VertexId(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HalfEdgeId(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FaceId(pub usize);

pub struct MeshVertex {
    pub position: Point3<f32>,
    pub half_edge: Option<HalfEdgeId>,
}

pub struct HalfEdge {
    pub origin: VertexId,
    pub twin: Option<HalfEdgeId>,
    pub next: Option<HalfEdgeId>,
    pub prev: Option<HalfEdgeId>,
    pub face: Option<FaceId>,
}

pub struct Face {
    pub half_edge: Option<HalfEdgeId>,
}

pub struct HalfEdgeMesh {
    pub vertices: Vec<MeshVertex>,
    pub half_edges: Vec<HalfEdge>,
    pub faces: Vec<Face>,
}

impl HalfEdgeMesh {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            half_edges: Vec::new(),
            faces: Vec::new(),
        }
    }

    /// Automatically constructs a Half-Edge graph from raw vertex positions and face index arrays
    pub fn from_faces(vertex_positions: &[Point3<f32>], face_indices: &[Vec<usize>]) -> Self {
        let mut mesh = Self::new();

        // 1. Create all vertices
        for pos in vertex_positions {
            mesh.vertices.push(MeshVertex {
                position: *pos,
                half_edge: None,
            });
        }

        // A map to find twins. Key is (Origin, Destination), Value is HalfEdgeId
        let mut edge_map: HashMap<(usize, usize), HalfEdgeId> = HashMap::new();

        // 2. Build Faces and Half-Edges
        for (f_idx, vertex_loop) in face_indices.iter().enumerate() {
            let face_id = FaceId(f_idx);
            mesh.faces.push(Face { half_edge: None });

            let mut face_edges = Vec::new();

            // Create all half-edges for this face
            for i in 0..vertex_loop.len() {
                let v_origin = vertex_loop[i];
                let v_dest = vertex_loop[(i + 1) % vertex_loop.len()]; // Wrap around to the start

                let edge_id = HalfEdgeId(mesh.half_edges.len());
                face_edges.push(edge_id);

                mesh.half_edges.push(HalfEdge {
                    origin: VertexId(v_origin),
                    twin: None,
                    next: None,
                    prev: None,
                    face: Some(face_id),
                });

                // Link the vertex to this outgoing edge
                mesh.vertices[v_origin].half_edge = Some(edge_id);

                // Check if our twin already exists in the map (from a neighboring face)
                if let Some(&twin_id) = edge_map.get(&(v_dest, v_origin)) {
                    mesh.half_edges[edge_id.0].twin = Some(twin_id);
                    mesh.half_edges[twin_id.0].twin = Some(edge_id);
                } else {
                    // Register ourselves in the map for future neighbors to find
                    edge_map.insert((v_origin, v_dest), edge_id);
                }
            }

            // 3. Link next and prev within the face loop
            mesh.faces[face_id.0].half_edge = Some(face_edges[0]);
            let edge_count = face_edges.len();
            for i in 0..edge_count {
                let curr = face_edges[i];
                let next = face_edges[(i + 1) % edge_count];
                let prev = face_edges[(i + edge_count - 1) % edge_count];

                mesh.half_edges[curr.0].next = Some(next);
                mesh.half_edges[curr.0].prev = Some(prev);
            }
        }

        mesh
    }

    /// Generates a procedural 1x1x1 cube using the from_faces builder
    pub fn create_cube() -> Self {
        let positions = vec![
            Point3::new(-0.5, -0.5, 0.5),  // 0: Front-Bottom-Left
            Point3::new(0.5, -0.5, 0.5),   // 1: Front-Bottom-Right
            Point3::new(0.5, 0.5, 0.5),    // 2: Front-Top-Right
            Point3::new(-0.5, 0.5, 0.5),   // 3: Front-Top-Left
            Point3::new(-0.5, -0.5, -0.5), // 4: Back-Bottom-Left
            Point3::new(0.5, -0.5, -0.5),  // 5: Back-Bottom-Right
            Point3::new(0.5, 0.5, -0.5),   // 6: Back-Top-Right
            Point3::new(-0.5, 0.5, -0.5),  // 7: Back-Top-Left
        ];

        let faces = vec![
            vec![0, 1, 2, 3], // Front
            vec![1, 5, 6, 2], // Right
            vec![5, 4, 7, 6], // Back
            vec![4, 0, 3, 7], // Left
            vec![3, 2, 6, 7], // Top
            vec![4, 5, 1, 0], // Bottom
        ];

        Self::from_faces(&positions, &faces)
    }

    /// Converts the Half-Edge mesh into flat buffers for wgpu rendering
    pub fn generate_render_buffers(&self) -> (Vec<([f32; 3], [f32; 3])>, Vec<u16>) {
        let mut render_verts = Vec::new();
        let mut render_indices = Vec::new();
        let mut current_vertex_index = 0;

        for face in &self.faces {
            if let Some(start_edge_id) = face.half_edge {
                let mut current_edge_id = start_edge_id;
                let mut face_positions = Vec::new();

                loop {
                    let edge = &self.half_edges[current_edge_id.0];
                    face_positions.push(self.vertices[edge.origin.0].position);
                    current_edge_id = edge.next.unwrap();
                    if current_edge_id == start_edge_id {
                        break;
                    }
                }

                if face_positions.len() >= 3 {
                    // 1. Calculate Face Normal using the first three vertices
                    let v0 = face_positions[0];
                    let v1 = face_positions[1];
                    let v2 = face_positions[2];

                    let normal = (v1 - v0).cross(v2 - v0).normalize();
                    let n_array = [normal.x, normal.y, normal.z];

                    // 2. Un-weld vertices for flat shading
                    let start_idx = current_vertex_index;
                    for pos in &face_positions {
                        render_verts.push(([pos.x, pos.y, pos.z], n_array));
                        current_vertex_index += 1;
                    }

                    // 3. Triangle Fan
                    for i in 1..(face_positions.len() - 1) as u16 {
                        render_indices.push(start_idx);
                        render_indices.push(start_idx + i);
                        render_indices.push(start_idx + i + 1);
                    }
                }
            }
        }

        (render_verts, render_indices)
    }
}
