//! Half-edge mesh: topology plus the flat triangle data the shaded pass consumes.
//!
//! Every editable component has a stable index newtype (`VertexId`, `HalfEdgeId`, `FaceId`).
//! Shared (undirected) edges are identified by [`EdgeId`], a normalized pair of vertex indices,
//! so both half-edges of an edge resolve to the same selectable component.

use cgmath::{InnerSpace, Point3, Vector3};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VertexId(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HalfEdgeId(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FaceId(pub usize);

/// An undirected edge, stored as its two vertex indices with the smaller one first. The two
/// half-edges that make up the edge both produce the same `EdgeId`, which makes it usable as a
/// key for selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId(pub usize, pub usize);

impl EdgeId {
    pub fn between(a: VertexId, b: VertexId) -> Self {
        if a.0 <= b.0 {
            Self(a.0, b.0)
        } else {
            Self(b.0, a.0)
        }
    }
}

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

/// One un-welded vertex of the shaded triangle soup.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

/// Flat triangle data for the shaded pass. `vertex_faces[i]` names the face that `vertices[i]`
/// came from, so the renderer can colour individual faces without the mesh knowing about colours.
pub struct RenderMesh {
    pub vertices: Vec<RenderVertex>,
    pub vertex_faces: Vec<FaceId>,
    pub indices: Vec<u16>,
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

    pub fn vertex_position(&self, vertex: VertexId) -> Point3<f32> {
        self.vertices[vertex.0].position
    }

    /// The vertex at the far end of a half-edge.
    pub fn half_edge_destination(&self, half_edge: HalfEdgeId) -> VertexId {
        let next = self.half_edges[half_edge.0]
            .next
            .expect("every half-edge is linked into a face loop");
        self.half_edges[next.0].origin
    }

    /// The undirected edge a half-edge belongs to.
    pub fn half_edge_edge(&self, half_edge: HalfEdgeId) -> EdgeId {
        EdgeId::between(
            self.half_edges[half_edge.0].origin,
            self.half_edge_destination(half_edge),
        )
    }

    /// The half-edges around a face, in winding order.
    pub fn face_half_edges(&self, face: FaceId) -> Vec<HalfEdgeId> {
        let mut result = Vec::new();
        let Some(start) = self.faces[face.0].half_edge else {
            return result;
        };

        let mut current = start;
        loop {
            result.push(current);
            let next = self.half_edges[current.0]
                .next
                .expect("every half-edge is linked into a face loop");
            if next == start {
                break;
            }
            current = next;
        }
        result
    }

    pub fn face_vertices(&self, face: FaceId) -> Vec<VertexId> {
        self.face_half_edges(face)
            .into_iter()
            .map(|half_edge| self.half_edges[half_edge.0].origin)
            .collect()
    }

    pub fn face_positions(&self, face: FaceId) -> Vec<Point3<f32>> {
        self.face_vertices(face)
            .into_iter()
            .map(|vertex| self.vertex_position(vertex))
            .collect()
    }

    /// Newell's method: a stable polygon normal that uses every vertex, so concave and
    /// non-planar faces still come out right (the old first-three-vertices version did not).
    pub fn face_normal(&self, face: FaceId) -> Vector3<f32> {
        let positions = self.face_positions(face);
        let mut normal = Vector3::new(0.0, 0.0, 0.0);

        for i in 0..positions.len() {
            let current = positions[i];
            let next = positions[(i + 1) % positions.len()];
            normal.x += (current.y - next.y) * (current.z + next.z);
            normal.y += (current.z - next.z) * (current.x + next.x);
            normal.z += (current.x - next.x) * (current.y + next.y);
        }

        if normal.magnitude2() > 0.0 {
            normal.normalize()
        } else {
            normal
        }
    }

    pub fn face_center(&self, face: FaceId) -> Point3<f32> {
        let positions = self.face_positions(face);
        let count = positions.len() as f32;
        let sum = positions
            .iter()
            .fold(Vector3::new(0.0, 0.0, 0.0), |acc, p| {
                acc + Vector3::new(p.x, p.y, p.z)
            });
        Point3::new(sum.x / count, sum.y / count, sum.z / count)
    }

    /// Every undirected edge once, paired with a half-edge that represents it.
    pub fn edges(&self) -> Vec<(EdgeId, HalfEdgeId)> {
        let mut seen = HashSet::new();
        let mut result = Vec::new();

        for index in 0..self.half_edges.len() {
            let half_edge = HalfEdgeId(index);
            let edge = self.half_edge_edge(half_edge);
            if seen.insert(edge) {
                result.push((edge, half_edge));
            }
        }

        result
    }

    /// The faces sharing an edge: `(its own face, the twin's face)`. A boundary edge has no
    /// second face.
    pub fn edge_faces(&self, edge: EdgeId) -> (Option<FaceId>, Option<FaceId>) {
        for index in 0..self.half_edges.len() {
            let half_edge = HalfEdgeId(index);
            if self.half_edge_edge(half_edge) != edge {
                continue;
            }

            let own = self.half_edges[half_edge.0].face;
            let twin = self.half_edges[half_edge.0]
                .twin
                .and_then(|twin| self.half_edges[twin.0].face);
            return (own, twin);
        }

        (None, None)
    }

    /// Converts the mesh into flat buffers for the shaded pass. Vertices are un-welded per face
    /// (flat shading), and `vertex_faces` records which face each one came from.
    pub fn generate_render_buffers(&self) -> RenderMesh {
        let mut render_mesh = RenderMesh {
            vertices: Vec::new(),
            vertex_faces: Vec::new(),
            indices: Vec::new(),
        };

        for face_index in 0..self.faces.len() {
            let face = FaceId(face_index);
            let positions = self.face_positions(face);
            if positions.len() < 3 {
                continue;
            }

            let normal = self.face_normal(face);
            let normal_array = [normal.x, normal.y, normal.z];
            let start_index = render_mesh.vertices.len() as u16;

            for position in &positions {
                render_mesh.vertices.push(RenderVertex {
                    position: [position.x, position.y, position.z],
                    normal: normal_array,
                });
                render_mesh.vertex_faces.push(face);
            }

            // Triangle fan
            for i in 1..(positions.len() - 1) as u16 {
                render_mesh.indices.push(start_index);
                render_mesh.indices.push(start_index + i);
                render_mesh.indices.push(start_index + i + 1);
            }
        }

        render_mesh
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Vertex positions of the unit cube, for comparing against face normals.
    fn cube() -> HalfEdgeMesh {
        HalfEdgeMesh::create_cube()
    }

    #[test]
    fn cube_topology_counts() {
        let mesh = cube();
        assert_eq!(mesh.vertices.len(), 8);
        assert_eq!(mesh.faces.len(), 6);
        assert_eq!(mesh.half_edges.len(), 24, "4 half-edges per quad face");
        assert_eq!(mesh.edges().len(), 12, "a cube has 12 unique edges");
    }

    #[test]
    fn every_interior_edge_has_a_twin() {
        let mesh = cube();
        assert!(
            mesh.half_edges.iter().all(|edge| edge.twin.is_some()),
            "a closed cube has no boundary edges"
        );
    }

    #[test]
    fn edge_id_is_order_independent() {
        let forward = EdgeId::between(VertexId(2), VertexId(5));
        let backward = EdgeId::between(VertexId(5), VertexId(2));
        assert_eq!(forward, backward);
        assert_eq!(forward, EdgeId(2, 5));
    }

    #[test]
    fn both_half_edges_of_an_edge_produce_the_same_edge_id() {
        let mesh = cube();
        for (index, half_edge) in mesh.half_edges.iter().enumerate() {
            let Some(twin) = half_edge.twin else { continue };
            assert_eq!(
                mesh.half_edge_edge(HalfEdgeId(index)),
                mesh.half_edge_edge(twin),
                "an edge and its twin must normalise to one EdgeId"
            );
        }
    }

    #[test]
    fn face_normals_point_outward_and_are_unit_length() {
        let mesh = cube();
        for face_index in 0..mesh.faces.len() {
            let face = FaceId(face_index);
            let normal = mesh.face_normal(face);
            let center = mesh.face_center(face);

            assert!(
                (normal.magnitude() - 1.0).abs() < 1e-5,
                "face {face_index} normal is not unit length"
            );
            // For the cube the outward direction is the face center itself.
            let outward = Vector3::new(center.x, center.y, center.z);
            assert!(
                normal.dot(outward) > 0.0,
                "face {face_index} normal points inward"
            );
        }
    }

    #[test]
    fn newell_normal_agrees_with_cross_product_on_a_planar_quad() {
        let positions = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(2.0, 3.0, 0.0),
            Point3::new(0.0, 3.0, 0.0),
        ];
        let mesh = HalfEdgeMesh::from_faces(&positions, &[vec![0, 1, 2, 3]]);
        let normal = mesh.face_normal(FaceId(0));
        assert!((normal - Vector3::new(0.0, 0.0, 1.0)).magnitude() < 1e-5);
    }

    #[test]
    fn concave_face_normal_ignores_vertex_order_noise() {
        // An arrow-head polygon: the first three vertices alone would give the wrong direction.
        let positions = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0), // the reflex notch
            Point3::new(2.0, 2.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
        ];
        let mesh = HalfEdgeMesh::from_faces(&positions, &[vec![0, 1, 2, 3, 4]]);
        assert!((mesh.face_normal(FaceId(0)) - Vector3::new(0.0, 0.0, 1.0)).magnitude() < 1e-5);
    }

    #[test]
    fn face_loops_are_closed_and_ordered() {
        let mesh = cube();
        for face_index in 0..mesh.faces.len() {
            let face = FaceId(face_index);
            let half_edges = mesh.face_half_edges(face);
            assert_eq!(half_edges.len(), 4);

            // Walking `next` from any of them must return to the start.
            let start = half_edges[0];
            let mut current = start;
            let mut steps = 0;
            loop {
                current = mesh.half_edges[current.0].next.unwrap();
                steps += 1;
                if current == start {
                    break;
                }
                assert!(steps <= 4, "face loop did not close");
            }
        }
    }

    #[test]
    fn edge_faces_names_both_neighbours() {
        let mesh = cube();
        for (edge, _) in mesh.edges() {
            let (a, b) = mesh.edge_faces(edge);
            assert!(a.is_some() && b.is_some());
            assert_ne!(a, b, "an edge must separate two distinct faces");
        }
    }

    #[test]
    fn render_mesh_covers_every_face_with_two_triangles() {
        let mesh = cube();
        let render_mesh = mesh.generate_render_buffers();
        assert_eq!(
            render_mesh.vertices.len(),
            6 * 4,
            "four un-welded verts per quad"
        );
        assert_eq!(render_mesh.vertex_faces.len(), render_mesh.vertices.len());
        assert_eq!(
            render_mesh.indices.len(),
            6 * 2 * 3,
            "two triangles per quad"
        );
    }

    #[test]
    fn face_center_averages_its_vertices() {
        let mesh = cube();
        for face_index in 0..mesh.faces.len() {
            let face = FaceId(face_index);
            let center = mesh.face_center(face);
            // Every cube face is centred on a half-unit axis.
            let axis_aligned = (center.x.abs() - 0.5).abs() < 1e-5
                || (center.y.abs() - 0.5).abs() < 1e-5
                || (center.z.abs() - 0.5).abs() < 1e-5;
            assert!(axis_aligned);
        }
    }
}
