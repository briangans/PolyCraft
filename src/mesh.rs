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

/// Document records are snapshot-able so `HistoryStack` can version them (see `scene::DocumentSnapshot`).
#[derive(Clone, Debug, PartialEq)]
pub struct MeshVertex {
    pub position: Point3<f32>,
    pub half_edge: Option<HalfEdgeId>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HalfEdge {
    pub origin: VertexId,
    pub twin: Option<HalfEdgeId>,
    pub next: Option<HalfEdgeId>,
    pub prev: Option<HalfEdgeId>,
    pub face: Option<FaceId>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Face {
    pub half_edge: Option<HalfEdgeId>,
}

/// Every topology violation [`HalfEdgeMesh::validate`] found, one human-readable line each.
///
/// A list rather than a single reason because a mislinked edit virtually always breaks several
/// invariants at once: reporting only the first sends the reader back for a second run after every
/// fix. `Vec<String>` rather than a typed error because the text is the product - it goes straight
/// into a panic message or a test failure, and no caller branches on the kind of breakage.
pub type TopologyErrors = Vec<String>;

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
    /// Triangle indices into `vertices`. `u32`, not `u16`: vertices are un-welded per face, so a
    /// modest model passes 65,536 of them and a `u16` index would silently wrap onto the wrong
    /// vertices rather than fail.
    pub indices: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HalfEdgeMesh {
    pub vertices: Vec<MeshVertex>,
    pub half_edges: Vec<HalfEdge>,
    pub faces: Vec<Face>,
}

impl Default for HalfEdgeMesh {
    fn default() -> Self {
        Self::new()
    }
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
            // Taken before the fan's vertices are pushed, so it is their first index.
            let start_index = render_mesh.vertices.len() as u32;

            for position in &positions {
                render_mesh.vertices.push(RenderVertex {
                    position: [position.x, position.y, position.z],
                    normal: normal_array,
                });
                render_mesh.vertex_faces.push(face);
            }

            // Triangle fan
            for i in 1..(positions.len() - 1) {
                let follower = start_index + i as u32;
                render_mesh.indices.push(start_index);
                render_mesh.indices.push(follower);
                render_mesh.indices.push(follower + 1);
            }
        }

        render_mesh
    }

    /// Reports every way this mesh violates the invariants the whole codebase assumes, or `Ok(())`.
    ///
    /// This exists because nothing else catches a mislinked edit. A wrong `twin` pointer compiles,
    /// renders almost convincingly, and then panics inside `half_edge_destination` or
    /// `face_half_edges` - or, worse, does neither and leaves `EdgeId` naming two different edges.
    /// Every edit operation added from here on should build a candidate mesh, `validate` it, and only
    /// then hand it to the document, so a bad extrude fails against a named invariant instead of
    /// becoming a corrupted viewport.
    ///
    /// It is a `Result`, not a panic or a `bool`, so a caller can *decline* a bad operation and keep
    /// the document it already had. An empty `Err` list is never returned. Cost is linear in
    /// half-edges; it is meant for after an edit and before a file import, not once per frame.
    pub fn validate(&self) -> Result<(), TopologyErrors> {
        let mut scan = TopologyScan::new(self);
        scan.check_vertices();
        scan.check_links_are_inverses();
        scan.check_twins();
        scan.check_face_loops();
        scan.check_face_back_references();
        scan.check_directed_edges_are_unique();
        scan.finish()
    }
}

/// Scratch state for [`HalfEdgeMesh::validate`], one pass per method.
///
/// The optional ids are resolved into plain index arrays once, up front, for two reasons. A mesh that
/// names a half-edge which does not exist has to be *reported*, not panicked on while describing
/// itself, so no pass below dereferences a raw id. And each pass then reads as the invariant it
/// checks rather than as bounds checking.
struct TopologyScan<'a> {
    mesh: &'a HalfEdgeMesh,
    next_of: Vec<Option<usize>>,
    prev_of: Vec<Option<usize>>,
    twin_of: Vec<Option<usize>>,
    face_of: Vec<Option<usize>>,
    errors: Vec<String>,
}

impl<'a> TopologyScan<'a> {
    fn new(mesh: &'a HalfEdgeMesh) -> Self {
        let edge_count = mesh.half_edges.len();
        let mut scan = Self {
            mesh,
            next_of: vec![None; edge_count],
            prev_of: vec![None; edge_count],
            twin_of: vec![None; edge_count],
            face_of: vec![None; edge_count],
            errors: Vec::new(),
        };
        scan.resolve_ids();
        scan
    }

    fn finish(self) -> Result<(), TopologyErrors> {
        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(self.errors)
        }
    }

    /// The vertex a half-edge points at, or `None` when its `next` link is missing or dangling. In a
    /// valid mesh the successor's origin is this edge's destination.
    fn destination(&self, half_edge: usize) -> Option<usize> {
        self.next_of[half_edge].map(|next| self.mesh.half_edges[next].origin.0)
    }

    /// Resolves every id the mesh records, reporting the ones that name nothing. After this runs the
    /// `*_of` arrays only ever hold indices that exist, which is what keeps the passes below total.
    fn resolve_ids(&mut self) {
        let edge_count = self.mesh.half_edges.len();
        let face_count = self.mesh.faces.len();

        for (index, edge) in self.mesh.half_edges.iter().enumerate() {
            if edge.origin.0 >= self.mesh.vertices.len() {
                self.errors.push(format!(
                    "half edge {index} starts at vertex {}, which does not exist",
                    edge.origin.0
                ));
            }

            self.next_of[index] =
                Self::resolve_link(edge.next, edge_count, "next", index, &mut self.errors);
            self.prev_of[index] =
                Self::resolve_link(edge.prev, edge_count, "prev", index, &mut self.errors);

            // A twin may legitimately be absent: that is a boundary edge, which an open mesh or an
            // extruded hole is expected to carry. Only a dangling or self-referential one is a fault.
            if let Some(twin) = edge.twin {
                if twin.0 >= edge_count {
                    self.errors.push(format!(
                        "half edge {index} is twinned with half edge {}, which does not exist",
                        twin.0
                    ));
                } else if twin.0 == index {
                    self.errors
                        .push(format!("half edge {index} is twinned with itself"));
                } else {
                    self.twin_of[index] = Some(twin.0);
                }
            }

            match edge.face {
                Some(face) if face.0 < face_count => self.face_of[index] = Some(face.0),
                Some(face) => self.errors.push(format!(
                    "half edge {index} belongs to face {}, which does not exist",
                    face.0
                )),
                None => self.errors.push(format!(
                    "half edge {index} belongs to no face, so it is in no face loop"
                )),
            }
        }
    }

    /// A `next` or `prev` link. Every half-edge of a valid mesh sits inside a closed face loop, so
    /// either missing or dangling is always a fault.
    fn resolve_link(
        link: Option<HalfEdgeId>,
        limit: usize,
        field: &str,
        index: usize,
        errors: &mut Vec<String>,
    ) -> Option<usize> {
        match link {
            Some(next) if next.0 < limit => Some(next.0),
            Some(next) => {
                errors.push(format!(
                    "half edge {index} has {field} half edge {}, which does not exist",
                    next.0
                ));
                None
            }
            None => {
                errors.push(format!("half edge {index} has no {field} half edge"));
                None
            }
        }
    }

    /// A vertex's `half_edge` must start at that vertex, and every vertex must start at least one
    /// half-edge. An edit that appends a vertex and forgets to link it would otherwise linger as an
    /// orphan that every later undo snapshot faithfully preserves.
    fn check_vertices(&mut self) {
        let mut starts_one = vec![false; self.mesh.vertices.len()];
        for edge in self.mesh.half_edges.iter() {
            if let Some(slot) = starts_one.get_mut(edge.origin.0) {
                *slot = true;
            }
        }

        for (index, vertex) in self.mesh.vertices.iter().enumerate() {
            if !starts_one[index] {
                self.errors.push(format!(
                    "vertex {index} starts no half edge, so no face can reach it"
                ));
            }

            match vertex.half_edge {
                Some(outgoing) => match self.mesh.half_edges.get(outgoing.0) {
                    Some(edge) if edge.origin.0 == index => {}
                    Some(edge) => self.errors.push(format!(
                        "vertex {index} is walked from half edge {}, which starts at vertex {}",
                        outgoing.0, edge.origin.0
                    )),
                    None => self.errors.push(format!(
                        "vertex {index} is walked from half edge {}, which does not exist",
                        outgoing.0
                    )),
                },
                // An outgoing edge is only missing on a vertex nothing has claimed, reported above.
                None if starts_one[index] => self.errors.push(format!(
                    "vertex {index} starts a half edge but names none, so it cannot be walked out from"
                )),
                None => {}
            }
        }
    }

    /// `next` and `prev` must be exact inverses. If they drift, walking a face forward and walking it
    /// backward disagree about which half-edges are adjacent, and any edit that rewires a neighbour's
    /// `prev` while inserting an edge leaves a loop that half-edge queries traverse one way and the
    /// overlay generator the other.
    fn check_links_are_inverses(&mut self) {
        for index in 0..self.next_of.len() {
            if let Some(next) = self.next_of[index] {
                if self.prev_of[next] != Some(index) {
                    self.errors.push(format!(
                        "half edge {index} is followed by half edge {next}, which does not name it as its predecessor"
                    ));
                }
            }
            if let Some(prev) = self.prev_of[index] {
                if self.next_of[prev] != Some(index) {
                    self.errors.push(format!(
                        "half edge {index} follows half edge {prev}, which does not name it as its successor"
                    ));
                }
            }
        }
    }

    /// Twinning is symmetric, and the pair runs opposite directions between the same two vertices.
    /// [`EdgeId`] is a bare vertex pair derived from a half-edge agreeing with its twin on that pair,
    /// so an asymmetric or reversed twin makes one edge mean two different things depending on which
    /// side of it the cursor landed on - selection and `edge_faces` would silently pick a side.
    fn check_twins(&mut self) {
        for (index, edge) in self.mesh.half_edges.iter().enumerate() {
            // A missing twin is a boundary edge, which is legal.
            let Some(twin) = self.twin_of[index] else {
                continue;
            };

            if self.twin_of[twin] != Some(index) {
                self.errors.push(format!(
                    "half edge {index} is twinned with half edge {twin}, which does not name it back"
                ));
            }

            let origin = edge.origin.0;
            if let (Some(destination), Some(twin_destination)) =
                (self.destination(index), self.destination(twin))
            {
                if self.mesh.half_edges[twin].origin.0 != destination || twin_destination != origin
                {
                    self.errors.push(format!(
                        "half edge {index} runs vertex {origin} to vertex {destination}, but its twin does not run that pair back"
                    ));
                }
                if let (Some(own_face), Some(twin_face)) = (self.face_of[index], self.face_of[twin])
                {
                    if own_face == twin_face {
                        self.errors.push(format!(
                            "half edge {index} and its twin bound face {own_face}, so that face touches itself"
                        ));
                    }
                }
            }
        }
    }

    /// Every half-edge sits in a `next` loop that closes on itself, and every half-edge of that loop
    /// names one face. This is the shape `face_half_edges` assumes when it walks until the id repeats:
    /// a loop that never returns makes it spin, and one that drifts into a neighbour's ring makes it
    /// return the wrong face's corners - which is what `generate_render_buffers` then triangulates.
    fn check_face_loops(&mut self) {
        let edge_count = self.next_of.len();
        let mut visited = vec![false; edge_count];
        let mut loops_per_face: HashMap<usize, usize> = HashMap::new();

        for start in 0..edge_count {
            if visited[start] {
                continue;
            }

            let mut loop_edges = Vec::new();
            let mut cursor = start;
            let mut closed = false;
            // Bounded by the edge count: a loop that has not closed after visiting every half-edge
            // never will, and an unbounded walk here would hang the report it is meant to emit.
            for _ in 0..=edge_count {
                visited[cursor] = true;
                loop_edges.push(cursor);
                match self.next_of[cursor] {
                    Some(next) if next == start => {
                        closed = true;
                        break;
                    }
                    Some(next) if !visited[next] => cursor = next,
                    // Either merged into a loop already walked, or ran out: neither can close here.
                    Some(_) | None => break,
                }
            }

            if !closed {
                self.errors.push(format!(
                    "following `next` from half edge {start} never returns to it, so these {} half edges form no face loop",
                    loop_edges.len()
                ));
            }

            let faces: HashSet<usize> = loop_edges
                .iter()
                .filter_map(|edge| self.face_of[*edge])
                .collect();
            if faces.len() > 1 {
                self.errors.push(format!(
                    "the loop from half edge {start} spans {} faces, but one loop belongs to exactly one face",
                    faces.len()
                ));
            }

            if let Some(&face) = faces.iter().next() {
                *loops_per_face.entry(face).or_insert(0) += 1;

                let corners: HashSet<usize> = loop_edges
                    .iter()
                    .map(|edge| self.mesh.half_edges[*edge].origin.0)
                    .collect();
                if corners.len() < 3 {
                    self.errors.push(format!(
                        "face {face} has {} distinct corners, which bounds nothing",
                        corners.len()
                    ));
                }
            }
        }

        for (face, loops) in loops_per_face {
            if loops > 1 {
                self.errors.push(format!(
                    "face {face} is walked as {loops} separate loops, so all but one would go unrendered"
                ));
            }
        }
    }

    /// Each face names one of its own half-edges, and that half-edge names the face back. Without the
    /// back-reference `face_half_edges` walks a ring belonging to another face, and every consumer -
    /// normals, centers, the render fan - quietly describes the wrong face.
    fn check_face_back_references(&mut self) {
        for (index, face) in self.mesh.faces.iter().enumerate() {
            let Some(half_edge) = face.half_edge else {
                self.errors.push(format!("face {index} names no half edge"));
                continue;
            };
            let Some(edge) = self.mesh.half_edges.get(half_edge.0) else {
                self.errors.push(format!(
                    "face {index} is walked from half edge {}, which does not exist",
                    half_edge.0
                ));
                continue;
            };
            if self.face_of[half_edge.0] != Some(index) {
                self.errors.push(format!(
                    "face {index} is walked from half edge {}, which belongs to face {:?}",
                    half_edge.0,
                    edge.face.map(|face| face.0)
                ));
            }
        }
    }

    /// No two half-edges may run the same directed way between the same pair of vertices, and none may
    /// start and end at one vertex. [`EdgeId`] holds a vertex pair and nothing else, so a repeated
    /// direction makes one selectable edge name two half-edge pairs at once - `edge_faces` returns on
    /// its first match and would report whichever neighbour it reached first. This is also the
    /// non-manifold case `from_faces` accepts silently, by overwriting its own twin lookup.
    fn check_directed_edges_are_unique(&mut self) {
        let mut owners: HashMap<(usize, usize), usize> = HashMap::new();

        for index in 0..self.next_of.len() {
            let origin = self.mesh.half_edges[index].origin.0;
            let Some(destination) = self.destination(index) else {
                continue;
            };
            if origin == destination {
                self.errors.push(format!(
                    "half edge {index} starts and ends at vertex {origin}, so it has no direction"
                ));
                continue;
            }
            if let Some(first) = owners.insert((origin, destination), index) {
                self.errors.push(format!(
                    "half edges {first} and {index} both run vertex {origin} to vertex {destination}"
                ));
            }
        }
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

    /// Builds `count` disconnected unit quads, each with four of its own vertices, so the render mesh
    /// has `count * 4` un-welded vertices - the quantity that pushes past 65,536.
    fn quad_grid(count: usize) -> HalfEdgeMesh {
        let mut positions = Vec::with_capacity(count * 4);
        let mut faces = Vec::with_capacity(count);

        for quad in 0..count {
            // Spacing quads two units apart keeps each one a real square, so face normals stay
            // non-degenerate, while still giving every quad vertices no other quad references.
            let x = (quad % 256) as f32 * 2.0;
            let y = (quad / 256) as f32 * 2.0;
            let first = positions.len();
            positions.extend([
                Point3::new(x, y, 0.0),
                Point3::new(x + 1.0, y, 0.0),
                Point3::new(x + 1.0, y + 1.0, 0.0),
                Point3::new(x, y + 1.0, 0.0),
            ]);
            faces.push(vec![first, first + 1, first + 2, first + 3]);
        }

        HalfEdgeMesh::from_faces(&positions, &faces)
    }

    #[test]
    fn render_indices_reach_vertices_beyond_the_u16_range() {
        // 20k quads un-weld to 80k vertices, so the final faces need indices no `u16` can hold. The
        // old `as u16` truncated them, wrapping a far vertex back onto an early one - wrong geometry
        // with no error to catch it.
        let mesh = quad_grid(20_000);
        let render_mesh = mesh.generate_render_buffers();
        assert_eq!(render_mesh.vertices.len(), 80_000);
        assert!(
            render_mesh.vertices.len() > u16::MAX as usize + 1,
            "the mesh must be big enough for this to mean anything"
        );

        let highest = *render_mesh
            .indices
            .iter()
            .max()
            .expect("a 20k-quad grid emits indices");
        assert_eq!(
            highest,
            render_mesh.vertices.len() as u32 - 1,
            "the last vertex needs its own index; a truncated u16 would have wrapped it"
        );
        assert!(
            render_mesh
                .indices
                .iter()
                .all(|i| (*i as usize) < render_mesh.vertices.len()),
            "no index may point past the vertex list"
        );
    }

    #[test]
    fn every_face_fans_from_its_own_first_vertex() {
        let mesh = quad_grid(8);
        let render_mesh = mesh.generate_render_buffers();

        for face in 0..8 {
            let start = (face * 4) as u32;
            // A triangle fan of a quad is two triangles sharing the fan's anchor vertex.
            assert_eq!(
                &render_mesh.indices[face * 6..face * 6 + 6],
                &[start, start + 1, start + 2, start, start + 2, start + 3],
                "face {face} must reference only the four vertices it owns"
            );
        }
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

    /// Asserts the mesh fails validation with at least one error naming `needle`, and hands back the
    /// full list for tests that care how many breaks a single mutation caused.
    fn expect_invalid(mesh: &HalfEdgeMesh, needle: &str) -> TopologyErrors {
        let errors = mesh
            .validate()
            .expect_err("this mesh was broken on purpose");
        assert!(
            !errors.is_empty(),
            "a broken mesh must never validate, and an empty error list is never returned"
        );
        assert!(
            errors.iter().any(|error| error.contains(needle)),
            "expected an error containing {needle:?}, got {errors:?}"
        );
        errors
    }

    #[test]
    fn the_cube_satisfies_every_topology_invariant() {
        cube()
            .validate()
            .unwrap_or_else(|errors| panic!("the cube should be valid: {errors:?}"));
    }

    #[test]
    fn an_empty_mesh_is_valid() {
        // The starting point of every edit: an empty mesh has no broken pointer, and the first
        // append is allowed to begin from it.
        assert!(HalfEdgeMesh::new().validate().is_ok());
    }

    #[test]
    fn a_single_face_is_valid_though_no_edge_has_a_twin() {
        // A missing twin is a boundary edge, not corruption. Extrude and cut will create these on
        // purpose, so the validator must not read an open mesh as broken.
        let positions = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let mesh = HalfEdgeMesh::from_faces(&positions, &[vec![0, 1, 2, 3]]);
        assert!(
            mesh.half_edges.iter().all(|edge| edge.twin.is_none()),
            "a lone quad has only boundary edges"
        );
        mesh.validate()
            .unwrap_or_else(|errors| panic!("an open quad should be valid: {errors:?}"));
    }

    #[test]
    fn two_quads_sharing_an_edge_are_valid() {
        // The smallest mesh mixing boundary and twinned edges, so the passes cannot only be right
        // about meshes that are entirely open or entirely closed.
        let positions = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
        ];
        let mesh = HalfEdgeMesh::from_faces(&positions, &[vec![0, 1, 2, 3], vec![4, 5, 2, 1]]);
        mesh.validate()
            .unwrap_or_else(|errors| panic!("two joined quads should be valid: {errors:?}"));

        let shared = EdgeId::between(VertexId(1), VertexId(2));
        let (own, neighbour) = mesh.edge_faces(shared);
        assert!(
            own.is_some() && neighbour.is_some(),
            "the joined edge must know both faces"
        );
    }

    #[test]
    fn an_asymmetric_twin_is_rejected() {
        let mut mesh = cube();
        // Cleared from one side only, so half edge 0 still points at a twin that no longer points
        // back. `EdgeId` would then name a different edge depending on the side it was read from.
        let twin = mesh.half_edges[0]
            .twin
            .expect("a closed cube has no boundary edges");
        mesh.half_edges[twin.0].twin = None;
        expect_invalid(&mesh, "does not name it back");
    }

    #[test]
    fn a_twin_that_reverses_nothing_is_rejected() {
        let mut mesh = cube();
        // Pointing an edge at some unrelated edge as its twin: symmetric now, but the two do not run
        // opposite directions between one shared pair of vertices.
        let stranger = HalfEdgeId(7);
        mesh.half_edges[0].twin = Some(stranger);
        mesh.half_edges[stranger.0].twin = Some(HalfEdgeId(0));
        expect_invalid(&mesh, "but its twin does not run that pair back");
    }

    #[test]
    fn next_and_prev_must_be_inverses() {
        let mut mesh = cube();
        // Both links still point at half-edges that exist, so only the disagreement itself fails -
        // the case an insert that forgets to rewind a neighbour's `prev` produces.
        mesh.half_edges[0].prev = mesh.half_edges[0].next;
        expect_invalid(&mesh, "which does not name it as its successor");
    }

    #[test]
    fn a_half_edge_outside_every_face_loop_is_rejected() {
        let mut mesh = cube();
        mesh.half_edges[0].next = None;
        expect_invalid(&mesh, "half edge 0 has no next half edge");
    }

    #[test]
    fn a_face_must_name_one_of_its_own_half_edges() {
        let mut mesh = cube();
        // Half edge 5 belongs to face 1, so face 0 would report another face's corners - and
        // `generate_render_buffers` would fan-triangulate them as its own.
        mesh.faces[0].half_edge = Some(HalfEdgeId(5));
        expect_invalid(&mesh, "which belongs to face");
    }

    #[test]
    fn a_vertex_must_name_an_edge_that_starts_at_it() {
        let mut mesh = cube();
        mesh.vertices[0].half_edge = Some(HalfEdgeId(1));
        expect_invalid(&mesh, "which starts at vertex");
    }

    #[test]
    fn an_unlinked_vertex_is_rejected() {
        // The signature of an edit that appends a vertex and never wires it into a face. It renders
        // as nothing and picks as nothing, yet every undo snapshot from here on carries it.
        let positions = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(9.0, 9.0, 9.0),
        ];
        let mesh = HalfEdgeMesh::from_faces(&positions, &[vec![0, 1, 2, 3]]);
        expect_invalid(&mesh, "vertex 4 starts no half edge");
    }

    #[test]
    fn two_faces_sharing_a_directed_edge_are_rejected() {
        // Non-manifold, and the one case `from_faces` accepts without complaint: its twin lookup keys
        // on the directed pair and overwrites its own entry, so both half-edges stay twinless.
        let positions = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(0.5, -1.0, 0.0),
        ];
        let mesh = HalfEdgeMesh::from_faces(&positions, &[vec![0, 1, 2], vec![0, 1, 3]]);
        expect_invalid(&mesh, "both run vertex 0 to vertex 1");
    }

    #[test]
    fn validation_reports_every_break_at_once() {
        let mut mesh = cube();
        mesh.half_edges[0].next = None;
        mesh.faces[1].half_edge = None;
        let errors = mesh.validate().expect_err("two deliberate breaks");
        assert!(
            errors.len() >= 2,
            "one run should list every break, got {errors:?}"
        );
    }

    #[test]
    fn a_broken_mesh_is_described_rather_than_followed() {
        // Every id here names nothing. Resolving them up front is what lets validation report the
        // damage instead of panicking inside `half_edge_destination` on the way to reporting it.
        let mut mesh = cube();
        mesh.half_edges[0].origin = VertexId(404);
        mesh.half_edges[1].next = Some(HalfEdgeId(404));
        mesh.half_edges[2].twin = Some(HalfEdgeId(404));
        mesh.half_edges[3].face = Some(FaceId(404));
        mesh.faces[0].half_edge = Some(HalfEdgeId(404));
        mesh.vertices[0].half_edge = Some(HalfEdgeId(404));

        let errors = mesh.validate().expect_err("six dangling ids");
        let dangling = errors
            .iter()
            .filter(|error| error.contains("does not exist"))
            .count();
        assert!(
            dangling >= 6,
            "each of the six dangling ids should be named by name, got {errors:?}"
        );
    }
}
