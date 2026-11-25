//! The write side of [`HalfEdgeMesh`]: the insertions every edit operation is built from.
//!
//! `mesh.rs` keeps the record types, the read-only topology queries and the validator. This module is
//! where topology gets *created*. They are split because the write side is new and, until the first
//! operation lands, unreachable from a production path - so one `allow(dead_code)` here marks a whole
//! pending layer rather than smearing the attribute over the live queries in `mesh.rs`.
//!
//! Every operation follows one shape, which is the digest's "validate after every edit" rule made
//! structural rather than advisory:
//!
//! 1. refuse an impossible argument **before** anything is touched,
//! 2. rewire a clone,
//! 3. `validate()` the clone and install it only if it holds.
//!
//! The clone is the point, twice over. A primitive that rewired the live mesh and then discovered a bad
//! `twin` would have to undo itself, and undoing half-edge surgery means restoring every field it
//! touched - precisely the bookkeeping that misses one. And because step 3 checks the *result* rather
//! than the input, a mesh that already arrives broken cannot be built on either: its damage rides along
//! into the candidate, which is then refused, so nothing worse is ever installed. Checking the candidate
//! is what makes both hold at once - and it is why there is deliberately no "is this mesh still sound?"
//! pre-check, which would answer the question about the wrong mesh and would also forbid
//! [`HalfEdgeMesh::add_vertex`] from leaving a vertex temporarily unclaimed until the next call links it.
//!
//! Every pointer read on the way is `Option`-safe for the same reason. The read-only queries in
//! `mesh.rs` `expect` a validated mesh, which is right for them and wrong here: an edit is exactly how a
//! bad mesh gets *reported*, so it has to describe damage rather than panic on it. Where a link is
//! missing, the edit declines and names it, and validation lists the full extent.
//!
//! History already pays a mesh clone per action (`scene::DocumentSnapshot`), so the clone is not the
//! expensive part, and these run on an edit - once per user gesture, not once per frame.
#![allow(dead_code)] // Remove when the extrude operation lands as the first caller.

use cgmath::Point3;
use std::collections::{HashMap, HashSet};

use crate::mesh::{
    EdgeId, Face, FaceId, HalfEdge, HalfEdgeId, HalfEdgeMesh, MeshVertex, TopologyErrors, VertexId,
};

/// What [`HalfEdgeMesh::split_edge`] created, so a follow-up operation does not have to rediscover it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitEdge {
    /// The vertex inserted between the two the edge joined.
    pub vertex: VertexId,
    /// The two edges that replaced the one that was split, ordered away from each original endpoint.
    /// The [`EdgeId`] the caller passed in no longer exists, which matters for selection: see
    /// [`HalfEdgeMesh::split_edge`].
    pub edges: [EdgeId; 2],
    /// The new half-edge leaving `vertex` toward the far endpoint. Walk `next`/`prev` from here for
    /// the rest; on a boundary edge it is the only half-edge either new edge has.
    pub half_edge: HalfEdgeId,
}

impl HalfEdgeMesh {
    /// Every edge with no twin - the hull of the mesh - each as the same `(EdgeId, HalfEdgeId)` pair
    /// [`HalfEdgeMesh::edges`] yields.
    ///
    /// A boundary edge has exactly one half-edge by definition, so nothing is reported twice and no
    /// dedup pass is needed. This is where extrude, bevel and cut start: an operation on a face can
    /// only grow walls outward across edges nothing else bounds, and a mesh that reports none is
    /// closed, so the operation has nowhere to go.
    ///
    /// Like the other queries here this reads `next` through [`HalfEdgeMesh::half_edge_edge`] and so
    /// assumes a mesh that [`HalfEdgeMesh::validate`] accepted.
    pub fn boundary_edges(&self) -> Vec<(EdgeId, HalfEdgeId)> {
        (0..self.half_edges.len())
            .filter(|&index| self.half_edges[index].twin.is_none())
            .map(|index| (self.half_edge_edge(HalfEdgeId(index)), HalfEdgeId(index)))
            .collect()
    }

    /// Whether `edge` exists on this mesh and is bounded by only one face.
    ///
    /// An edge the mesh does not have reports `false` rather than failing: callers use this to ask
    /// "can an operation grow here", and a stale id - one a previous split invalidated - cannot grow
    /// anything either. Use [`HalfEdgeMesh::edge_faces`] when you need to tell absent from present.
    pub fn is_boundary_edge(&self, edge: EdgeId) -> bool {
        (0..self.half_edges.len())
            .find(|&index| self.half_edge_edge(HalfEdgeId(index)) == edge)
            .is_some_and(|index| self.half_edges[index].twin.is_none())
    }

    /// Every half-edge starting at `vertex`, in index order.
    ///
    /// There is no fan walk to write. [`crate::mesh::MeshVertex::half_edge`] records one outgoing
    /// edge, and stepping `twin` then `next` around the vertex breaks at the first boundary edge -
    /// which is exactly the vertex extrude and bevel work on. So this scans, matching `edges` and
    /// `edge_faces`, and stays correct on open meshes at linear cost.
    pub fn outgoing_half_edges(&self, vertex: VertexId) -> Vec<HalfEdgeId> {
        self.half_edges
            .iter()
            .enumerate()
            .filter(|(_, edge)| edge.origin == vertex)
            .map(|(index, _)| HalfEdgeId(index))
            .collect()
    }

    /// Whether `vertex` sits on the hull: at least one edge leaving it has no twin.
    pub fn vertex_is_boundary(&self, vertex: VertexId) -> bool {
        self.outgoing_half_edges(vertex)
            .iter()
            .any(|&half_edge| self.half_edges[half_edge.0].twin.is_none())
    }

    /// Appends a vertex that no face claims yet, returning its id.
    ///
    /// The mesh does not satisfy [`HalfEdgeMesh::validate`] until a face links it, which is fine: no
    /// primitive checks the mesh it starts from, only the candidate it produced, so an unclaimed vertex
    /// is legal for as long as it takes the next [`HalfEdgeMesh::append_face`] to claim it. It is
    /// exposed rather than kept private because an operation building a face from new geometry needs the
    /// id before it can name the corners.
    pub fn add_vertex(&mut self, position: Point3<f32>) -> VertexId {
        self.vertices.push(MeshVertex {
            position,
            half_edge: None,
        });
        VertexId(self.vertices.len() - 1)
    }

    /// Inserts a face bounded by `corners` in winding order, twinning each of its edges with the
    /// half-edge that already runs that pair the other way - which is what fuses two separate shells
    /// into one mesh instead of two coincident ones.
    ///
    /// The corners must already exist: this creates topology *between existing vertices*, which is
    /// what a cut, a bridge or a bevel fill needs. [`HalfEdgeMesh::add_vertex`] supplies new ones when
    /// a face is built from nothing. On success the whole mesh validates and the new [`FaceId`] is
    /// returned; on failure nothing about the mesh has changed and every reason is returned, so a
    /// declined operation leaves the document exactly as it was.
    ///
    /// A corner that already names an outgoing edge keeps it - any outgoing edge is a legal place to
    /// start a walk - and one that named none gains one, which is what keeps it walkable.
    pub fn append_face(&mut self, corners: &[VertexId]) -> Result<FaceId, TopologyErrors> {
        let reasons = self.append_face_blockers(corners);
        if !reasons.is_empty() {
            return Err(reasons);
        }

        let mut candidate = self.clone();
        let face_id = FaceId(candidate.faces.len());
        let corner_count = corners.len();
        let first_edge = candidate.half_edges.len();
        let owners = candidate.directed_edge_map();

        candidate.faces.push(Face {
            half_edge: Some(HalfEdgeId(first_edge)),
        });

        for index in 0..corner_count {
            let origin = corners[index];
            let destination = corners[(index + 1) % corner_count];
            // Every reverse the blockers passed is known twinless, so linking it cannot displace an
            // existing pair.
            let twin = owners.get(&(destination.0, origin.0)).copied();
            let inserted = HalfEdgeId(first_edge + index);

            candidate.half_edges.push(HalfEdge {
                origin,
                twin,
                next: Some(HalfEdgeId(first_edge + (index + 1) % corner_count)),
                prev: Some(HalfEdgeId(
                    first_edge + (index + corner_count - 1) % corner_count,
                )),
                face: Some(face_id),
            });

            if let Some(twin_id) = twin {
                candidate.half_edges[twin_id.0].twin = Some(inserted);
            }
            let vertex = &mut candidate.vertices[origin.0];
            if vertex.half_edge.is_none() {
                vertex.half_edge = Some(inserted);
            }
        }

        install_if_valid(
            candidate,
            self,
            format!(
                "appending a face through {} produced topology that does not validate",
                describe_corners(corners)
            ),
        )?;
        Ok(face_id)
    }

    /// Inserts a vertex between the two endpoints of `edge`, at `position`, splitting the edge in two
    /// and rewiring every face on either side. This is the primitive `cut` and `bevel` are built from.
    ///
    /// The two faces that bounded the edge each gain a corner and keep their winding; a boundary edge
    /// has only the one face to rewire. No half-edge is removed or moved to another face, so ids
    /// already handed out stay valid - with one exception worth planning around: the `EdgeId` passed in
    /// names a vertex pair that no longer exists, so it resolves to nothing afterwards. An operation
    /// driven by a selection must therefore rebuild the selected edges rather than assume the old ones
    /// survived, which is the same caveat face extrusion carries.
    ///
    /// Returns what was created, or every reason the split cannot happen, having changed nothing.
    pub fn split_edge(
        &mut self,
        edge: EdgeId,
        position: Point3<f32>,
    ) -> Result<SplitEdge, TopologyErrors> {
        let Some(forward) = self.half_edge_on(edge) else {
            return Err(vec![format!(
                "no half edge runs between vertex {} and vertex {}, so there is nothing to split",
                edge.0, edge.1
            )]);
        };
        // `half_edge_on` matched on the normalized vertex pair, so whichever end this half-edge does
        // not start at is the far one. Reading it back out of the id instead of walking `next` again is
        // what keeps a mesh with a broken link out of the panic business.
        let near = self.half_edges[forward.0].origin;
        let far = if edge.0 == near.0 {
            VertexId(edge.1)
        } else {
            VertexId(edge.0)
        };

        let mut candidate = self.clone();
        let split = candidate.splice_half_edge(forward, far, position);
        install_if_valid(
            candidate,
            self,
            format!(
                "splitting the edge between vertex {} and vertex {} produced topology that does not validate",
                edge.0, edge.1
            ),
        )?;
        Ok(split)
    }

    /// The vertex a half-edge points at, or `None` when its `next` link is missing or dangling.
    ///
    /// The public [`HalfEdgeMesh::half_edge_destination`] panics on such a link, which is the right
    /// call for a validated mesh and the wrong one here: the edit path reads pointers before anything
    /// has been checked, and a mesh that already arrives broken has to be *refused*, not crashed on.
    /// Same reasoning that makes `TopologyScan` resolve its ids before checking them.
    fn destination_of(&self, half_edge: HalfEdgeId) -> Option<usize> {
        self.half_edges
            .get(half_edge.0)?
            .next
            .and_then(|next| self.half_edges.get(next.0))
            .map(|edge| edge.origin.0)
    }

    /// [`HalfEdgeMesh::half_edge_edge`] for a mesh that may not have been validated yet.
    fn edge_id_of(&self, half_edge: HalfEdgeId) -> Option<EdgeId> {
        let origin = self.half_edges.get(half_edge.0)?.origin;
        self.destination_of(half_edge)
            .map(|destination| EdgeId::between(origin, VertexId(destination)))
    }

    /// The first half-edge running between the two vertices of `edge`, if the mesh has one. A valid
    /// mesh has at most two and both report the same `EdgeId`, so either names the edge to split.
    fn half_edge_on(&self, edge: EdgeId) -> Option<HalfEdgeId> {
        (0..self.half_edges.len())
            .map(HalfEdgeId)
            .find(|&half_edge| self.edge_id_of(half_edge) == Some(edge))
    }

    /// Every directed vertex pair the mesh already runs, mapped to the half-edge that runs it. Keyed
    /// by the *directed* pair on purpose: [`EdgeId`] normalizes the two vertices, so keying with it
    /// would make `0 -> 1` and `1 -> 0` collide and every twin lookup would find its own edge.
    fn directed_edge_map(&self) -> HashMap<(usize, usize), HalfEdgeId> {
        let mut owners = HashMap::new();
        for index in 0..self.half_edges.len() {
            let half_edge = HalfEdgeId(index);
            let Some(destination) = self.destination_of(half_edge) else {
                continue;
            };
            owners.insert((self.half_edges[index].origin.0, destination), half_edge);
        }
        owners
    }

    /// Every reason `corners` cannot become a face, gathered rather than short-circuited so one call
    /// reports all of them, matching how [`HalfEdgeMesh::validate`] reports.
    ///
    /// The only geometric objection is a directed edge the mesh already runs: half-edge structure
    /// allows one half-edge per direction between two vertices, and a second would leave `EdgeId`
    /// naming two half-edges at once. There is deliberately no separate "already bounded on both
    /// sides" check - a reverse half-edge that has a twin implies the forward one exists, so that case
    /// is always caught here first and a check for it could never be reached.
    fn append_face_blockers(&self, corners: &[VertexId]) -> TopologyErrors {
        let mut reasons = Vec::new();

        if corners.len() < 3 {
            reasons.push(format!(
                "a face needs at least 3 corners, {} were given",
                corners.len()
            ));
            return reasons;
        }

        let mut listed = HashSet::new();
        for corner in corners {
            if corner.0 >= self.vertices.len() {
                reasons.push(format!("vertex {} does not exist on this mesh", corner.0));
            } else if !listed.insert(*corner) {
                reasons.push(format!(
                    "vertex {} is listed twice, so the face would touch itself",
                    corner.0
                ));
            }
        }
        if !reasons.is_empty() {
            // The edge pass below indexes into `vertices`, so it cannot run until every corner is known good.
            return reasons;
        }

        let owners = self.directed_edge_map();
        let corner_count = corners.len();
        for index in 0..corner_count {
            let origin = corners[index];
            let destination = corners[(index + 1) % corner_count];
            if let Some(&existing) = owners.get(&(origin.0, destination.0)) {
                reasons.push(format!(
                    "half edge {} already runs vertex {} to vertex {}, so a second face cannot bound that same directed edge",
                    existing.0, origin.0, destination.0
                ));
            }
        }
        reasons
    }

    /// The rewire itself, performed on the clone: insert a vertex into the edge `forward` belongs to,
    /// whose far endpoint the caller has already resolved.
    ///
    /// `forward` keeps its origin and gains a successor that starts at the new vertex. That is the
    /// whole trick: a half-edge has no destination field - its destination is read from its successor's
    /// origin - so re-linking rather than rewriting is what turns `forward` from "near to far" into
    /// "near to new" without touching a single existing vertex. The twin side is mirrored, and the two
    /// new half-edges are cross-linked as twins of the two existing ones, because those are the halves
    /// that now run each new edge in reverse.
    fn splice_half_edge(
        &mut self,
        forward: HalfEdgeId,
        far: VertexId,
        position: Point3<f32>,
    ) -> SplitEdge {
        let near = self.half_edges[forward.0].origin;
        // A twin id that names nothing is damage for the final validation to report. Filtering it here
        // keeps this function total, so the edit describes the damage instead of panicking on it.
        let twin = self.half_edges[forward.0]
            .twin
            .filter(|&twin| twin.0 < self.half_edges.len());
        let forward_next = self.half_edges[forward.0].next;
        let forward_face = self.half_edges[forward.0].face;

        let vertex = self.add_vertex(position);

        let inserted = HalfEdgeId(self.half_edges.len());
        self.half_edges.push(HalfEdge {
            origin: vertex,
            twin,
            next: forward_next,
            prev: Some(forward),
            face: forward_face,
        });

        if let Some(twin_id) = twin {
            let twin_next = self.half_edges[twin_id.0].next;
            let twin_face = self.half_edges[twin_id.0].face;
            let twin_inserted = HalfEdgeId(self.half_edges.len());
            self.half_edges.push(HalfEdge {
                origin: vertex,
                twin: Some(forward),
                next: twin_next,
                prev: Some(twin_id),
                face: twin_face,
            });

            self.half_edges[forward.0].twin = Some(twin_inserted);
            self.half_edges[twin_id.0].twin = Some(inserted);
            self.half_edges[twin_id.0].next = Some(twin_inserted);
            if let Some(followed) = twin_next.and_then(|next| self.half_edges.get_mut(next.0)) {
                followed.prev = Some(twin_inserted);
            }
        }

        self.half_edges[forward.0].next = Some(inserted);
        if let Some(followed) = forward_next.and_then(|next| self.half_edges.get_mut(next.0)) {
            followed.prev = Some(inserted);
        }
        self.vertices[vertex.0].half_edge = Some(inserted);

        SplitEdge {
            vertex,
            edges: [EdgeId::between(near, vertex), EdgeId::between(vertex, far)],
            half_edge: inserted,
        }
    }
}

/// Validate the mutated clone and install it only if it holds.
///
/// Taking the candidate by value is what makes the step atomic: on failure it is simply dropped, so
/// "nothing changed" holds by construction rather than depending on an undo list kept perfect by hand.
/// This one check is also what stops an edit building on a corrupted mesh - that damage arrives with the
/// clone and fails here - so the message attributes it to neither cause. It may be damage the document
/// already carried or a bug in the primitive that just rewired it, and the list below is what tells them
/// apart.
fn install_if_valid(
    candidate: HalfEdgeMesh,
    document: &mut HalfEdgeMesh,
    context: String,
) -> Result<(), TopologyErrors> {
    match candidate.validate() {
        Ok(()) => {
            *document = candidate;
            Ok(())
        }
        Err(mut errors) => {
            errors.insert(
                0,
                format!(
                    "{context}, so nothing was installed; the damage below either arrived with the \
                     mesh or was introduced here"
                ),
            );
            Err(errors)
        }
    }
}

/// A rejected loop as `0 -> 1 -> 2`, so a refusal names the corners it was asked about rather than
/// just how many there were.
fn describe_corners(corners: &[VertexId]) -> String {
    corners
        .iter()
        .map(|corner| corner.0.to_string())
        .collect::<Vec<String>>()
        .join(" -> ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The eight corners of the unit cube in `create_cube`'s order, so a hand-built cube can be set
    /// against the generated one.
    fn cube_positions() -> Vec<Point3<f32>> {
        vec![
            Point3::new(-0.5, -0.5, 0.5),
            Point3::new(0.5, -0.5, 0.5),
            Point3::new(0.5, 0.5, 0.5),
            Point3::new(-0.5, 0.5, 0.5),
            Point3::new(-0.5, -0.5, -0.5),
            Point3::new(0.5, -0.5, -0.5),
            Point3::new(0.5, 0.5, -0.5),
            Point3::new(-0.5, 0.5, -0.5),
        ]
    }

    /// The cube's faces as vertex indices, matching `create_cube`'s winding.
    const CUBE_FACES: [[usize; 4]; 6] = [
        [0, 1, 2, 3], // front
        [1, 5, 6, 2], // right
        [5, 4, 7, 6], // back
        [4, 0, 3, 7], // left
        [3, 2, 6, 7], // top
        [4, 5, 1, 0], // bottom
    ];

    fn ids(indices: &[usize]) -> Vec<VertexId> {
        indices.iter().map(|&index| VertexId(index)).collect()
    }

    /// One quad standing alone: the smallest mesh with a hull to grow from.
    fn open_quad() -> HalfEdgeMesh {
        let positions = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        HalfEdgeMesh::from_faces(&positions, &[vec![0, 1, 2, 3]])
    }

    /// A cube with the bottom face left off: closed on five sides, with a four-edge rim still to cap.
    fn open_box() -> HalfEdgeMesh {
        let faces: Vec<Vec<usize>> = CUBE_FACES[..5].iter().map(|face| face.to_vec()).collect();
        HalfEdgeMesh::from_faces(&cube_positions(), &faces)
    }

    /// Component counts, so a refusal can be shown to have added and removed nothing.
    fn shape(mesh: &HalfEdgeMesh) -> (usize, usize, usize) {
        (mesh.vertices.len(), mesh.half_edges.len(), mesh.faces.len())
    }

    /// Asserts the mesh is byte-for-byte the one passed in - the guarantee every primitive makes, since
    /// a declined edit must leave the document the user was looking at intact.
    fn assert_identical(mesh: &HalfEdgeMesh, before: &HalfEdgeMesh) {
        assert_eq!(
            mesh, before,
            "a refused edit must leave every pointer exactly as it was"
        );
    }

    /// Asserts a refusal named `needle` and never came back empty-handed.
    fn expect_reason(errors: &TopologyErrors, needle: &str) {
        assert!(!errors.is_empty(), "a refusal must always give a reason");
        assert!(
            errors.iter().any(|error| error.contains(needle)),
            "expected a reason containing {needle:?}, got {errors:?}"
        );
    }

    #[test]
    fn a_closed_cube_reports_no_boundary_edges() {
        assert!(
            HalfEdgeMesh::create_cube().boundary_edges().is_empty(),
            "every edge of a closed solid is bounded on both sides"
        );
    }

    #[test]
    fn an_open_quad_puts_every_edge_on_the_hull() {
        let mesh = open_quad();
        let hull = mesh.boundary_edges();
        assert_eq!(hull.len(), 4, "a lone quad is four boundary edges");
        assert_eq!(
            hull.len(),
            mesh.edges().len(),
            "nothing is reported twice, so the hull holds every edge exactly once"
        );
    }

    #[test]
    fn joining_two_quads_takes_only_the_shared_edge_off_the_hull() {
        let positions = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
        ];
        let mesh = HalfEdgeMesh::from_faces(&positions, &[vec![0, 1, 2, 3], vec![4, 5, 2, 1]]);

        assert_eq!(mesh.edges().len(), 7, "four plus four less the shared edge");
        assert_eq!(mesh.boundary_edges().len(), 6);
        assert!(
            !mesh.is_boundary_edge(EdgeId::between(VertexId(1), VertexId(2))),
            "the joined edge is bounded on both sides"
        );
        assert!(mesh.is_boundary_edge(EdgeId::between(VertexId(0), VertexId(1))));
    }

    #[test]
    fn outgoing_half_edges_finds_each_face_at_a_cube_corner() {
        let mesh = HalfEdgeMesh::create_cube();
        let fan = mesh.outgoing_half_edges(VertexId(0));

        assert_eq!(fan.len(), 3, "three faces meet at a corner of a cube");
        assert!(
            fan.iter()
                .all(|&half_edge| mesh.half_edges[half_edge.0].origin == VertexId(0)),
            "every edge returned must start at the vertex asked about"
        );
    }

    #[test]
    fn outgoing_half_edges_of_an_unclaimed_vertex_is_empty() {
        // Documents the consequence of the write API being allowed to leave a vertex pending: it is not
        // "on the boundary", it is nowhere, until a face claims it.
        let mut mesh = open_quad();
        let orphan = mesh.add_vertex(Point3::new(4.0, 4.0, 4.0));
        assert!(mesh.outgoing_half_edges(orphan).is_empty());
        assert!(!mesh.vertex_is_boundary(orphan));
    }

    #[test]
    fn every_corner_of_an_open_quad_is_on_the_hull_and_none_of_a_cube_is() {
        let quad = open_quad();
        for index in 0..quad.vertices.len() {
            assert!(quad.vertex_is_boundary(VertexId(index)));
        }

        let cube = HalfEdgeMesh::create_cube();
        for index in 0..cube.vertices.len() {
            assert!(
                !cube.vertex_is_boundary(VertexId(index)),
                "a closed solid leaves no corner exposed"
            );
        }
    }

    #[test]
    fn four_new_vertices_become_a_standalone_face() {
        // The from-nothing path: `add_vertex` leaves each corner unclaimed and the very next call links
        // all four, so no intermediate state is ever handed to the document.
        let mut mesh = HalfEdgeMesh::new();
        let corners: Vec<VertexId> = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ]
        .iter()
        .map(|&position| mesh.add_vertex(position))
        .collect();

        let face = mesh
            .append_face(&corners)
            .unwrap_or_else(|errors| panic!("four fresh corners should form a face: {errors:?}"));

        assert_eq!(mesh.face_vertices(face).len(), 4);
        assert_eq!(
            mesh.boundary_edges().len(),
            4,
            "all four sides are still open"
        );
        mesh.validate()
            .unwrap_or_else(|errors| panic!("the new face should validate: {errors:?}"));
    }

    #[test]
    fn append_face_grows_a_second_quad_and_twins_the_edge_it_crosses() {
        let mut mesh = open_quad();
        let far_left = mesh.add_vertex(Point3::new(2.0, 0.0, 0.0));
        let far_right = mesh.add_vertex(Point3::new(2.0, 1.0, 0.0));

        mesh.append_face(&[far_left, far_right, VertexId(2), VertexId(1)])
            .unwrap_or_else(|errors| panic!("the second quad should append cleanly: {errors:?}"));

        assert_eq!(mesh.faces.len(), 2);
        assert_eq!(mesh.edges().len(), 7);
        assert_eq!(
            mesh.boundary_edges().len(),
            6,
            "only the crossed edge left the hull"
        );

        let shared = EdgeId::between(VertexId(1), VertexId(2));
        let (own, neighbour) = mesh.edge_faces(shared);
        assert!(
            matches!((own, neighbour), (Some(_), Some(_))),
            "the edge the new face grew across must now know both sides"
        );
    }

    #[test]
    fn append_face_caps_an_open_box_into_a_closed_solid() {
        // The real test of twinning: five faces built by `from_faces`, the sixth added by hand. If the
        // new face merely parked its own half-edges the rim would still show as boundary; if it linked
        // the wrong pairs, validation would catch an asymmetric or unreversed twin.
        let mut mesh = open_box();
        assert_eq!(mesh.boundary_edges().len(), 4, "the uncapped rim");

        mesh.append_face(&ids(&CUBE_FACES[5]))
            .unwrap_or_else(|errors| panic!("the bottom should cap the box: {errors:?}"));

        assert_eq!(mesh.faces.len(), 6);
        assert!(
            mesh.half_edges.iter().all(|edge| edge.twin.is_some()),
            "a closed solid leaves no half-edge without a partner"
        );
        assert!(mesh.boundary_edges().is_empty());
        assert_eq!(
            mesh.generate_render_buffers().indices.len(),
            36,
            "six quad faces fan to 12 triangles"
        );
    }

    #[test]
    fn a_cube_built_one_face_at_a_time_matches_the_generated_one() {
        // Sufficiency check for the whole layer: with only `add_vertex` and `append_face`, and no help
        // from `from_faces` or `create_cube`, a cube comes out indistinguishable from the generated one.
        let positions = cube_positions();
        let mut mesh = HalfEdgeMesh::new();
        let mut claimed: HashMap<usize, VertexId> = HashMap::new();

        for face in CUBE_FACES {
            let corners: Vec<VertexId> = face
                .iter()
                .map(|&index| {
                    *claimed
                        .entry(index)
                        .or_insert_with(|| mesh.add_vertex(positions[index]))
                })
                .collect();
            mesh.append_face(&corners).unwrap_or_else(|errors| {
                panic!("cube face {face:?} should append cleanly: {errors:?}")
            });
        }

        let generated = HalfEdgeMesh::create_cube();
        mesh.validate()
            .unwrap_or_else(|errors| panic!("the hand-built cube should validate: {errors:?}"));
        assert_eq!(shape(&mesh), shape(&generated));
        assert!(mesh.boundary_edges().is_empty());
        assert!(mesh
            .faces
            .iter()
            .enumerate()
            .all(|(index, _)| mesh.face_vertices(FaceId(index)).len() == 4));

        // Ids come out in the order the faces needed the corners, so equivalence is asserted over the
        // geometry the ids point at rather than over the ids themselves.
        let sorted = |mesh: &HalfEdgeMesh| {
            let mut points: Vec<[f32; 3]> = mesh
                .vertices
                .iter()
                .map(|vertex| [vertex.position.x, vertex.position.y, vertex.position.z])
                .collect();
            points.sort_by(|a, b| a.partial_cmp(b).unwrap());
            points
        };
        assert_eq!(sorted(&mesh), sorted(&generated));
    }

    #[test]
    fn append_face_refuses_a_directed_edge_the_mesh_already_runs() {
        let before = HalfEdgeMesh::create_cube();
        let mut mesh = before.clone();

        let errors = mesh
            .append_face(&ids(&[3, 2, 1, 0]))
            .expect_err("the front face is already bounded");
        expect_reason(&errors, "already runs vertex 3 to vertex 2");
        assert_identical(&mesh, &before);
    }

    #[test]
    fn append_face_refuses_fewer_corners_than_a_face_needs() {
        let before = open_quad();
        let mut mesh = before.clone();

        for corners in [vec![], vec![VertexId(0)], vec![VertexId(0), VertexId(1)]] {
            let errors = mesh
                .append_face(&corners)
                .expect_err("fewer than three corners bound nothing");
            expect_reason(&errors, "at least 3 corners");
        }
        assert_identical(&mesh, &before);
    }

    #[test]
    fn append_face_refuses_a_corner_that_is_not_on_the_mesh() {
        let before = open_quad();
        let mut mesh = before.clone();

        let errors = mesh
            .append_face(&ids(&[0, 1, 42, 3]))
            .expect_err("vertex 42 does not exist");
        expect_reason(&errors, "vertex 42 does not exist");
        assert_identical(&mesh, &before);
    }

    #[test]
    fn append_face_refuses_a_corner_it_already_listed() {
        let before = open_quad();
        let mut mesh = before.clone();

        let errors = mesh
            .append_face(&ids(&[0, 1, 0, 3]))
            .expect_err("a face that visits one vertex twice touches itself");
        expect_reason(&errors, "vertex 0 is listed twice");
        assert_identical(&mesh, &before);
    }

    #[test]
    fn append_face_refuses_to_build_on_a_mesh_that_is_already_broken() {
        // The argument here is perfectly good - it is the open box's own uncapped rim. What stops the
        // edit is damage elsewhere riding along into the candidate and failing its validation, which is
        // how a corrupted document is kept from being edited into something worse.
        let mut mesh = open_box();
        mesh.half_edges[7].face = None;
        let before = mesh.clone();

        let errors = mesh
            .append_face(&ids(&CUBE_FACES[5]))
            .expect_err("the mesh is corrupt even though the corners are not");
        expect_reason(&errors, "belongs to no face");
        expect_reason(&errors, "nothing was installed");
        assert_identical(&mesh, &before);
    }

    #[test]
    fn split_edge_adds_a_corner_to_both_faces_of_the_edge() {
        let mut mesh = HalfEdgeMesh::create_cube();
        let shared = EdgeId::between(VertexId(0), VertexId(1));

        let split = mesh
            .split_edge(shared, Point3::new(0.0, -0.5, 0.5))
            .unwrap_or_else(|errors| panic!("splitting a cube edge should succeed: {errors:?}"));

        assert_eq!(mesh.vertices.len(), 9);
        assert_eq!(mesh.half_edges.len(), 26, "two half-edges, one per side");
        assert_eq!(mesh.faces.len(), 6, "no face is added, removed or moved");
        mesh.validate()
            .unwrap_or_else(|errors| panic!("the split mesh should validate: {errors:?}"));

        let corner_counts: Vec<usize> = (0..mesh.faces.len())
            .map(|index| mesh.face_vertices(FaceId(index)).len())
            .collect();
        assert_eq!(
            corner_counts.iter().filter(|&&count| count == 5).count(),
            2,
            "the front and bottom faces each gained a corner: {corner_counts:?}"
        );
        assert_eq!(
            corner_counts.iter().filter(|&&count| count == 4).count(),
            4,
            "every other face is untouched: {corner_counts:?}"
        );
        assert_eq!(split.vertex, VertexId(8));
    }

    #[test]
    fn split_edge_on_the_hull_rewires_only_the_face_it_borders() {
        let mut mesh = open_quad();
        let hull = EdgeId::between(VertexId(0), VertexId(1));

        mesh.split_edge(hull, Point3::new(0.5, 0.0, 0.0))
            .unwrap_or_else(|errors| {
                panic!("splitting a boundary edge should succeed: {errors:?}")
            });

        assert_eq!(mesh.vertices.len(), 5);
        assert_eq!(
            mesh.half_edges.len(),
            5,
            "a boundary edge has one side, so the split one is replaced by exactly two"
        );
        assert_eq!(mesh.face_vertices(FaceId(0)).len(), 5);
        assert_eq!(
            mesh.boundary_edges().len(),
            5,
            "one open edge became two, so the hull grew by one"
        );
    }

    #[test]
    fn split_edge_replaces_the_edge_id_a_selection_was_holding() {
        // The caveat split leaves for callers: component ids survive, but the *edge* that was named no
        // longer exists, so a selection held across an edit must be rebuilt rather than trusted.
        let mut mesh = HalfEdgeMesh::create_cube();
        let old = EdgeId::between(VertexId(0), VertexId(1));

        let split = mesh
            .split_edge(old, Point3::new(0.0, -0.5, 0.5))
            .expect("splitting a cube edge should succeed");

        assert_eq!(
            split.edges,
            [
                EdgeId::between(VertexId(0), split.vertex),
                EdgeId::between(split.vertex, VertexId(1)),
            ]
        );
        let present: Vec<EdgeId> = mesh.edges().into_iter().map(|(edge, _)| edge).collect();
        assert!(present.contains(&split.edges[0]));
        assert!(present.contains(&split.edges[1]));
        assert!(
            !present.contains(&old),
            "the pair it joined no longer exists"
        );
        assert_eq!(mesh.edge_faces(old), (None, None));
        assert!(
            !mesh.is_boundary_edge(old),
            "an absent edge is not on the hull"
        );
    }

    #[test]
    fn split_edge_leaves_the_new_vertex_inside_the_mesh() {
        let mut cube = HalfEdgeMesh::create_cube();
        let interior = cube
            .split_edge(
                EdgeId::between(VertexId(0), VertexId(1)),
                Point3::new(0.0, -0.5, 0.5),
            )
            .expect("splitting a cube edge should succeed");
        let fan = cube.outgoing_half_edges(interior.vertex);
        assert_eq!(
            fan.len(),
            2,
            "both halves of the split start at the new vertex"
        );
        assert!(
            fan.iter()
                .all(|&half_edge| cube.half_edges[half_edge.0].origin == interior.vertex),
            "the new vertex must be walkable out from"
        );
        assert!(!cube.vertex_is_boundary(interior.vertex));
        assert!(
            cube.boundary_edges().is_empty(),
            "the solid is still closed"
        );

        let mut quad = open_quad();
        let on_hull = quad
            .split_edge(
                EdgeId::between(VertexId(0), VertexId(1)),
                Point3::new(0.5, 0.0, 0.0),
            )
            .expect("splitting a boundary edge should succeed");
        assert!(
            quad.vertex_is_boundary(on_hull.vertex),
            "a vertex on an open edge is itself on the hull"
        );
    }

    #[test]
    fn split_edge_refuses_an_edge_the_mesh_does_not_have() {
        let before = open_quad();
        let mut mesh = before.clone();

        let errors = mesh
            .split_edge(
                EdgeId::between(VertexId(0), VertexId(2)),
                Point3::new(0.5, 0.5, 0.0),
            )
            .expect_err("that pair is a diagonal, not an edge");
        expect_reason(&errors, "no half edge runs between vertex 0 and vertex 2");
        assert_identical(&mesh, &before);
    }

    #[test]
    fn split_edge_refuses_to_build_on_a_mesh_that_is_already_broken() {
        let mut mesh = open_quad();
        mesh.half_edges[2].face = None;
        let before = mesh.clone();

        let errors = mesh
            .split_edge(
                EdgeId::between(VertexId(0), VertexId(1)),
                Point3::new(0.5, 0.0, 0.0),
            )
            .expect_err("the damage elsewhere must stop this edit");
        expect_reason(&errors, "belongs to no face");
        expect_reason(&errors, "nothing was installed");
        assert_identical(&mesh, &before);
    }
}
