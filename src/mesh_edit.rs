//! The write side of [`HalfEdgeMesh`]: the insertions every edit operation is built from.
//!
//! `mesh.rs` keeps the record types, the read-only topology queries and the validator. This module is
//! where topology gets *created*. They stay split because the two halves are read under different
//! contracts: the queries in `mesh.rs` may `expect` a mesh that already validates, and nothing here may,
//! for the reason given below.
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

use cgmath::{InnerSpace, Point3, Vector3};
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

/// How far a face must travel to count as extruded. Below this the walls collapse onto their own base
/// edge and every one of them is a degenerate face with no area - which [`HalfEdgeMesh::validate`] does
/// not look for, since judging area needs geometry rather than topology.
const MIN_EXTRUDE_OFFSET: f32 = 1e-6;

/// What [`HalfEdgeMesh::extrude_faces`] created.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Extruded {
    /// The faces the caller named, in the order given and without repeats, unchanged in id and now
    /// translated by the offset - the operation removes nothing, so a selection held across an extrusion
    /// still names the same faces.
    pub source_faces: Vec<FaceId>,
    /// The walls grown along the region's free edges, in boundary order: each is `[origin, destination,
    /// destination's copy, origin's copy]` for one half-edge of the region. Reported so a caller can count
    /// them, highlight them or chain them into another operation; the selection itself deliberately stays
    /// on `source_faces`, both because that is what Maya does and because it is what makes the next press
    /// travel along the way the geometry under the cursor faces.
    pub side_faces: Vec<FaceId>,
    /// The translated copy of every corner the region touched, in the order they were created.
    pub new_vertices: Vec<VertexId>,
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
        let face_id = candidate.push_face(corners);
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

    /// The insertion [`HalfEdgeMesh::append_face`] performs once its blockers are clear, without the
    /// clone-and-validate wrapped around it: every edge is twinned with the half-edge already running it
    /// in reverse, which is what fuses a new face onto its neighbours.
    ///
    /// Extrude needs the mechanics without the per-call validation, because it leaves the mesh
    /// deliberately half-built - region lifted off its neighbours, walls not yet attached - so nothing
    /// can validate until the last wall is on. A caller that uses this owns the promise that whatever it
    /// installs passed [`HalfEdgeMesh::validate`], and assumes the blockers already held: corners that
    /// exist, at least three of them, and no directed pair the mesh already runs.
    fn push_face(&mut self, corners: &[VertexId]) -> FaceId {
        let face_id = FaceId(self.faces.len());
        let corner_count = corners.len();
        let first_edge = self.half_edges.len();
        let owners = self.directed_edge_map();

        self.faces.push(Face {
            half_edge: Some(HalfEdgeId(first_edge)),
        });

        for index in 0..corner_count {
            let origin = corners[index];
            let destination = corners[(index + 1) % corner_count];
            // Every reverse the blockers passed is known twinless, so linking it cannot displace an
            // existing pair.
            let twin = owners.get(&(destination.0, origin.0)).copied();
            let inserted = HalfEdgeId(first_edge + index);

            self.half_edges.push(HalfEdge {
                origin,
                twin,
                next: Some(HalfEdgeId(first_edge + (index + 1) % corner_count)),
                prev: Some(HalfEdgeId(
                    first_edge + (index + corner_count - 1) % corner_count,
                )),
                face: Some(face_id),
            });

            if let Some(twin_id) = twin {
                self.half_edges[twin_id.0].twin = Some(inserted);
            }
            let vertex = &mut self.vertices[origin.0];
            if vertex.half_edge.is_none() {
                vertex.half_edge = Some(inserted);
            }
        }

        face_id
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

    /// Pushes `face` outward along `offset`, leaving it translated in place and a wall grown along every
    /// edge of it that nothing else bounds.
    ///
    /// This forwards to [`HalfEdgeMesh::extrude_faces`] with a one-face region; see that method for the
    /// mechanics and the refusals.
    pub fn extrude_face(
        &mut self,
        face: FaceId,
        offset: Vector3<f32>,
    ) -> Result<Extruded, TopologyErrors> {
        self.extrude_faces(std::slice::from_ref(&face), offset)
    }

    /// Pushes a whole set of faces outward along `offset` as one piece.
    ///
    /// Nothing is removed to make the shape: each named face keeps its id and its winding and merely
    /// re-points at a translated copy of the corners it touched, and the space they left is closed by new
    /// quad walls. That is what the append-only storage rule demands of an operation, and it leaves ids
    /// already handed out - including a live selection - naming the same things afterwards.
    ///
    /// Faces sharing an edge move together and grow no wall between themselves, so extruding a strip of a
    /// model reads as one slab rather than several coincident shells; faces sharing nothing are simply
    /// extruded side by side. Each wall is twinned against whatever already ran its edge - a neighbouring
    /// face on the base side, the lifted source face on the top side, the walls either side of it along
    /// each corner - so a region taken out of a closed solid leaves it closed rather than holed.
    ///
    /// Refuses, having changed nothing, when a face is absent or its loop broken, when `offset` is zero or
    /// not finite, when the selection shares every one of its edges with itself (no free boundary to grow
    /// along), or when a corner belongs to no face outside the selection and would be left stranded -
    /// which is the case Maya resolves by splitting the corner per diverging face, and this build does not
    /// do yet.
    pub fn extrude_faces(
        &mut self,
        faces: &[FaceId],
        offset: Vector3<f32>,
    ) -> Result<Extruded, TopologyErrors> {
        let loops = self.region_loops(faces)?;
        if loops.is_empty() {
            return Err(vec![format!(
                "an extrusion needs at least one face to move, and {} named none",
                faces.len()
            )]);
        }
        if let Some(reason) = offset_blocker(offset) {
            return Err(vec![reason]);
        }

        let region: HashSet<FaceId> = loops.iter().map(|&(face, _)| face).collect();
        // Captured while the base corners still exist on these half-edges.
        let walls = self.region_walls(&loops, &region);
        if walls.is_empty() {
            return Err(vec![
                "every edge of the selection is shared with the selection itself, so the extrusion has \
                 no free boundary to grow a wall along and nothing would change"
                    .to_string(),
            ]);
        }
        if let Some(reason) = self.stranded_corner_blocker(&loops, &walls, &region) {
            return Err(vec![reason]);
        }

        let mut candidate = self.clone();

        // One translated copy of each corner the region touches. A corner two selected faces share is
        // copied once, which is exactly what keeps them welded as they travel.
        let mut moved: HashMap<VertexId, VertexId> = HashMap::new();
        let mut new_vertices = Vec::with_capacity(self.vertices.len());
        for (_, loop_edges) in &loops {
            for &half_edge in loop_edges {
                let origin = candidate.half_edges[half_edge.0].origin;
                if moved.contains_key(&origin) {
                    continue;
                }
                let position = candidate.vertices[origin.0].position + offset;
                let copied = candidate.add_vertex(position);
                moved.insert(origin, copied);
                new_vertices.push(copied);
            }
        }

        // Lift the region. Each half-edge keeps its `next`, `prev`, `twin` and `face` and takes a new
        // origin, which is all a face needs to become its own translated copy - and why no loop, twin or
        // untouched face is disturbed.
        let mut arrived: HashMap<VertexId, HalfEdgeId> = HashMap::new();
        for index in 0..candidate.half_edges.len() {
            let edge = &mut candidate.half_edges[index];
            if edge.face.is_some_and(|face| region.contains(&face)) {
                if let Some(&copied) = moved.get(&edge.origin) {
                    edge.origin = copied;
                    arrived.entry(copied).or_insert(HalfEdgeId(index));
                }
            }
        }
        for (copied, half_edge) in arrived {
            candidate.vertices[copied.0].half_edge = Some(half_edge);
        }

        // Grow the walls, now that the lifted face offers each wall's top edge as its twin.
        let mut side_faces = Vec::with_capacity(walls.len());
        for (origin, destination) in &walls {
            side_faces.push(candidate.push_face(&[
                *origin,
                *destination,
                moved[destination],
                moved[origin],
            ]));
        }

        candidate.repair_outgoing_edges(&moved.keys().copied().collect());

        install_if_valid(
            candidate,
            self,
            format!(
                "extruding {} face{} along ({}, {}, {}) produced topology that does not validate",
                region.len(),
                if region.len() == 1 { "" } else { "s" },
                offset.x,
                offset.y,
                offset.z
            ),
        )?;

        Ok(Extruded {
            source_faces: loops.iter().map(|&(face, _)| face).collect(),
            side_faces,
            new_vertices,
        })
    }

    /// The loops of `faces` in the order given, repeats dropped, or every reason one of them cannot be
    /// used. A face named twice moves once; a face that is absent or unwalkable is refused by name while
    /// the rest of the selection is reported in the same pass, so one refusal says everything wrong.
    fn region_loops(
        &self,
        faces: &[FaceId],
    ) -> Result<Vec<(FaceId, Vec<HalfEdgeId>)>, TopologyErrors> {
        let mut reasons = TopologyErrors::new();
        let mut seen: HashSet<FaceId> = HashSet::new();
        let mut loops = Vec::with_capacity(faces.len());

        for &face in faces {
            if face.0 >= self.faces.len() {
                reasons.push(format!("face {} does not exist on this mesh", face.0));
                continue;
            }
            if !seen.insert(face) {
                continue;
            }
            match self.face_walk(face) {
                Some(loop_edges) => loops.push((face, loop_edges)),
                None => reasons.push(format!(
                    "face {} does not bound a closed loop of at least three corners, so it cannot be extruded",
                    face.0
                )),
            }
        }

        if reasons.is_empty() {
            Ok(loops)
        } else {
            Err(reasons)
        }
    }

    /// The half-edges bounding `face`, following `next` until the loop comes round, or `None` when it
    /// will not.
    ///
    /// [`HalfEdgeMesh::face_half_edges`] `expect`s a mesh that [`HalfEdgeMesh::validate`] accepted, which
    /// is right for a query and wrong here: an extrude is asked about a mesh before anything has checked
    /// it, and must refuse rather than crash. The iteration cap is what turns a loop that wanders into
    /// another face into a refusal instead of a spin.
    fn face_walk(&self, face: FaceId) -> Option<Vec<HalfEdgeId>> {
        let first = self.faces.get(face.0)?.half_edge?;
        let mut walked = Vec::new();
        let mut cursor = first;

        for _ in 0..=self.half_edges.len() {
            let edge = self.half_edges.get(cursor.0)?;
            if edge.face != Some(face) {
                return None;
            }
            walked.push(cursor);
            cursor = edge.next?;
            if cursor == first {
                return (walked.len() >= 3).then_some(walked);
            }
        }

        None
    }

    /// The directed corner pairs of the region that nothing on the far side bounds - the base edge of one
    /// wall each, in loop order so `side_faces` comes out walkable alongside the region.
    ///
    /// An edge whose twin names no face, or has no twin at all, counts as free: the region is the only
    /// thing holding that side, so a wall belongs there.
    fn region_walls(
        &self,
        loops: &[(FaceId, Vec<HalfEdgeId>)],
        region: &HashSet<FaceId>,
    ) -> Vec<(VertexId, VertexId)> {
        let mut walls = Vec::new();

        for (_, loop_edges) in loops {
            let count = loop_edges.len();
            for (index, &half_edge) in loop_edges.iter().enumerate() {
                let far_side_face = self
                    .half_edges
                    .get(half_edge.0)
                    .and_then(|edge| edge.twin)
                    .and_then(|twin| self.half_edges.get(twin.0))
                    .and_then(|edge| edge.face);
                if far_side_face.is_none_or(|face| !region.contains(&face)) {
                    walls.push((
                        self.half_edges[half_edge.0].origin,
                        self.half_edges[loop_edges[(index + 1) % count].0].origin,
                    ));
                }
            }
        }

        walls
    }

    /// Why a corner of the region would be left owning nothing after the lift, if any would be.
    ///
    /// A corner travels with the faces that claim it. If no face around it lies outside the region it
    /// moves entirely, and unless a wall also starts there the vertex it was is left behind with no
    /// outgoing edge. `validate` would catch that, but only once the work is done and with a message
    /// about an orphan vertex that says nothing about the selection that caused it - and Maya's answer,
    /// splitting the corner once per diverging face, is not built yet.
    fn stranded_corner_blocker(
        &self,
        loops: &[(FaceId, Vec<HalfEdgeId>)],
        walls: &[(VertexId, VertexId)],
        region: &HashSet<FaceId>,
    ) -> Option<String> {
        let mut corners: Vec<VertexId> = loops
            .iter()
            .flat_map(|(_, loop_edges)| loop_edges.iter())
            .filter_map(|&half_edge| self.half_edges.get(half_edge.0))
            .map(|edge| edge.origin)
            .collect();
        corners.sort_by_key(|corner| corner.0);
        corners.dedup();

        let walled: HashSet<VertexId> = walls.iter().flat_map(|&(a, b)| [a, b]).collect();
        let mut kept: HashSet<VertexId> = HashSet::new();
        let mut lifted: HashSet<VertexId> = HashSet::new();
        for edge in &self.half_edges {
            let origin = edge.origin;
            if !corners.contains(&origin) {
                continue;
            }
            match edge.face {
                Some(face) if region.contains(&face) => {
                    lifted.insert(origin);
                }
                _ => {
                    kept.insert(origin);
                }
            }
        }

        corners
            .iter()
            .find(|&corner| {
                lifted.contains(corner) && !kept.contains(corner) && !walled.contains(corner)
            })
            .map(|corner| {
                format!(
                    "vertex {} belongs only to the faces being extruded, so the extrusion would leave it \
                     stranded; Maya splits it into one corner per diverging face, which this build does \
                     not do yet",
                    corner.0
                )
            })
    }

    /// Gives every corner the operation moved a recorded outgoing edge that actually starts at it.
    ///
    /// `MeshVertex::half_edge` is only a cached entry point - any outgoing edge is a legal one - and
    /// lifting a region leaves the corners it came from naming an edge that has since travelled. This has
    /// to run after the walls are up, since a base corner frequently has nothing left to point at until
    /// its wall exists. Scoped to the corners the operation disturbed, so damage elsewhere stays visible
    /// to `validate` rather than being quietly patched over.
    fn repair_outgoing_edges(&mut self, corners: &HashSet<VertexId>) {
        let mut first_out: HashMap<usize, HalfEdgeId> = HashMap::new();
        for index in 0..self.half_edges.len() {
            first_out
                .entry(self.half_edges[index].origin.0)
                .or_insert(HalfEdgeId(index));
        }

        for &corner in corners {
            let still_points_home = self
                .vertices
                .get(corner.0)
                .and_then(|vertex| vertex.half_edge)
                .and_then(|edge| self.half_edges.get(edge.0))
                .is_some_and(|edge| edge.origin == corner);
            if !still_points_home {
                self.vertices[corner.0].half_edge = first_out.get(&corner.0).copied();
            }
        }
    }
}

/// Why `offset` cannot drive an extrusion, if it cannot.
///
/// A zero-length offset is refused rather than waved through: it would grow walls of no area, and the
/// validator judges topology, not geometry, so it would never notice.
fn offset_blocker(offset: Vector3<f32>) -> Option<String> {
    if !offset.x.is_finite() || !offset.y.is_finite() || !offset.z.is_finite() {
        return Some(format!(
            "the extrusion offset ({}, {}, {}) is not a finite distance",
            offset.x, offset.y, offset.z
        ));
    }
    if offset.dot(offset) < MIN_EXTRUDE_OFFSET * MIN_EXTRUDE_OFFSET {
        return Some(
            "an extrusion needs a non-zero offset, or every wall it grows collapses onto its own base \
             edge as a face with no area"
                .to_string(),
        );
    }
    None
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

    /// The unit vector the front face of the cube is pushed by, matching its outward (+z) normal.
    fn push_front() -> Vector3<f32> {
        Vector3::new(0.0, 0.0, 0.5)
    }

    /// Asserts a face is the flat quad `positions` in order, which is how a lifted copy is told apart
    /// from the face it came from and from a wall: only one of the three can hold these four points.
    fn assert_face_at(mesh: &HalfEdgeMesh, face: FaceId, positions: &[[f32; 3]]) {
        let actual: Vec<[f32; 3]> = mesh
            .face_positions(face)
            .iter()
            .map(|point| [point.x, point.y, point.z])
            .collect();
        let same_set = {
            let mut got = actual.clone();
            let mut want: Vec<[f32; 3]> = positions.to_vec();
            // Sorted on the raw floats with `partial_cmp` rather than `sort_by_key`, which f32 cannot
            // satisfy because it is only `PartialOrd`. Unreachable in practice: no test feeds a NaN.
            let order = |a: &[f32; 3], b: &[f32; 3]| {
                (a[0], a[1], a[2]).partial_cmp(&(b[0], b[1], b[2])).unwrap()
            };
            got.sort_by(order);
            want.sort_by(order);
            got == want
        };
        assert!(
            same_set,
            "face {} should bound exactly {positions:?}, got {actual:?}",
            face.0
        );
    }

    #[test]
    fn extruding_a_cube_face_lifts_it_and_keeps_the_solid_closed() {
        let before = HalfEdgeMesh::create_cube();
        let mut mesh = before.clone();

        let shape = mesh
            .extrude_face(FaceId(0), push_front())
            .expect("a face of a cube has four free edges to grow walls from");

        assert_eq!(
            shape.source_faces,
            vec![FaceId(0)],
            "the face named is the face moved"
        );
        assert_eq!(shape.side_faces.len(), 4, "one wall per edge of the face");
        assert_eq!(shape.new_vertices.len(), 4);
        assert_eq!(
            mesh.vertices.len(),
            12,
            "append-only: four added, none removed"
        );
        assert_eq!(mesh.faces.len(), 10, "the cube plus four walls");
        assert!(
            mesh.boundary_edges().is_empty(),
            "a region taken out of a closed solid must leave it closed, not holed"
        );
        assert_eq!(
            mesh.generate_render_buffers().indices.len(),
            60,
            "ten quad faces fan to 20 triangles"
        );
        // The untouched five are byte-identical, which is what lets a selection or an id handed out
        // before the edit keep naming the same geometry after it.
        for face in 1..6 {
            assert_eq!(
                mesh.face_vertices(FaceId(face)),
                before.face_vertices(FaceId(face)),
                "face {} must be untouched by an extrusion it is not part of",
                face
            );
        }
    }

    #[test]
    fn the_lifted_face_is_the_original_id_at_the_translated_corners() {
        let mut mesh = HalfEdgeMesh::create_cube();
        mesh.extrude_face(FaceId(0), push_front())
            .expect("a face of a cube should extrude");

        let lifted: Vec<[f32; 3]> = (0..4)
            .map(|index| Point3::new(-0.5 + 0.5 * (index % 2) as f32 * 2.0 - 0.5, 0.0, 0.0))
            .map(|_| [0.0f32; 3])
            .collect();
        let _ = lifted;

        // The front face of the cube spans x and y at z = 0.5; pushed 0.5 along z it spans the same x
        // and y at z = 1.0, and it is still FaceId(0).
        assert_face_at(
            &mesh,
            FaceId(0),
            &[
                [-0.5, -0.5, 1.0],
                [0.5, -0.5, 1.0],
                [0.5, 0.5, 1.0],
                [-0.5, 0.5, 1.0],
            ],
        );
        assert_eq!(
            mesh.face_normal(FaceId(0)).z,
            1.0,
            "the lifted copy keeps the winding of the face it came from"
        );
    }

    #[test]
    fn each_wall_bridges_a_base_edge_to_the_edge_it_lifted() {
        let mut mesh = HalfEdgeMesh::create_cube();
        let shape = mesh
            .extrude_face(FaceId(0), push_front())
            .expect("a face of a cube should extrude");

        // Every wall is a quad with two corners from the original face and two from its lifted copy,
        // which is what makes the extrusion a solid prism rather than a face floating off a hole.
        for &wall in &shape.side_faces {
            let corners = mesh.face_vertices(wall);
            assert_eq!(corners.len(), 4, "a wall is a quad");
            let bases = corners
                .iter()
                .filter(|corner| [0usize, 1, 2, 3].contains(&corner.0))
                .count();
            let lifted = corners
                .iter()
                .filter(|corner| shape.new_vertices.contains(corner))
                .count();
            assert_eq!(bases, 2, "two corners came off the face where it was");
            assert_eq!(lifted, 2, "two corners came off the face where it went");
        }

        // The walls meet the lifted face along its new edges and the neighbouring faces along the old
        // ones, so no edge of a wall is left unbounded.
        for &wall in &shape.side_faces {
            for half_edge in mesh.face_half_edges(wall) {
                let twin = mesh.half_edges[half_edge.0]
                    .twin
                    .expect("an extrusion of a closed solid leaves no wall edge open");
                assert!(
                    mesh.half_edges[twin.0].face.is_some(),
                    "every wall edge bounds a face on its far side"
                );
            }
        }
    }

    #[test]
    fn extruding_the_whole_solid_is_refused_for_want_of_a_free_edge() {
        let before = HalfEdgeMesh::create_cube();
        let mut mesh = before.clone();

        let all: Vec<FaceId> = (0..6).map(FaceId).collect();
        let errors = mesh
            .extrude_faces(&all, push_front())
            .expect_err("a closed solid has no hull left to grow walls from");
        expect_reason(&errors, "no free boundary to grow a wall along");
        assert_identical(&mesh, &before);
    }

    #[test]
    fn a_zero_or_infinite_offset_is_refused_before_anything_is_grown() {
        for offset in [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(f32::NAN, 0.0, 0.0),
            Vector3::new(0.0, f32::INFINITY, 0.0),
        ] {
            let before = HalfEdgeMesh::create_cube();
            let mut mesh = before.clone();

            let errors = mesh
                .extrude_face(FaceId(0), offset)
                .expect_err("a degenerate offset must not grow zero-area walls");
            assert!(
                !errors.is_empty(),
                "a refused offset must always say why, and {offset:?} said nothing"
            );
            assert_identical(&mesh, &before);
        }
    }

    #[test]
    fn an_absent_face_is_refused_by_name() {
        let before = HalfEdgeMesh::create_cube();
        let mut mesh = before.clone();

        let errors = mesh
            .extrude_face(FaceId(99), push_front())
            .expect_err("face 99 is not on a cube");
        expect_reason(&errors, "face 99 does not exist");
        assert_identical(&mesh, &before);
    }

    #[test]
    fn a_corner_the_region_does_not_release_is_refused_as_stranded() {
        // Front, right and top all meet at corner 2, and no other selected face touches it. Every edge at
        // corner 2 is shared inside the selection, so nothing walls it off and the lift would carry it
        // away whole, leaving the vertex it was owning no edge at all. Maya splits such a corner into one
        // copy per diverging face; refusing is the honest answer until that exists, and it has to say so
        // up front rather than report an orphan vertex after the damage is done.
        let before = HalfEdgeMesh::create_cube();
        let mut mesh = before.clone();

        let errors = mesh
            .extrude_faces(&[FaceId(0), FaceId(1), FaceId(4)], push_front())
            .expect_err("corner 2 belongs only to the three faces being moved");
        expect_reason(&errors, "vertex 2 belongs only to the faces being extruded");
        assert_identical(&mesh, &before);
    }

    #[test]
    fn two_faces_sharing_an_edge_move_as_one_slab_without_a_wall_between_them() {
        // Front and bottom share edge 0-1, so they weld: one copy of each shared corner and no wall grown
        // across the hinge. That is the difference between one slab and two coincident shells, which would
        // render as z-fighting and read as a self-intersection.
        let before = HalfEdgeMesh::create_cube();
        let mut mesh = before.clone();

        let shape = mesh
            .extrude_faces(&[FaceId(0), FaceId(5)], push_front())
            .expect("two faces hinged on one edge should extrude together");

        assert_eq!(
            mesh.vertices.len(),
            14,
            "six corners moved, each copied once"
        );
        assert_eq!(shape.new_vertices.len(), 6);
        assert_eq!(
            shape.side_faces.len(),
            6,
            "eight outer edges less the two at the hinge"
        );
        assert!(
            mesh.boundary_edges().is_empty(),
            "the slab must leave the solid closed where it detached"
        );
        assert_eq!(
            mesh.face_vertices(FaceId(0)).len(),
            4,
            "the lifted face keeps its four corners, welded to its neighbour"
        );
    }

    #[test]
    fn a_repeated_face_in_the_selection_moves_once() {
        let mut once = HalfEdgeMesh::create_cube();
        let mut twice = HalfEdgeMesh::create_cube();

        let single = once
            .extrude_face(FaceId(0), push_front())
            .expect("one face should extrude");
        let doubled = twice
            .extrude_faces(&[FaceId(0), FaceId(0)], push_front())
            .expect("naming the same face twice is the same extrusion, not two");

        assert_eq!(single.side_faces.len(), doubled.side_faces.len());
        assert_eq!(shape(&once), shape(&twice));
    }

    #[test]
    fn extruding_the_face_of_an_open_quad_leaves_the_base_open() {
        // The other half of the closed-solid rule: on a sheet there is nothing behind the face, so the
        // walls grow from all four of its edges and the bottom stays a four-edge hole. Refusing this would
        // make it impossible to extrude a plane, which is the first thing anyone does with one.
        let mut mesh = open_quad();
        let shape = mesh
            .extrude_face(FaceId(0), Vector3::new(0.0, 0.0, 1.0))
            .expect("a lone quad has four free edges to grow from");

        assert_eq!(mesh.faces.len(), 5, "the lifted sheet plus four walls");
        assert_eq!(shape.side_faces.len(), 4);
        assert_eq!(
            mesh.boundary_edges().len(),
            4,
            "the rim the sheet left behind is still open"
        );
        assert_eq!(mesh.half_edges.len(), 20, "four lifted, sixteen new");
    }
}
