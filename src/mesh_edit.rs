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
    face_area, EdgeId, Face, FaceId, HalfEdge, HalfEdgeId, HalfEdgeMesh, MeshVertex,
    TopologyErrors, VertexId, MIN_FACE_AREA,
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
/// edge and every one of them becomes a degenerate face with no area, which `validate` now rejects - so
/// this is not the only thing standing between a zero offset and a broken mesh, it is what makes the
/// refusal arrive as "an extrusion needs a non-zero offset" instead of a list of faces with no area.
const MIN_EXTRUDE_OFFSET: f32 = 1e-6;

/// How far two faces must tilt apart for [`HalfEdgeMesh::bevel_edge`] to find a facet worth opening.
///
/// Compared against the magnitude of the cross product of their unit normals, which is the sine of the
/// angle between them - so unlike an area threshold it does not care how big the model is, and it refuses
/// a hinge that is flat rather than a bevel that is small. A generous fraction of a degree: a corner that
/// visibly creased is a corner the user can see they are beveling, and the thin facet it produces is real
/// geometry they can still select and undo.
const MIN_BEVEL_DIHEDRAL_SINE: f32 = 1e-4;

/// What [`HalfEdgeMesh::split_face`] created: `face` cut in two along one new edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitFace {
    /// The face the caller named. It keeps its id and now bounds the half that runs `b` back to `a`.
    pub original_face: FaceId,
    /// The appended half, holding the chain that ran `a` forward to `b`.
    pub new_face: FaceId,
    /// The two half-edges of the new edge, as `(b to a on new_face, a to b on original_face)`. Both
    /// name the same [`EdgeId`], which is the edge that was inserted.
    pub half_edges: [HalfEdgeId; 2],
}

/// What [`HalfEdgeMesh::insert_edge_loop`] created.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EdgeLoopCut {
    /// The edges the ring passed through, in ring order. None of them exist any more: each was split in
    /// two, so a selection held across the cut cannot keep naming them.
    pub cut_edges: Vec<EdgeId>,
    /// One vertex inserted at the middle of each cut edge, in the same order as `cut_edges`.
    pub new_vertices: Vec<VertexId>,
    /// The ring that was inserted - one edge across each face the loop crossed. These are live, and they
    /// are what the command leaves selected, because they are the thing the user asked for.
    pub new_edges: Vec<EdgeId>,
    /// The faces appended by those cuts, in the same order as `new_edges`.
    pub new_faces: Vec<FaceId>,
}

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

/// What [`HalfEdgeMesh::bevel_edge`] created.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Beveled {
    /// The flat facet that now sits where the edge did.
    pub face: FaceId,
    /// Its corners in the order they are wound: the two cut points on the near face, then the two on the
    /// far face, so the order reads around the facet rather than jumping across it.
    pub corners: [VertexId; 4],
    /// The two faces the edge bounded. Each keeps its corner count, having surrendered a strip of itself
    /// to the facet.
    pub shortened_faces: [FaceId; 2],
    /// The two faces that met the edge at its endpoints. Each gained a corner, taking the shape of the
    /// cut: a cube's neighbour becomes a pentagon.
    pub trimmed_faces: [FaceId; 2],
    /// The two vertices that were created, one at each end of the edge, on the far side of it. The two on
    /// the near side are the edge's own endpoints, relocated - see the note on reuse in
    /// [`HalfEdgeMesh::bevel_edge`].
    pub new_vertices: [VertexId; 2],
}

/// What [`HalfEdgeMesh::delete_faces`] removed, and where everything that survived ended up.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Deleted {
    /// The faces retired, in the order they were asked for and with repeats dropped.
    pub faces: Vec<FaceId>,
    /// Half-edges that went with them. Every half-edge belongs to a face, so none can outlive one.
    pub half_edges: usize,
    /// Corners that belonged to nothing else and were dropped alongside them.
    pub vertices: usize,
    /// Where each surviving component moved, so whatever holds ids can move them the same way.
    pub remap: IdRemap,
}

/// What [`HalfEdgeMesh::fill_holes`] closed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Filled {
    /// The cap grown across each opening, one per rim, in rim order. These are the only new ids an
    /// append creates, so everything handed out before the fill still names what it did.
    pub faces: Vec<FaceId>,
    /// How many corners each cap took, alongside the face it describes - echoed so a three-corner closing
    /// is not reported as a quad.
    pub corners: Vec<usize>,
}

/// Where each component went when a mesh was rebuilt from a subset of its own records.
///
/// Deleting is the one edit that cannot be done by appending: records genuinely leave the `Vec`s and
/// every id above a removed one shifts down. A `usize` picked up beforehand - by the selection, above all
/// others - would then name a *different* component, quietly, and the viewport would highlight the wrong
/// face with nothing anywhere to follow. So the operation returns the correspondence rather than promising
/// ids did not move, and leaves `Scene` to apply it.
///
/// An entry of `None` means the component is gone, which is what a caller needs in order to *drop* it
/// rather than relocate it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IdRemap {
    vertices: Vec<Option<VertexId>>,
    faces: Vec<Option<FaceId>>,
}

impl IdRemap {
    /// Where a corner went, or `None` if it went with the faces that owned it.
    pub fn vertex(&self, vertex: VertexId) -> Option<VertexId> {
        self.vertices.get(vertex.0).copied().flatten()
    }

    /// Where a face went, or `None` if it was one of the ones deleted.
    pub fn face(&self, face: FaceId) -> Option<FaceId> {
        self.faces.get(face.0).copied().flatten()
    }

    /// Where an edge went, following the two corners it names.
    ///
    /// `Some` here means *both corners still exist*, not that the edge does: the faces that ran it may
    /// have been the ones deleted, leaving two survivors with nothing between them. Numbering is all a
    /// remap can answer; the caller confirms existence against the rebuilt mesh, which is what `Scene`'s
    /// selection clip already does for every edge it holds.
    pub fn edge(&self, edge: EdgeId) -> Option<EdgeId> {
        let first = self.vertex(VertexId(edge.0))?;
        let second = self.vertex(VertexId(edge.1))?;
        Some(EdgeId::between(first, second))
    }
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

    /// Cuts `face` in two along a new edge joining two of its existing corners, `a` and `b`, and appends
    /// the half that comes off the cut. This is what an edge loop uses to open a face; nothing is removed.
    ///
    /// The face the caller named keeps its id and bounds the half running from `b` back round to `a`; the
    /// appended face takes the chain that ran from `a` forward to `b`. Both halves inherit the original
    /// winding untouched, because each is a contiguous slice of one loop - which is why no corner is
    /// reordered here, and why the renderer's back-face rule keeps agreeing with `validate` afterwards.
    ///
    /// The two corners must not already be joined: the distance between them along the loop, and its
    /// complement, must both be at least 2. That one rule refuses a cut drawn on top of an edge that
    /// exists, refuses the same cut named the other way round, and is the reason a triangle cannot be cut
    /// at all - there is no pair of its corners that isn't already an edge.
    ///
    /// Returns what was created, or every reason the cut cannot happen, having changed nothing.
    pub fn split_face(
        &mut self,
        face: FaceId,
        a: VertexId,
        b: VertexId,
    ) -> Result<SplitFace, TopologyErrors> {
        let (loop_edges, a_at, b_at) = self.split_face_setup(face, a, b)?;

        let mut candidate = self.clone();
        let cut = candidate.splice_face(face, &loop_edges, a_at, b_at);
        install_if_valid(
            candidate,
            self,
            format!(
                "cutting face {} between vertex {} and vertex {} produced topology that does not validate",
                face.0, a.0, b.0
            ),
        )?;
        Ok(cut)
    }

    /// Cuts a ring of faces in two along one new edge per face - Maya's *Insert Edge Loop*. The ring is
    /// the set of edges reached by crossing from an edge to the edge opposite it in each face it enters,
    /// which on the quad grid this app builds means two steps round the loop and gives the parallel,
    /// evenly-spaced cut the tool is named for.
    ///
    /// Nothing is deleted, so this stays on append-only storage like face extrusion: the crossed faces
    /// keep their ids, and so does every edge that was not part of the ring.
    ///
    /// Two kinds of id do die, and both are reported rather than left to be discovered:
    ///
    /// - Each edge the ring passed through was split in two, so it names a vertex pair that no longer
    ///   exists - the caveat [`HalfEdgeMesh::split_edge`] carries, arriving here for a whole ring at once.
    /// - Each crossed face gains two corners, so the faces themselves are unchanged but no longer the
    ///   quads they were until their own cut is applied.
    ///
    /// The new ring edges are returned as `new_edges`, which is what the command leaves selected: they are
    /// the geometry the user asked for, and they are live.
    ///
    /// Quads only, deliberately. "The edge opposite this one" is not a well-defined notion on an n-gon,
    /// and guessing - nearest corner, halfway round - would place cuts the user did not ask for. The ring
    /// therefore stops at a non-quad face, cutting everything it reached on the way, exactly as it stops at
    /// an open boundary. Only a loop that cannot start at all is refused, and it says which face stopped it.
    ///
    /// Returns what was created, or every reason the ring cannot be cut, having changed nothing. A ring
    /// that runs off an open boundary is cut as far as it goes rather than refused: that is what a loop on
    /// an unfinished mesh means, and stopping early loses nothing.
    pub fn insert_edge_loop(&mut self, seed: EdgeId) -> Result<EdgeLoopCut, TopologyErrors> {
        let crossings = edge_loop_crossings(self, seed)?;

        // The edges to open are each crossing's entry edge, plus the final exit edge: on a closed ring that
        // last one is already the first entry, which is why the list is deduplicated rather than counted
        // out. Order is kept, so `cut_edges` reads as the ring.
        let mut cut_edges: Vec<EdgeId> = Vec::new();
        for crossing in &crossings {
            if !cut_edges.contains(&crossing.enters) {
                cut_edges.push(crossing.enters);
            }
        }
        if let Some(last) = crossings.last() {
            if !cut_edges.contains(&last.exits) {
                cut_edges.push(last.exits);
            }
        }

        // Every edge of the ring is opened before any face is cut, so each crossed face has both of its new
        // corners in hand when its turn comes. Opening one cannot disturb another: the list has no repeats,
        // and `split_edge` rewires only the half-edges of the edge it is given.
        let mut candidate = self.clone();
        let mut new_vertices = Vec::with_capacity(cut_edges.len());
        let mut midpoints: HashMap<(usize, usize), VertexId> = HashMap::new();
        for &edge in &cut_edges {
            let Some(position) = candidate.midpoint(edge) else {
                return Err(vec![format!(
                    "the ring names the edge between vertex {} and vertex {}, but one of those vertices is \
                     not on the mesh",
                    edge.0, edge.1
                )]);
            };
            match candidate.split_edge(edge, position) {
                Ok(split) => {
                    midpoints.insert((edge.0, edge.1), split.vertex);
                    new_vertices.push(split.vertex);
                }
                Err(mut errors) => {
                    errors.insert(
                        0,
                        format!(
                            "the edge loop could not open the edge between vertex {} and vertex {}",
                            edge.0, edge.1
                        ),
                    );
                    return Err(errors);
                }
            }
        }

        // Then each crossed face is cut between the midpoints it now owns. Both are corners of that face by
        // construction - the walk only names edges of the face it is crossing - so a refusal here means the
        // walk and the rewiring disagree, and the message says so rather than pretending the arguments were
        // sound.
        let mut new_edges = Vec::with_capacity(crossings.len());
        let mut new_faces = Vec::with_capacity(crossings.len());
        for crossing in &crossings {
            let (Some(a), Some(b)) = (
                midpoints.get(&(crossing.enters.0, crossing.enters.1)),
                midpoints.get(&(crossing.exits.0, crossing.exits.1)),
            ) else {
                return Err(vec![format!(
                    "cutting face {} for the edge loop found no midpoint on the edges the ring walked \
                     through it by; the walk and the mesh disagree about its corners",
                    crossing.face.0
                )]);
            };

            match candidate.split_face(crossing.face, *a, *b) {
                Ok(cut) => {
                    new_edges.push(EdgeId::between(*a, *b));
                    new_faces.push(cut.new_face);
                }
                Err(mut errors) => {
                    errors.insert(
                        0,
                        format!("the edge loop could not cut face {}", crossing.face.0),
                    );
                    return Err(errors);
                }
            }
        }

        // Every step validated its own result before accepting it, so the finished candidate is a mesh
        // that already holds. Installing by assignment is what makes the cut visible: without it this
        // function would report the ring it planned rather than the one it made.
        *self = candidate;

        Ok(EdgeLoopCut {
            cut_edges,
            new_vertices,
            new_edges,
            new_faces,
        })
    }

    /// The loop of `face` together with where `a` and `b` sit in it, or the reason they name no cut.
    ///
    /// Each check here makes the next one measurable rather than merely unwelcome - the corner positions
    /// cannot be looked up without a walkable loop, and the adjacency test cannot be run without two
    /// corners that were found - so these report singly, unlike [`HalfEdgeMesh::append_face`], whose
    /// objections are independent of one another.
    fn split_face_setup(
        &self,
        face: FaceId,
        a: VertexId,
        b: VertexId,
    ) -> Result<(Vec<HalfEdgeId>, usize, usize), TopologyErrors> {
        if a == b {
            return Err(vec![format!(
                "a cut needs two different corners, vertex {} was named for both ends of it",
                a.0
            )]);
        }
        if face.0 >= self.faces.len() {
            return Err(vec![format!("face {} does not exist on this mesh", face.0)]);
        }
        let Some(loop_edges) = self.face_walk(face) else {
            return Err(vec![format!(
                "face {} does not bound a closed loop of at least three corners, so it cannot be cut",
                face.0
            )]);
        };

        let Some(a_at) = self.corner_at(&loop_edges, a) else {
            return Err(vec![format!(
                "vertex {} is not a corner of face {}",
                a.0, face.0
            )]);
        };
        let Some(b_at) = self.corner_at(&loop_edges, b) else {
            return Err(vec![format!(
                "vertex {} is not a corner of face {}",
                b.0, face.0
            )]);
        };

        let count = loop_edges.len();
        let forward = (b_at + count - a_at) % count;
        if forward < 2 || count - forward < 2 {
            return Err(vec![format!(
                "vertices {} and {} are already joined by an edge of face {}, so there is nothing to cut \
                 between them",
                a.0, b.0, face.0
            )]);
        }

        Ok((loop_edges, a_at, b_at))
    }

    /// Where in a face loop the half-edge leaving `corner` sits, or `None` when the loop has no such
    /// corner. Indexing is safe because [`HalfEdgeMesh::face_walk`] only yields entries it resolved.
    fn corner_at(&self, loop_edges: &[HalfEdgeId], corner: VertexId) -> Option<usize> {
        loop_edges
            .iter()
            .position(|&edge| self.half_edges[edge.0].origin == corner)
    }

    /// The rewire [`HalfEdgeMesh::split_face`] performs on its clone once the setup has agreed the cut is
    /// possible: two half-edges appended, one each way along the new edge, and the chain between the two
    /// corners moved onto a freshly appended face.
    ///
    /// Twinship is all that fuses two faces along an edge here, so making the two new half-edges twins of
    /// each other *is* the cut; the four links at the ends of each chain are what have to notice. That is
    /// the whole cost, constant however many corners the face had, and no corner's own record is touched,
    /// which is why both halves keep the winding the caller started with.
    ///
    /// `a_at` and `b_at` come from [`HalfEdgeMesh::split_face_setup`], which already proved both halves end
    /// up with at least three edges. That proof is what makes the four links four *distinct* half-edges, so
    /// no assignment below overwrites another.
    fn splice_face(
        &mut self,
        face: FaceId,
        loop_edges: &[HalfEdgeId],
        a_at: usize,
        b_at: usize,
    ) -> SplitFace {
        let count = loop_edges.len();
        let new_face = FaceId(self.faces.len());
        let a_to_b = HalfEdgeId(self.half_edges.len());
        let b_to_a = HalfEdgeId(self.half_edges.len() + 1);

        let a = self.half_edges[loop_edges[a_at].0].origin;
        let b = self.half_edges[loop_edges[b_at].0].origin;
        // The edge arriving at each corner and the one leaving it: the chord slots between the pair, once
        // per half.
        let into_a = loop_edges[(a_at + count - 1) % count];
        let from_a = loop_edges[a_at];
        let into_b = loop_edges[(b_at + count - 1) % count];
        let from_b = loop_edges[b_at];

        // Everything from `a` up to but not including `b` becomes the new face's loop; the rest of the
        // original loop is what the face the caller named keeps.
        let mut step = a_at;
        while step != b_at {
            self.half_edges[loop_edges[step].0].face = Some(new_face);
            step = (step + 1) % count;
        }

        // The face the caller named can no longer start from the half-edge it recorded, because that one
        // just moved. `from_b` is the first edge of the chain it keeps, so repointing the cache there is
        // what leaves both halves walkable - and `face_walk` reads through that cache, so skipping this
        // would strand the original face even though every one of its own links is correct.
        self.faces[face.0].half_edge = Some(from_b);

        self.faces.push(Face {
            half_edge: Some(b_to_a),
        });
        self.half_edges.push(HalfEdge {
            origin: a,
            twin: Some(b_to_a),
            next: Some(from_b),
            prev: Some(into_a),
            face: Some(face),
        });
        self.half_edges.push(HalfEdge {
            origin: b,
            twin: Some(a_to_b),
            next: Some(from_a),
            prev: Some(into_b),
            face: Some(new_face),
        });

        self.half_edges[into_a.0].next = Some(a_to_b);
        self.half_edges[from_b.0].prev = Some(a_to_b);
        self.half_edges[into_b.0].next = Some(b_to_a);
        self.half_edges[from_a.0].prev = Some(b_to_a);

        SplitFace {
            original_face: face,
            new_face,
            half_edges: [b_to_a, a_to_b],
        }
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

    /// As [`HalfEdgeMesh::half_edge_on`], but only a half-edge that actually bounds a face. An edge with no
    /// face on either side has nothing for a loop to cross, so it cannot seed one.
    fn face_half_edge_on(&self, edge: EdgeId) -> Option<HalfEdgeId> {
        (0..self.half_edges.len())
            .map(HalfEdgeId)
            .find(|&half_edge| {
                self.edge_id_of(half_edge) == Some(edge)
                    && self.half_edges[half_edge.0].face.is_some()
            })
    }

    /// The middle of `edge`, for dropping a vertex into it. `None` when either end is not on the mesh,
    /// which is what a ring named against a stale document looks like.
    fn midpoint(&self, edge: EdgeId) -> Option<Point3<f32>> {
        let a = self.vertices.get(edge.0)?.position;
        let b = self.vertices.get(edge.1)?.position;
        Some(Point3::new(
            (a.x + b.x) * 0.5,
            (a.y + b.y) * 0.5,
            (a.z + b.z) * 0.5,
        ))
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
        if let Some(reason) = self.degenerate_wall_blocker(&walls, offset) {
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

    /// Why this offset would stand a wall on an edge it runs parallel to, if it would.
    ///
    /// A wall is the quad `[a, b, b + offset, a + offset]`, so its area is the cross product of the base
    /// edge and the offset: push straight along an edge and that wall flattens to a line. `validate` now
    /// rejects a face with no area, which is why extruding the cube's front *and* bottom faces along +z
    /// used to fail - and it does still catch it, but as "face 10 encloses no area", which never mentions
    /// the direction that caused it. Refusing here says which edge, before a vertex or a face is created.
    ///
    /// The area is measured exactly the way `validate` measures it, rather than against a fresh tolerance,
    /// so the two cannot disagree about where the boundary sits.
    fn degenerate_wall_blocker(
        &self,
        walls: &[(VertexId, VertexId)],
        offset: Vector3<f32>,
    ) -> Option<String> {
        walls
            .iter()
            .find(|&&(near, far)| {
                let near = self.vertices[near.0].position;
                let far = self.vertices[far.0].position;
                face_area(&[near, far, far + offset, near + offset]) < MIN_FACE_AREA
            })
            .map(|&(near, far)| {
                format!(
                    "the extrusion offset ({}, {}, {}) runs along the edge from vertex {} to vertex {}, so \
                     the wall standing on that edge would have no area; extrude the faces one at a time, or \
                     pick a selection whose edges do not point that way",
                    offset.x, offset.y, offset.z, near.0, far.0
                )
            })
    }

    /// Trims one edge, sliding the two faces that bounded it apart and filling the gap with a new flat
    /// facet - Maya's *Bevel*. `amount` is the fraction of each adjoining edge the cut runs, so `0.25`
    /// leaves the facet a quarter of the way out from where the edge was, at both ends and on both sides.
    ///
    /// Four faces are involved and none is removed, which is what keeps this on append-only storage like
    /// extrude and edge loop: the facet is new, the two faces that bounded the edge keep their corner
    /// counts, and the two that met the edge at its endpoints gain one corner each. Beveling a cube edge
    /// takes the solid from 8 vertices, 12 edges and 6 faces to 10, 15 and 7, leaving two neighbours as
    /// pentagons.
    ///
    /// Every id survives, which sounds like the good news and is the opposite. The `EdgeId` handed in is
    /// reused: both of its endpoints become corners of the new facet, so the very same pair still resolves,
    /// but to the facet's near edge, offset from where the original ran. A stale `EdgeId` after a
    /// `split_edge` at least refuses to resolve; a stale one after a bevel answers with different geometry,
    /// so a selection held across this operation must be rebuilt from `corners` rather than trusted.
    ///
    /// The edge's own endpoints are relocated rather than replaced, each sliding into the face called the
    /// near one below, and the far face gains a fresh vertex in place of each. That is why only two
    /// vertices appear in `new_vertices` where four corners are cut.
    ///
    /// Three things are refused before a pointer moves, each naming what it refused:
    ///
    /// - an edge the mesh does not have, or one with only one face behind it. Beveling a boundary edge
    ///   means building a rim around the opening, which is a different construction from this one.
    /// - an endpoint where more than one other face also meets, which is any corner of degree four or
    ///   higher. Maya splits such a corner once per diverging face; here it would strand vertices, so the
    ///   corner is refused by number along with its degree.
    /// - an `amount` outside `(0, 1)`, or one small enough that the facet would enclose no area.
    ///
    /// Returns what was created, or every reason the bevel cannot happen, having changed nothing.
    pub fn bevel_edge(&mut self, edge: EdgeId, amount: f32) -> Result<Beveled, TopologyErrors> {
        let setup = self.bevel_setup(edge, amount)?;

        let mut candidate = self.clone();
        let shape = candidate.install_bevel(setup);
        install_if_valid(
            candidate,
            self,
            format!(
                "beveling the edge between vertex {} and vertex {} produced topology that does not validate",
                edge.0, edge.1
            ),
        )?;
        Ok(shape)
    }

    /// Reads a mesh for [`HalfEdgeMesh::bevel_edge`], refusing by name whatever the bevel cannot use.
    ///
    /// Every pointer read is `Option`-safe, as elsewhere on the write path: a mesh that arrives broken has
    /// to be refused with its damage listed, not crashed on while the bevel is still looking for corners.
    fn bevel_setup(&self, edge: EdgeId, amount: f32) -> Result<ResolvedBevel, TopologyErrors> {
        let mut reasons = TopologyErrors::new();

        if !amount.is_finite() || amount <= 0.0 || amount >= 1.0 {
            reasons.push(format!(
                "a bevel cuts {} of the way along each edge meeting the one it trims, so the amount has to \
                 sit between 0 and 1",
                amount
            ));
        }

        // Both directions of the edge, since the facet has to be wound against each of them. Scanning for
        // one of each rather than taking the first match is what makes this agree with the mesh whichever
        // way round it happens to run the pair.
        let a = VertexId(edge.0);
        let b = VertexId(edge.1);
        let (mut near, mut far) = (None, None);
        for index in 0..self.half_edges.len() {
            let found = HalfEdgeId(index);
            if self.edge_id_of(found) != Some(edge) {
                continue;
            }
            if self.half_edges[index].origin == a {
                near = Some(found);
            } else {
                far = Some(found);
            }
        }
        let (Some(near), Some(far)) = (near, far) else {
            // Worth splitting: a boundary edge really does have a half-edge running it, and telling the
            // user there is no such edge would send them looking for a selection bug that isn't there.
            let one_side_open = near.is_some() || far.is_some();
            return Err(vec![if one_side_open {
                format!(
                    "only one face bounds the edge between vertex {} and vertex {}, so there is nothing to \
                     slide apart; beveling an edge at the rim of an opening has to build a rim around it, \
                     which is not implemented",
                    edge.0, edge.1
                )
            } else {
                format!(
                    "no half edge runs between vertex {} and vertex {}, so there is nothing to bevel",
                    edge.0, edge.1
                )
            }]);
        };

        // The four corners the cut runs along, each one link round its own loop from the edge. A link that
        // is missing or names nothing means the mesh is damaged, and there is no direction to cut without
        // guessing at it.
        let damaged = |where_: &str| -> TopologyErrors {
            vec![format!(
                "the {where_} does not continue round past this edge, so the mesh needs repairing before it \
                 can be beveled"
            )]
        };
        let leaving_for = |from: HalfEdgeId| -> Option<VertexId> {
            let onward = self.half_edges[from.0].next?;
            self.half_edges.get(onward.0).map(|edge| edge.origin)
        };
        let arriving_from = |into: HalfEdgeId| -> Option<VertexId> {
            let back = self.half_edges[into.0].prev?;
            self.half_edges.get(back.0).map(|edge| edge.origin)
        };

        let Some(near_face) = self.half_edges[near.0].face else {
            return Err(damaged("near side of this edge"));
        };
        let Some(far_face) = self.half_edges[far.0].face else {
            return Err(damaged("far side of this edge"));
        };
        let Some(d) = arriving_from(near) else {
            return Err(damaged("near face"));
        };
        // Two links round from `near`, not one: `origin(near.next)` is merely `near`'s destination, which
        // is `b`, and the cut at `b` runs towards the corner *after* it. The far face's `x` below reads the
        // same shape correctly, which is what makes this one easy to miss by symmetry.
        let Some(c) = self.half_edges[near.0].next.and_then(leaving_for) else {
            return Err(damaged("near face"));
        };
        let Some(far_from_a) = self.half_edges[far.0].next else {
            return Err(damaged("far face"));
        };
        let Some(x) = leaving_for(far_from_a) else {
            return Err(damaged("far face"));
        };
        let Some(y) = arriving_from(far) else {
            return Err(damaged("far face"));
        };

        if near_face == far_face {
            reasons.push(format!(
                "both sides of the edge between vertex {} and vertex {} belong to face {}, which is not a \
                 solid this bevel can trim",
                edge.0, edge.1, near_face.0
            ));
        }

        // What actually collapses a facet is the angle between the two faces, not the amount: at any amount
        // whatever, two faces lying in one plane produce a strip folded shut. Unit normals make the cross
        // product's magnitude the sine of that angle, so this is scale-free - it refuses the same flat pair
        // whether the model is in millimetres or kilometres - and unlike an area test it cannot be satisfied
        // by pushing a small bevel along a long edge. It runs here rather than once the corners have been
        // resolved because a flat seam is usually also a busy corner, and the hinge is the reason that
        // applies; `validate` judges topology, so nothing downstream would ever report a folded facet.
        let dihedral_sine = self
            .face_normal(near_face)
            .cross(self.face_normal(far_face))
            .magnitude();
        if dihedral_sine < MIN_BEVEL_DIHEDRAL_SINE {
            reasons.push(format!(
                "faces {} and {} lie flat against each other, so a bevel between them has no corner to open \
                 into a facet",
                near_face.0, far_face.0
            ));
        }

        let neighbour_at_a = self.bevel_neighbour(a, near_face, far_face, &mut reasons);
        let neighbour_at_b = self.bevel_neighbour(b, near_face, far_face, &mut reasons);

        // The three links to open, each found by the edge it runs rather than by stepping round a loop: the
        // neighbour faces are reached from their corner, not from the edge being trimmed, so nothing here
        // depends on which way round either of them happens to be stored.
        let mut openings: [Option<(FaceId, HalfEdgeId)>; 3] = [None; 3];
        for (slot, face, between) in [
            (0usize, neighbour_at_a, EdgeId::between(a, x)),
            (1, neighbour_at_b, EdgeId::between(b, c)),
            (2, neighbour_at_b, EdgeId::between(b, y)),
        ] {
            let Some(face) = face else { continue };
            match self.face_edge_in(face, between) {
                Some(found) => openings[slot] = Some((face, found)),
                None => reasons.push(format!(
                    "face {} does not bound the edge between vertex {} and vertex {}, so the mesh needs \
                     repairing before this bevel can open it",
                    face.0, between.0, between.1
                )),
            }
        }
        // Carrying the face alongside each opening is what proves both neighbours exist: an opening is only
        // filled for a face that was found, so this one check covers all three links and both corners, and
        // anything the two `bevel_neighbour` calls objected to is already in `reasons`.
        let (
            Some((neighbour_at_a, open_before_a)),
            Some((neighbour_at_b, open_before_b)),
            Some((_, hand_over_b)),
        ) = (openings[0], openings[1], openings[2])
        else {
            return Err(reasons);
        };

        let at = |corner: VertexId| self.vertices[corner.0].position;
        let moved = [slide(at(a), at(d), amount), slide(at(b), at(c), amount)];
        let created = [slide(at(a), at(x), amount), slide(at(b), at(y), amount)];

        // The flat-hinge check ran above, before any corner was resolved.

        if !reasons.is_empty() {
            return Err(reasons);
        }

        Ok(ResolvedBevel {
            a,
            b,
            near_face,
            far_face,
            far_from_a,
            far_from_b: far,
            neighbour_at_a,
            open_before_a,
            neighbour_at_b,
            open_before_b,
            hand_over_b,
            moved,
            created,
        })
    }

    /// The one face meeting `corner` besides the two the bevel is separating, refusing the corner when that
    /// is not exactly one face.
    ///
    /// A corner of degree three leaves a single neighbour to trim and one unambiguous edge to cut along.
    /// More than that is a corner where several faces diverge, which Maya handles by splitting the vertex
    /// once per face - a construction this build does not have, so picking one neighbour to trim would move
    /// geometry nobody asked for. None is a boundary: the bevel would have to close a hole rather than open
    /// a facet, and that is a different operation again.
    fn bevel_neighbour(
        &self,
        corner: VertexId,
        near_face: FaceId,
        far_face: FaceId,
        reasons: &mut TopologyErrors,
    ) -> Option<FaceId> {
        let mut others: Vec<FaceId> = Vec::new();
        for edge in self.half_edges.iter().filter(|edge| edge.origin == corner) {
            let Some(face) = edge.face else { continue };
            if face != near_face && face != far_face && !others.contains(&face) {
                others.push(face);
            }
        }

        match others.as_slice() {
            [only] => Some(*only),
            [] => {
                reasons.push(format!(
                    "vertex {} sits on the boundary with no third face to trim, and a bevel of an edge that \
                     ends at an opening has to build a rim around it, which is not implemented",
                    corner.0
                ));
                None
            }
            many => {
                reasons.push(format!(
                    "vertex {} is shared by {} faces, so the bevel cannot tell which neighbour to trim; \
                     Maya splits a corner of that degree once per face, which is not implemented",
                    corner.0,
                    many.len() + 2
                ));
                None
            }
        }
    }

    /// Performs the bevel [`HalfEdgeMesh::bevel_setup`] resolved, on the clone it will be validated on.
    ///
    /// Nothing here can fail: every id and position was checked in the read pass, and each link written is
    /// one that pass promised existed. Order matters in exactly one place - the facet goes on last, because
    /// [`HalfEdgeMesh::push_face`] finds a twin by looking for the reverse-directed edge already in the
    /// mesh, and those four only come into being as the three faces around it are rewritten.
    fn install_bevel(&mut self, bevel: ResolvedBevel) -> Beveled {
        let ResolvedBevel {
            a,
            b,
            near_face,
            far_face,
            far_from_a,
            far_from_b,
            neighbour_at_a,
            open_before_a,
            neighbour_at_b,
            open_before_b,
            hand_over_b,
            moved,
            created,
        } = bevel;

        let a_far = self.add_vertex(created[0]);
        let b_far = self.add_vertex(created[1]);
        self.vertices[a.0].position = moved[0];
        self.vertices[b.0].position = moved[1];

        // The far face gives up both its corners, and two origins are the whole of it: a half-edge carries
        // no destination, so rewriting the pair turns `b -> a` into `b_far -> a_far` without that half-edge
        // itself being touched, and every edge it shares with its own neighbours follows along.
        self.half_edges[far_from_a.0].origin = a_far;
        self.half_edges[far_from_b.0].origin = b_far;

        // The neighbour at `a` gains its corner on the far side of the cut, which also turns the edge that
        // arrived at `a` from `x` into the one arriving at `a_far`.
        self.insert_corner(open_before_a, a_far);

        // At `b` the neighbour keeps the original corner and hands `y` across instead, so the half-edge
        // leaving `b` is rewritten first and the new corner opened behind it. That mirror is why the facet
        // cannot simply be wound the other way at one end.
        self.half_edges[hand_over_b.0].origin = b_far;
        self.insert_corner(open_before_b, b);

        // The facet, and the only face the operation adds. All four of its twins now exist, so one
        // `push_face` fuses it to the near face, the far face and both neighbours.
        let face = self.push_face(&[b, a, a_far, b_far]);

        // Relocating `a` and `b` can leave either naming the half-edge that just moved to `b_far`, and the
        // new corners need naming too, so all four are re-pointed once the facet is on.
        self.repair_outgoing_edges(&HashSet::from([a, b, a_far, b_far]));

        Beveled {
            face,
            corners: [b, a, a_far, b_far],
            shortened_faces: [near_face, far_face],
            trimmed_faces: [neighbour_at_a, neighbour_at_b],
            new_vertices: [a_far, b_far],
        }
    }

    /// Opens `arriving` to take one more corner: it keeps its own origin and gains a new destination, the
    /// half-edge inserted carries `origin` on to wherever `arriving` was heading, and both belong to
    /// `arriving`'s face.
    ///
    /// The twin is left unset for the caller to fuse, which is how a facet's four edges get linked in a
    /// single [`HalfEdgeMesh::push_face`] instead of four separate passes - and why this belongs to a
    /// primitive that finishes the job rather than standing alone as an edit.
    fn insert_corner(&mut self, arriving: HalfEdgeId, origin: VertexId) -> HalfEdgeId {
        let face = self.half_edges[arriving.0].face;
        let onward = self.half_edges[arriving.0].next;
        let inserted = HalfEdgeId(self.half_edges.len());

        self.half_edges.push(HalfEdge {
            origin,
            twin: None,
            next: onward,
            prev: Some(arriving),
            face,
        });
        self.half_edges[arriving.0].next = Some(inserted);
        if let Some(following) = onward.and_then(|onward| self.half_edges.get_mut(onward.0)) {
            following.prev = Some(inserted);
        }

        inserted
    }

    /// The half-edge bounding `face` that runs `between`, whichever way round, for a mesh that may not have
    /// been validated yet.
    fn face_edge_in(&self, face: FaceId, between: EdgeId) -> Option<HalfEdgeId> {
        (0..self.half_edges.len())
            .map(HalfEdgeId)
            .find(|&half_edge| {
                self.half_edges[half_edge.0].face == Some(face)
                    && self.edge_id_of(half_edge) == Some(between)
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

    /// Retires the named faces and everything that existed only to bound them, leaving an opening where
    /// they were. Nothing is closed up: this is Maya's *Delete*, not a hole fill.
    ///
    /// This is the one edit that cannot be done by appending, and the reason the append-only rule held for
    /// every operation before it is that none of them reduced the number of records. Removing a face has to
    /// remove its half-edges, and the corners those half-edges were the last to claim - so two options
    /// existed. Slots could be poisoned in place, keeping every id ever handed out valid at the cost of
    /// `Vec`s that never shrink and a `validate` that must learn which holes are deliberate; that is what
    /// `slotmap`/generational ids buy, and they are still not a dependency. Instead the mesh is **rebuilt
    /// densely** from the records that survive, and the operation reports where everything went as an
    /// [`IdRemap`]. Ids therefore *do* move, which is why `Deleted` carries the remap rather than the
    /// operation being quiet, and why the mesh stays cheap to walk: every query in `mesh.rs` still assumes
    /// no gaps, and `picking`/`overlay`/`selection` still hold plain `usize`s across frames.
    ///
    /// Deleting the whole mesh is allowed and yields an empty one, which validates - an empty document is a
    /// state a user can reach and undo out of, not a broken one.
    ///
    /// Returns what went and where the rest landed, or every reason nothing can be deleted, having changed
    /// nothing.
    pub fn delete_faces(&mut self, faces: &[FaceId]) -> Result<Deleted, TopologyErrors> {
        let mut reasons = TopologyErrors::new();
        let mut removed: Vec<FaceId> = Vec::with_capacity(faces.len());
        let mut marked: HashSet<FaceId> = HashSet::new();
        for &face in faces {
            if face.0 >= self.faces.len() {
                reasons.push(format!("face {} does not exist on this mesh", face.0));
            } else if marked.insert(face) {
                removed.push(face);
            }
        }
        if !reasons.is_empty() {
            return Err(reasons);
        }
        if removed.is_empty() {
            return Err(vec!["there is no face named to delete".to_string()]);
        }

        let mut keep_face = vec![true; self.faces.len()];
        for &face in &removed {
            keep_face[face.0] = false;
        }

        // Counted off the candidate before it moves, because `install_if_valid` consumes it either way and
        // a refusal has to be able to report nothing at all rather than a half-measured deletion.
        let (candidate, remap) = self.rebuild_keeping(&keep_face);
        let half_edges = self.half_edges.len() - candidate.half_edges.len();
        let vertices = self.vertices.len() - candidate.vertices.len();

        install_if_valid(
            candidate,
            self,
            format!(
                "deleting face{} {} produced topology that does not validate",
                if removed.len() == 1 { "" } else { "s" },
                removed
                    .iter()
                    .map(|face| face.0.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
        )?;

        Ok(Deleted {
            faces: removed,
            half_edges,
            vertices,
            remap,
        })
    }

    /// Rebuilds this mesh without the faces `keep_face` marks false, densely, and reports where
    /// everything went.
    ///
    /// Nothing here judges whether the request makes sense - [`HalfEdgeMesh::delete_faces`] refuses the
    /// impossible first - so this reads as pure rewrite, which is what keeps it reviewable. The three
    /// correspondences are computed up front in old-index order and every pass below looks a slot up in
    /// them, so there is no place a live record can be copied twice or skipped.
    ///
    /// A dangling id is handed on dangling rather than smoothed into `None`: it is damage for the
    /// validator to report, and rewriting it away here would leave the report nothing to name.
    fn rebuild_keeping(&self, keep_face: &[bool]) -> (HalfEdgeMesh, IdRemap) {
        let mut new_face: Vec<Option<FaceId>> = vec![None; self.faces.len()];
        let mut next = 0;
        for index in 0..self.faces.len() {
            if keep_face[index] {
                new_face[index] = Some(FaceId(next));
                next += 1;
            }
        }

        // Every half-edge belongs to exactly one face - `resolve_ids` refuses one that belongs to none -
        // so a half-edge survives on exactly one condition: its face does.
        let mut new_half_edge: Vec<Option<HalfEdgeId>> = vec![None; self.half_edges.len()];
        let mut next = 0;
        for (slot, edge) in new_half_edge.iter_mut().zip(&self.half_edges) {
            if edge.face.is_some_and(|face| keep_face[face.0]) {
                *slot = Some(HalfEdgeId(next));
                next += 1;
            }
        }

        // A corner survives when some surviving half-edge starts at it, which is the same question
        // `check_vertices` asks - so a corner that stays is always one the mesh can still be walked out
        // from, and no separate sweep for orphans is needed.
        let mut keep_vertex = vec![false; self.vertices.len()];
        for index in 0..self.half_edges.len() {
            if new_half_edge[index].is_some() {
                keep_vertex[self.half_edges[index].origin.0] = true;
            }
        }

        // Then numbered in index order, exactly like the two tables above. Folding this into the scan
        // would hand ids out in half-edge order while the dense `vertices` vector is written in index
        // order, and two corners would trade places: a mesh whose ids all resolve, drawing the wrong
        // shape. Surviving corners are numbered contiguously either way, so nothing else notices - only
        // the geometry does, which is nothing at all.
        let mut new_vertex: Vec<Option<VertexId>> = vec![None; self.vertices.len()];
        let mut next = 0;
        for index in 0..self.vertices.len() {
            if keep_vertex[index] {
                new_vertex[index] = Some(VertexId(next));
                next += 1;
            }
        }

        let moved = |id: HalfEdgeId| match new_half_edge.get(id.0) {
            None => Some(id),
            Some(&fresh) => fresh,
        };

        // The same hand-on-dangling rule for face ids, kept a separate closure because the two tables
        // are different lengths and a half-edge id must never be looked up in the face one.
        let kept = |id: FaceId| match new_face.get(id.0) {
            None => Some(id),
            Some(&fresh) => fresh,
        };

        let mut half_edges: Vec<HalfEdge> =
            Vec::with_capacity(new_half_edge.iter().filter(|slot| slot.is_some()).count());
        let mut first_out: HashMap<usize, HalfEdgeId> = HashMap::new();
        for (edge, slot) in self.half_edges.iter().zip(&new_half_edge) {
            if slot.is_none() {
                continue;
            }
            let fresh = HalfEdgeId(half_edges.len());
            let origin = new_vertex
                .get(edge.origin.0)
                .copied()
                .flatten()
                .unwrap_or(edge.origin);
            first_out.entry(origin.0).or_insert(fresh);

            half_edges.push(HalfEdge {
                origin,
                twin: None,
                next: edge.next.and_then(moved),
                prev: edge.prev.and_then(moved),
                face: edge.face.and_then(kept),
            });
        }

        // Twins are the one link that crosses between faces, so they cannot be copied in the pass above:
        // the partner may not have been written yet. Linking them afterwards, by looking each id up in the
        // same table, is what makes a survivor whose partner's face was deleted come out as a fresh
        // boundary edge rather than a pointer into a hole.
        for (index, slot) in new_half_edge.iter().enumerate() {
            let (Some(fresh), Some(twin)) = (*slot, self.half_edges[index].twin) else {
                continue;
            };
            half_edges[fresh.0].twin = match new_half_edge.get(twin.0) {
                None => Some(twin),
                Some(&survivor) => survivor,
            };
        }

        let vertices: Vec<MeshVertex> = (0..self.vertices.len())
            .filter(|&index| keep_vertex[index])
            .map(|index| MeshVertex {
                position: self.vertices[index].position,
                half_edge: new_vertex[index].and_then(|fresh| first_out.get(&fresh.0).copied()),
            })
            .collect();

        let faces: Vec<Face> = (0..self.faces.len())
            .filter(|&index| keep_face[index])
            .map(|index| Face {
                half_edge: self.faces[index].half_edge.and_then(moved),
            })
            .collect();

        (
            HalfEdgeMesh {
                vertices,
                half_edges,
                faces,
            },
            IdRemap {
                vertices: new_vertex,
                faces: new_face,
            },
        )
    }

    /// Every opening in the mesh, one rim of half-edges per opening.
    ///
    /// A rim is chained corner to corner: from a boundary half-edge, step to the boundary half-edge
    /// *leaving the vertex that one arrives at*. That is the rule that traces the outline of a hole.
    /// Two alternatives do not work here: walking `next` circles the neighbour face that owns the
    /// boundary edge rather than the opening it bounds, and walking the fan around a vertex breaks at
    /// the first boundary edge by definition - which is why [`HalfEdgeMesh::outgoing_half_edges`] scans
    /// instead of walking. Corner-to-corner chaining needs no fan and lands on the opening.
    ///
    /// Two half-edges leaving the same corner along the boundary means two openings meeting at a single
    /// vertex, so the chain has a choice to make there - and the choice decides whether each opening gets
    /// its own cap or one face is laid figure-eight across both. [`HalfEdgeMesh::rim_continuation`]
    /// resolves it the way a walker does, by keeping this opening on its right. Refusing was the earlier
    /// behaviour; it rested on a claim that turns out to be false, that a walk cannot tell two rims apart
    /// at a shared corner, and its price was that one shared corner blocked *both* holes from closing.
    ///
    /// A chain that runs out of boundary edges before arriving home is reported; it is not reachable from
    /// the operations this app has, and it is stated so the next operation does not assume it.
    ///
    /// Rims come back in the order their first half-edge is scanned, each starting at that half-edge, so
    /// two calls on one mesh agree and anything built on them is deterministic.
    pub fn boundary_rims(&self) -> Result<Vec<Vec<HalfEdgeId>>, TopologyErrors> {
        // Which boundary half-edges leave each corner. Most corners bound one opening and hold exactly
        // one, which is the case [`HalfEdgeMesh::rim_continuation`] costs nothing for; a corner holding two
        // is where two openings meet, and the turn rule there decides which chain continues.
        let mut leaving: HashMap<usize, Vec<HalfEdgeId>> = HashMap::new();
        for (index, edge) in self.half_edges.iter().enumerate() {
            if edge.twin.is_none() {
                leaving
                    .entry(edge.origin.0)
                    .or_default()
                    .push(HalfEdgeId(index));
            }
        }

        let mut rims: Vec<Vec<HalfEdgeId>> = Vec::new();
        let mut claimed: HashSet<HalfEdgeId> = HashSet::new();
        for index in 0..self.half_edges.len() {
            let start = HalfEdgeId(index);
            if self.half_edges[index].twin.is_some() || claimed.contains(&start) {
                continue;
            }

            let mut rim: Vec<HalfEdgeId> = vec![start];
            claimed.insert(start);
            let mut current = start;
            loop {
                let destination = self.half_edge_destination(current);
                if destination == self.half_edges[start.0].origin {
                    break;
                }
                // Stopping rather than revisiting a half-edge is what keeps this total on a mesh whose
                // twins arrived mislinked; the closure check below is what turns the stop into a report.
                let Some(next) = self.rim_continuation(current, destination, &leaving, &claimed)
                else {
                    break;
                };
                claimed.insert(next);
                rim.push(next);
                current = next;
            }

            let last = *rim
                .last()
                .expect("a rim always holds the half-edge it started at");
            if self.half_edge_destination(last) != self.half_edges[start.0].origin {
                return Err(vec![format!(
                    "the boundary running from vertex {} does not close on itself, so it bounds nothing \
                     that can be filled",
                    self.half_edges[start.0].origin.0
                )]);
            }
            rims.push(rim);
        }
        Ok(rims)
    }

    /// The boundary half-edge that continues the rim `arrived` is on, out of those leaving `corner`.
    ///
    /// A corner bounding one opening holds one boundary half-edge, so the common case is a lookup with
    /// nothing to decide. Where two openings meet, two leave and one has to be chosen, and the other choice
    /// walks the two rims as a single circuit: the cap would then be one face spanning both holes, whose
    /// corner loop repeats the shared vertex - refused by `append_face_blockers` rather than installed,
    /// which is the only reason guessing here was ever survivable.
    ///
    /// The rule is the sharpest **left** turn about the normal of the face `arrived` lies in, and it
    /// follows from the one fact a rim always carries: a boundary half-edge still belongs to a face, faces
    /// wind counter-clockwise, so that face is on the *left* of the direction travelled and the opening is
    /// on the *right*. Holding on to the same opening for the whole circuit therefore means taking the exit
    /// that turns left hardest relative to where you came from, because every other exit swings the walk
    /// past the corner and hands it to the neighbouring opening.
    ///
    /// Measured about the arriving face's normal rather than an average over the corner's faces, because
    /// the arriving face is guaranteed to have one - and the faces at a pinch are precisely the case where
    /// an average can cancel to nothing, since that is how a corner ends up bounding two openings.
    ///
    /// `None` when nothing unclaimed leaves this corner, which ends the walk so the caller can report an
    /// opening that fails to close on itself.
    fn rim_continuation(
        &self,
        arrived: HalfEdgeId,
        corner: VertexId,
        leaving: &HashMap<usize, Vec<HalfEdgeId>>,
        claimed: &HashSet<HalfEdgeId>,
    ) -> Option<HalfEdgeId> {
        let mut open = leaving
            .get(&corner.0)?
            .iter()
            .copied()
            .filter(|&candidate| !claimed.contains(&candidate));
        let first = open.next()?;
        // Taking the second candidate out by name rather than testing `open.next().is_none()` matters: the
        // test would *consume* it, and the ranking loop below would then see one exit out of two and score
        // the wrong one - which is how two holes that touch at a corner came out as a single figure-eight.
        let Some(second) = open.next() else {
            return Some(first);
        };

        // Two openings meet here, so the ranking normal is the face the walk is standing on.
        let normal = self.half_edges[arrived.0]
            .face
            .map(|face| self.face_normal(face))?;
        let travelled =
            self.vertex_position(corner) - self.vertex_position(self.half_edges[arrived.0].origin);

        let mut best: Option<(f32, HalfEdgeId)> = None;
        for candidate in [first, second].into_iter().chain(open) {
            let onward = self.vertex_position(self.half_edge_destination(candidate))
                - self.vertex_position(corner);
            let Some(turn) = left_turn_about(travelled, onward, normal) else {
                continue;
            };
            let better = best.map(|(best_turn, _)| turn > best_turn).unwrap_or(true);
            if better {
                best = Some((turn, candidate));
            }
        }
        // Flat or degenerate geometry can leave a turn unmeasurable at every exit. Taking the first is no
        // worse than the geometry that made it unmeasurable, and refusing would block both openings again.
        best.map(|(_, candidate)| candidate).or(Some(first))
    }

    /// Grows a face across every opening in the mesh, closing each rim on itself: Maya's *Fill Hole*.
    ///
    /// This is the inverse of [`HalfEdgeMesh::delete_faces`], and where delete is the one edit that
    /// rebuilds, filling is back to **appending**: a cap adds one face and the half-edges bounding it and
    /// links each to the rim half-edge running the same edge in reverse. Nothing leaves the `Vec`s, so
    /// every id handed out beforehand stays valid - which is why this returns no [`IdRemap`] and why
    /// `Scene` needs no remap pass here. Deleting and then filling renumbers for the delete and appends
    /// for the fill; the two are separate undo steps and neither borrows the other's caveat.
    ///
    /// The corners of a cap are the rim's corners **reversed**, and that reversal is the whole trick
    /// rather than a detail. A rim half-edge runs, say, corner 1 to corner 0; the cap must run 0 to 1 for
    /// those two to be twins, and it has to get that right on all of its edges at once. Reversing the
    /// rim's own cyclic order is the single ordering that satisfies every edge simultaneously, and it also
    /// makes the cap wind the way the missing face did - so the new face points along the normal the
    /// surface had instead of folding inward through it.
    ///
    /// That orientation is not something `validate` can be left to catch: it judges twinning, and a
    /// wrongly wound cap twins *correctly* while facing the wrong way, since the rim it closes is the only
    /// thing it touches. What pins it instead is `filling_an_open_box_grows_the_face_that_was_left_off`,
    /// which requires the cap to reproduce the missing face's own corner loop in the same winding and then
    /// measures its normal against the direction it has to point - the one check a flipped cap would fail
    /// and nothing else would.
    ///
    /// Every opening is closed in one clone, so a mesh with two holes takes one undo step, and if any one
    /// of them cannot be capped the operation is refused with all of the reasons and nothing installed.
    /// Capping them one at a time would leave a half-filled mesh behind on the first refusal.
    ///
    /// Returns the caps created and how many corners each took, or every reason nothing can be closed.
    /// A mesh that is already closed is refused: it is the one case where the command cannot do anything
    /// the user could have meant, and a silent success reads as a dead key.
    pub fn fill_holes(&mut self) -> Result<Filled, TopologyErrors> {
        let rims = self.boundary_rims()?;
        if rims.is_empty() {
            return Err(vec![
                "this mesh is closed on every side, so there is no opening to fill".to_string(),
            ]);
        }

        let mut candidate = self.clone();
        let mut filled = Filled::default();
        let mut reasons = TopologyErrors::new();
        for rim in &rims {
            // Read the corners off the clone, not `self`: an earlier cap in this same loop has already
            // twinned part of the boundary, and `append_face_blockers` has to be asked about the mesh the
            // face is actually going into.
            let corners: Vec<VertexId> = rim
                .iter()
                .rev()
                .map(|&half_edge| candidate.half_edges[half_edge.0].origin)
                .collect();

            let blockers = candidate.append_face_blockers(&corners);
            if !blockers.is_empty() {
                reasons.extend(blockers.into_iter().map(|blocker| {
                    format!(
                        "the opening running from vertex {} could not be closed: {}",
                        corners[0].0, blocker
                    )
                }));
                continue;
            }
            let face = candidate.push_face(&corners);
            filled.faces.push(face);
            filled.corners.push(corners.len());
        }
        if !reasons.is_empty() {
            return Err(reasons);
        }

        install_if_valid(
            candidate,
            self,
            format!(
                "filling {} opening{} produced topology that does not validate",
                rims.len(),
                if rims.len() == 1 { "" } else { "s" }
            ),
        )?;

        Ok(filled)
    }
}

/// The counter-clockwise turn from `from` to `to` about `axis`, in `[0, 2π)`, or `None` when either vector
/// has no direction left once flattened onto the plane across `axis`.
///
/// The range is the whole point and not a detail: the correct exit at a pinch is usually a **reflex**
/// turn, because a quad rim meeting another quad rim at a shared corner turns 270° there, and the
/// symmetric `atan2` range reports that as -90°. Ranking by that would put the right answer *below* a
/// wrong 90° and merge the two holes into one rim again. Folding into `[0, 2π)` makes "leftest" a total
/// order that agrees with the hand you walk the opening with.
fn left_turn_about(from: Vector3<f32>, to: Vector3<f32>, axis: Vector3<f32>) -> Option<f32> {
    // Both vectors flattened onto the plane across the axis, since the component along it says nothing
    // about which way the turn goes.
    let flatten = |v: Vector3<f32>| v - axis * v.dot(axis);
    let (a, b) = (flatten(from), flatten(to));
    let (length_a, length_b) = (a.magnitude(), b.magnitude());
    if length_a <= 0.0 || length_b <= 0.0 || !length_a.is_finite() || !length_b.is_finite() {
        return None;
    }

    let mut turn = a.cross(b).dot(axis).atan2(a.dot(b));
    if turn < 0.0 {
        turn += std::f32::consts::TAU;
    }
    Some(turn)
}

/// Everything [`HalfEdgeMesh::bevel_edge`] needs to know before it touches a pointer.
///
/// The names follow one picture: the edge runs `a -> b` along the near face, so the far face runs
/// `b -> a`; `d` and `c` are the near face's corners beyond `a` and `b`, and `x` and `y` are the far
/// face's. The facet is wound `b -> a -> a_far -> b_far`, which runs against the near face on the edge
/// it straddles and against the far face on the edge it replaces - the only cyclic order that does
/// both, and the reason the two ends of the cut are mirror images rather than twins.
struct ResolvedBevel {
    a: VertexId,
    b: VertexId,
    near_face: FaceId,
    far_face: FaceId,
    /// The far face's half-edge leaving each end. Rewriting these two origins is all that face needs,
    /// because a destination is read from the successor's origin rather than stored.
    far_from_a: HalfEdgeId,
    far_from_b: HalfEdgeId,
    /// The face meeting the edge at `a`, and the half-edge in it arriving from `x`. Opening that one
    /// turns `x -> a` into `x -> a_far` and starts the new `a_far -> a`.
    neighbour_at_a: FaceId,
    open_before_a: HalfEdgeId,
    /// The face meeting the edge at `b`, the half-edge in it arriving from `c`, and the one leaving for
    /// `y`. This end mirrors `a`: the neighbour keeps `b` and hands `y` over, so `hand_over_b` starts at
    /// `b_far` before `open_before_b` is opened behind it.
    neighbour_at_b: FaceId,
    open_before_b: HalfEdgeId,
    hand_over_b: HalfEdgeId,
    /// Where `a` and `b` move to, into the near face.
    moved: [Point3<f32>; 2],
    /// Where the two new corners go, into the far face.
    created: [Point3<f32>; 2],
}

/// The point `amount` of the way from `from` toward `toward`, with `amount` a fraction of the distance
/// rather than a distance itself.
///
/// A fraction is what makes one bevel amount behave on a cube and on a house alike: Maya's bevel slider
/// is width, but width against an edge of unknown length needs the edge measured first, and every edge
/// meeting the one being trimmed has its own length. Fractions keep the four corners of a bevel coplanar
/// on any box, which absolute widths do not.
fn slide(from: Point3<f32>, toward: Point3<f32>, amount: f32) -> Point3<f32> {
    Point3::new(
        from.x + (toward.x - from.x) * amount,
        from.y + (toward.y - from.y) * amount,
        from.z + (toward.z - from.z) * amount,
    )
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

/// One face a ring passes through: entered by `enters`, left by `exits`. Those two share no corner, and
/// the cut the loop makes in this face joins the middle of one to the middle of the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Crossing {
    face: FaceId,
    enters: EdgeId,
    exits: EdgeId,
}

/// The faces a ring crosses, in ring order, entering at `seed`.
///
/// Crossing means: from the edge in hand, take the edge two steps round the face - the one sharing no
/// corner with it - then step through that edge's twin into the next face. A ring is recognised by arriving
/// back at a face already crossed; an open mesh simply runs out of twins, which ends the walk without an
/// objection because a loop that stops at a boundary is what a loop on an unfinished mesh means.
///
/// Quads only, and a ring stops where the quads stop. "The edge opposite this one" is not a thing an n-gon
/// has - nearest corner and halfway round give different answers as soon as there are five corners - so the
/// walk ends there rather than guessing, exactly as it ends at an open boundary. A ring that cannot even
/// start is the exception and is refused: cutting nothing is indistinguishable from a dead key.
fn edge_loop_crossings(mesh: &HalfEdgeMesh, seed: EdgeId) -> Result<Vec<Crossing>, TopologyErrors> {
    let Some(mut entering) = mesh.face_half_edge_on(seed) else {
        return Err(vec![format!(
            "the edge between vertex {} and vertex {} bounds no face, so no loop can be run through it",
            seed.0, seed.1
        )]);
    };

    // The edge `entering` belongs to is carried alongside it rather than recomputed each turn: the first
    // one is `seed` by the check above, and every later one is the edge the previous face exited by, so no
    // turn has to re-derive it and no turn can meet a link it cannot resolve.
    let mut entering_edge = seed;
    let mut crossings = Vec::new();
    let mut walked: HashSet<FaceId> = HashSet::new();

    // The walk ends when `entering` names a half-edge that belongs to no face, which is how a loop running
    // off an open mesh stops; the two breaks below end it when the ring comes round or a twin is missing.
    while let Some(face) = mesh.half_edges.get(entering.0).and_then(|edge| edge.face) {
        if !walked.insert(face) {
            break;
        }

        let Some(loop_edges) = mesh.face_walk(face) else {
            return Err(vec![format!(
                "face {} does not bound a closed loop of at least three corners, so the edge loop cannot \
                 be carried through it",
                face.0
            )]);
        };
        if loop_edges.len() != 4 {
            // A ring that runs into an n-gon stops there, exactly as one that runs off an open boundary
            // does: the cut is laid down as far as it can be measured, and where it stopped is visible.
            // Refusing the whole ring instead would make a single n-gon anywhere on a mesh turn the tool
            // dead, and "the loop ended here" is a better answer than "nothing happened".
            //
            // The face the user aimed at is the exception. A loop that cannot start has produced nothing at
            // all, which is indistinguishable from a dead key unless it says so - hence the refusal when
            // nothing has been crossed yet, naming the seed face and its corner count.
            if crossings.is_empty() {
                return Err(vec![format!(
                    "an edge loop cuts quad faces, and the face it starts on is face {} with {} corners; \
                     the edge opposite another is not a thing an n-gon has, so there is no cut to begin \
                     with here",
                    face.0,
                    loop_edges.len()
                )]);
            }
            break;
        }

        let exits = loop_edges[(index_in_loop(&loop_edges, entering) + 2) % loop_edges.len()];
        let Some(exiting_edge) = mesh.edge_id_of(exits) else {
            return Err(vec![format!(
                "face {} has a half-edge whose far corner cannot be read, so it needs repairing before \
                 the loop can cut it",
                face.0
            )]);
        };

        crossings.push(Crossing {
            face,
            enters: entering_edge,
            exits: exiting_edge,
        });

        let Some(twin) = mesh.half_edges.get(exits.0).and_then(|edge| edge.twin) else {
            break;
        };
        entering = twin;
        entering_edge = exiting_edge;

        // A ring around a solid comes round inside as many steps as there are faces. Anything still walking
        // at that point is following links into a cycle no closed solid has, and saying so beats spinning.
        if crossings.len() > mesh.faces.len() {
            return Err(vec![
                "the edge loop kept walking without coming round, which means edges are linked into a \
                 cycle no closed solid has"
                    .to_string(),
            ]);
        }
    }

    Ok(crossings)
}

/// Where `wanted` sits in a face loop. Its presence is not in question - the walk only ever steps from a
/// half-edge that came out of that face's own loop - so falling back to 0 keeps the walk total instead of
/// adding a panic to a path whose whole job is to describe damage rather than crash on it.
fn index_in_loop(loop_edges: &[HalfEdgeId], wanted: HalfEdgeId) -> usize {
    loop_edges
        .iter()
        .position(|&edge| edge == wanted)
        .unwrap_or(0)
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

    /// Two quads sharing the edge between corners 1 and 2 - the smallest mesh with an interior edge, an
    /// exterior edge, corners shared by two faces and corners claimed by only one.
    fn joined_quads() -> HalfEdgeMesh {
        let positions = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
        ];
        HalfEdgeMesh::from_faces(&positions, &[vec![0, 1, 2, 3], vec![4, 5, 2, 1]])
    }

    /// A 3x3 plate of unit quads in the `z = 0` plane, wound counter-clockwise about +z, with the listed
    /// faces left out. Faces are numbered row-major from the bottom left, so `0` is the plate's bottom-left
    /// corner and `4` is its centre: the pair that meets at corner 5 and shares no edge, which is two
    /// openings meeting at a single corner.
    ///
    /// Deleting the pair through [`HalfEdgeMesh::delete_faces`] is impossible - with the first gone the
    /// centre corner joins two separate fans, which `validate` refuses, and the all-or-nothing rule stops
    /// the second delete from ever landing. So the plate is authored straight from `from_faces`, which is
    /// also how a mesh with a pinch reaches this app: imported from a file.
    fn plate_missing(missing: &[usize]) -> HalfEdgeMesh {
        let positions: Vec<Point3<f32>> = (0..16)
            .map(|index| Point3::new((index % 4) as f32 - 1.5, (index / 4) as f32 - 1.5, 0.0))
            .collect();
        let mut quads: Vec<Vec<usize>> = Vec::new();
        for row in 0..3 {
            for column in 0..3 {
                if missing.contains(&(row * 3 + column)) {
                    continue;
                }
                let bottom_left = row * 4 + column;
                quads.push(vec![
                    bottom_left,
                    bottom_left + 1,
                    bottom_left + 5,
                    bottom_left + 4,
                ]);
            }
        }
        HalfEdgeMesh::from_faces(&positions, &quads)
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
        let mesh = joined_quads();

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

    /// A lone triangle, so the rule that a face with no diagonal cannot be cut has a witness.
    fn lone_triangle() -> HalfEdgeMesh {
        let positions = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        HalfEdgeMesh::from_faces(&positions, &[vec![0, 1, 2]])
    }

    /// A flat five-cornered face, the smallest shape where a cut can come out uneven.
    fn lone_pentagon() -> HalfEdgeMesh {
        let positions: Vec<Point3<f32>> = (0..5)
            .map(|step| {
                let angle = std::f32::consts::TAU * step as f32 / 5.0;
                Point3::new(angle.cos(), angle.sin(), 0.0)
            })
            .collect();
        HalfEdgeMesh::from_faces(&positions, &[vec![0, 1, 2, 3, 4]])
    }

    #[test]
    fn cutting_a_quad_across_a_diagonal_makes_two_triangles() {
        let mut mesh = open_quad();
        let cut = mesh
            .split_face(FaceId(0), VertexId(0), VertexId(2))
            .unwrap_or_else(|errors| panic!("a diagonal is exactly what a cut is for: {errors:?}"));

        assert_eq!(cut.original_face, FaceId(0), "the face named keeps its id");
        assert_eq!(cut.new_face, FaceId(1));
        // Both halves start at the far end of the cut, which is where their recorded edge now begins.
        assert_eq!(mesh.face_vertices(cut.new_face), ids(&[2, 0, 1]));
        assert_eq!(mesh.face_vertices(cut.original_face), ids(&[2, 3, 0]));

        let diagonal = EdgeId::between(VertexId(0), VertexId(2));
        let (own, neighbour) = mesh.edge_faces(diagonal);
        assert!(
            matches!(
                (own, neighbour),
                (Some(FaceId(0)), Some(FaceId(1))) | (Some(FaceId(1)), Some(FaceId(0)))
            ),
            "the cut must fuse the halves, got {own:?} and {neighbour:?}"
        );

        // Winding is inherited, not recomputed: each half is a contiguous slice of one loop, so the
        // renderer's back-face rule keeps agreeing with `validate` after the cut.
        for face in [cut.original_face, cut.new_face] {
            assert!(
                mesh.face_normal(face).z > 0.99,
                "half {} should still face +z like the quad it came from",
                face.0
            );
        }

        assert_eq!(mesh.generate_render_buffers().indices.len(), 6);
        mesh.validate()
            .unwrap_or_else(|errors| panic!("the cut mesh should validate: {errors:?}"));
    }

    #[test]
    fn cutting_a_pentagon_leaves_a_quad_and_a_triangle() {
        // Pins which half goes where: the chain running from `a` forward to `b` is what comes off as the
        // appended face, and the face named keeps the rest - including the corner between them.
        let mut mesh = lone_pentagon();
        let cut = mesh
            .split_face(FaceId(0), VertexId(0), VertexId(3))
            .unwrap_or_else(|errors| panic!("vertices 0 and 3 are not joined: {errors:?}"));

        assert_eq!(mesh.face_vertices(cut.new_face), ids(&[3, 0, 1, 2]));
        assert_eq!(mesh.face_vertices(cut.original_face), ids(&[3, 4, 0]));
        mesh.validate()
            .unwrap_or_else(|errors| panic!("the cut pentagon should validate: {errors:?}"));
    }

    /// Asserts a mesh is still a closed sphere topologically: vertices minus edges plus faces equals 2.
    ///
    /// Every edit here should preserve it, and it is the one check that catches a mistake none of the
    /// individual invariants notice - a cut that closes one seam and opens another elsewhere reports
    /// perfectly valid links while quietly turning the solid into something with a hole in it.
    fn euler_characteristic(mesh: &HalfEdgeMesh) -> i64 {
        mesh.vertices.len() as i64 - mesh.edges().len() as i64 + mesh.faces.len() as i64
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
        // Front and right share edge 1-2, so they weld: one copy of each shared corner and no wall grown
        // across the hinge. That is the difference between one slab and two coincident shells, which would
        // render as z-fighting and read as a self-intersection.
        //
        // The offset is diagonal on purpose. A cube's edges run along all three axes, so an axis-aligned
        // push is parallel to *some* boundary edge of any two-face region and the wall standing there would
        // have no area - pushing front and bottom along +z flattens the wall on the bottom's edge 5-1, which
        // the guard refuses. Averaging the two outward normals, which is what `Scene` hands down, points
        // between the axes and is parallel to none of them.
        let before = HalfEdgeMesh::create_cube();
        let mut mesh = before.clone();

        let shape = mesh
            .extrude_faces(&[FaceId(0), FaceId(1)], Vector3::new(0.5, 0.0, 0.5))
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
            "seven distinct edges around the pair, less the hinge they share"
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

    #[test]
    fn a_triangle_refuses_a_cut_because_every_pair_of_its_corners_is_already_an_edge() {
        // The witness for the adjacency rule: three corners, three edges, no pair that is not already
        // joined. A triangle is the smallest face there is, so "cut this face" has no meaning here and the
        // refusal has to say so rather than invent a corner to cut to.
        let before = lone_triangle();
        let mut mesh = before.clone();

        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            let errors = mesh
                .split_face(FaceId(0), VertexId(a), VertexId(b))
                .expect_err("every pair of a triangle's corners is an edge already");
            expect_reason(&errors, "already joined by an edge of face 0");
        }
        assert_identical(&mesh, &before);
    }

    #[test]
    fn cutting_a_face_refuses_a_corner_that_belongs_to_another_face() {
        let mut mesh = open_box();
        let before = mesh.clone();

        let errors = mesh
            .split_face(FaceId(0), VertexId(0), VertexId(6))
            .expect_err("vertex 6 is a back corner, not one of the front face's");
        expect_reason(&errors, "vertex 6 is not a corner of face 0");
        assert_identical(&mesh, &before);
    }

    #[test]
    fn cutting_a_face_refuses_the_same_corner_named_twice() {
        let mut mesh = open_quad();
        let before = mesh.clone();

        let errors = mesh
            .split_face(FaceId(0), VertexId(0), VertexId(0))
            .expect_err("one corner cannot span a cut");
        expect_reason(&errors, "a cut needs two different corners");
        assert_identical(&mesh, &before);
    }

    #[test]
    fn an_edge_loop_across_a_cube_splits_it_into_two_layers() {
        // The headline behaviour: a ring seeded on one edge of a cube runs all the way round it, cutting
        // the four faces it passes and leaving the two it enters corner-first alone. Ten faces, and the
        // solid stays closed.
        let mut mesh = HalfEdgeMesh::create_cube();
        let seed = EdgeId::between(VertexId(0), VertexId(1));

        let loop_cut = mesh
            .insert_edge_loop(seed)
            .unwrap_or_else(|errors| panic!("a cube edge should seed a clean loop: {errors:?}"));

        assert_eq!(
            loop_cut.cut_edges.len(),
            4,
            "one ring edge per crossed face"
        );
        assert_eq!(loop_cut.new_vertices.len(), 4);
        assert_eq!(
            loop_cut.new_edges.len(),
            4,
            "one cut per face the ring reached"
        );
        assert_eq!(mesh.faces.len(), 10, "four faces became eight, two stayed");
        assert_eq!(
            mesh.vertices.len(),
            12,
            "the original eight plus one per ring edge"
        );
        mesh.validate()
            .unwrap_or_else(|errors| panic!("the cut cube should validate: {errors:?}"));
    }

    #[test]
    fn an_edge_loop_keeps_the_solid_a_closed_sphere() {
        // Euler's formula is the one check the six validator passes cannot make between them: a cut that
        // closes one seam while opening another elsewhere reports perfectly sound links, and this is what
        // notices. A sphere stays at 2 however many loops are run through it.
        let mut mesh = HalfEdgeMesh::create_cube();
        assert_eq!(euler_characteristic(&mesh), 2, "the uncut cube");

        mesh.insert_edge_loop(EdgeId::between(VertexId(0), VertexId(1)))
            .expect("the first loop should cut cleanly");
        assert_eq!(euler_characteristic(&mesh), 2, "after one ring");

        // A second loop at right angles to the first, which is the cut most likely to disturb a link the
        // first one set up.
        mesh.insert_edge_loop(EdgeId::between(VertexId(8), VertexId(1)))
            .unwrap_or_else(|errors| panic!("a second ring should cut cleanly: {errors:?}"));

        assert_eq!(euler_characteristic(&mesh), 2, "after two rings");
        assert_eq!(
            mesh.faces.len(),
            14,
            "each belt crosses four faces and cuts them in half, so the second adds four - a belt is a \
             ring round the solid, not a sweep over every face"
        );
        assert!(mesh.boundary_edges().is_empty(), "and still sealed");
    }

    #[test]
    fn an_edge_loop_refuses_when_the_face_it_starts_on_is_not_a_quad() {
        // The one case where stopping is not an answer: nothing at all would be cut, which a user could
        // only read as a dead key. The face they aimed at is named with its corner count.
        let mut mesh = lone_pentagon();
        let before = mesh.clone();

        let errors = mesh
            .insert_edge_loop(EdgeId::between(VertexId(0), VertexId(1)))
            .expect_err("an n-gon has no edge opposite another");
        expect_reason(&errors, "face 0 with 5 corners");
        assert_identical(&mesh, &before);
    }

    #[test]
    fn an_edge_loop_stops_at_a_face_that_is_not_a_quad() {
        // Two quads running into a pentagon. The ring is measurable across the quads and stops dead at the
        // pentagon's edge, which is what Maya's Insert Edge Loop does where the quads give out - so the cut
        // the user asked for lands, and where it ended is visible rather than reported as a failure.
        let positions = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
            Point3::new(3.0, 0.0, 0.0),
            Point3::new(3.6, 0.5, 0.0),
            Point3::new(3.0, 1.0, 0.0),
        ];
        let mut mesh = HalfEdgeMesh::from_faces(
            &positions,
            &[vec![0, 1, 2, 3], vec![1, 4, 5, 2], vec![4, 6, 7, 8, 5]],
        );
        mesh.validate()
            .unwrap_or_else(|errors| panic!("the test patch should validate: {errors:?}"));

        // The hull edge of the first quad, so the walk starts there rather than on the far side of it.
        let loop_cut = mesh
            .insert_edge_loop(EdgeId::between(VertexId(0), VertexId(3)))
            .unwrap_or_else(|errors| {
                panic!("the ring should cut the quads it reaches: {errors:?}")
            });

        assert_eq!(
            loop_cut.new_edges.len(),
            2,
            "one ring segment per face crossed - the pentagon contributes none, which is what stopping means"
        );
        assert_eq!(
            loop_cut.new_vertices.len(),
            3,
            "both edges the ring passed through, plus the pentagon's own border edge, so the cut has a \
             visible end where it halted"
        );
        assert_eq!(mesh.faces.len(), 5, "each quad reached was split in two");
        assert_eq!(
            mesh.face_vertices(FaceId(2)).len(),
            6,
            "the pentagon is not cut, but the ring ends on its border edge and that edge is shared, so \
             splitting it hands the n-gon one extra corner - which is what Maya does here too"
        );
        assert!(
            mesh.face_normal(FaceId(2)).z > 0.99,
            "the pentagon still faces +z: it was given a corner on the edge the ring stopped at, nothing more"
        );
        mesh.validate()
            .unwrap_or_else(|errors| panic!("the part-cut patch should validate: {errors:?}"));
    }

    #[test]
    fn an_edge_loop_stops_at_an_open_boundary_instead_of_refusing() {
        // A ring on an unfinished mesh runs as far as the geometry does. Refusing here would make loops
        // unusable mid-model, and nothing is lost: the faces it reached are still cut correctly.
        let mut mesh = open_box();
        let seed = EdgeId::between(VertexId(0), VertexId(1));

        let loop_cut = mesh
            .insert_edge_loop(seed)
            .unwrap_or_else(|errors| panic!("a loop should run until it runs out: {errors:?}"));

        assert!(
            loop_cut.new_edges.len() < 4,
            "the missing bottom face means the ring has nowhere to go, got {}",
            loop_cut.new_edges.len()
        );
        assert!(
            !loop_cut.new_edges.is_empty(),
            "and it still cut what it reached"
        );
        mesh.validate()
            .unwrap_or_else(|errors| panic!("the partially cut box should validate: {errors:?}"));
    }

    #[test]
    fn an_edge_loop_refuses_an_edge_that_bounds_no_face() {
        let mut mesh = open_quad();
        mesh.half_edges.iter_mut().for_each(|edge| edge.face = None);
        let before = mesh.clone();

        let errors = mesh
            .insert_edge_loop(EdgeId::between(VertexId(0), VertexId(1)))
            .expect_err("with no face to cross, there is no ring to walk");
        expect_reason(&errors, "bounds no face");
        assert_identical(&mesh, &before);
    }

    // ---------------------------------------------------------------------------------------------
    // Bevel. Nothing here was covered when the primitive first landed, which is how a construction as
    // link-heavy as this one hides an unwound facet until a user selects the wrong edge.
    // ---------------------------------------------------------------------------------------------

    /// A square pyramid: five faces, and a corner where four of them meet. That apex is the smallest mesh
    /// that gives bevel a degree-four corner to refuse, and its base edges are ordinary degree-three ones
    /// to compare against, so a refusal there cannot be mistaken for a refusal everywhere.
    ///
    /// Wound outward on purpose. `from_faces` links twins by directed vertex pair, so one face wound the
    /// other way would leave two half-edges running the same pair the same way and the mesh would arrive
    /// already invalid - which every test below would report as a bevel bug.
    fn square_pyramid() -> HalfEdgeMesh {
        let positions = vec![
            Point3::new(-1.0, -1.0, 0.0),
            Point3::new(1.0, -1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(-1.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ];
        HalfEdgeMesh::from_faces(
            &positions,
            &[
                vec![0, 3, 2, 1], // base, facing down
                vec![0, 1, 4],
                vec![1, 2, 4],
                vec![2, 3, 4],
                vec![3, 0, 4],
            ],
        )
    }

    fn near(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn beveling_a_cube_edge_opens_a_facet_between_the_two_faces_it_bounded() {
        let mut mesh = HalfEdgeMesh::create_cube();
        let before = shape(&mesh);

        let bevel = mesh
            .bevel_edge(EdgeId::between(VertexId(0), VertexId(1)), 0.25)
            .unwrap_or_else(|errors| {
                panic!("a cube edge is the easiest bevel there is: {errors:?}")
            });

        assert_eq!(before, (8, 24, 6), "the cube starts where the digest says");
        assert_eq!(
            shape(&mesh),
            (10, 30, 7),
            "two corners and one facet, taking half-edges from 24 to 30 (three edges net, since each \
             inserted corner replaces one)"
        );
        // Euler written as V + F - E so the middle term cannot go negative: `edges()` is 15 here, and in
        // usize `10 - 15` wrapped into a huge number long before the assertion could report the truth.
        assert_eq!(
            10 + 7 - mesh.edges().len(),
            2,
            "V - E + F = 2 for a closed genus-zero shell, so every half-edge found a twin"
        );
        mesh.validate()
            .unwrap_or_else(|errors| panic!("the beveled solid should validate: {errors:?}"));
        assert!(
            mesh.boundary_edges().is_empty(),
            "a bevel opens a facet inside the solid, never a hole in it"
        );

        assert_eq!(
            bevel.face,
            FaceId(6),
            "the facet is the one face the op appends"
        );
        assert_eq!(bevel.new_vertices, [VertexId(8), VertexId(9)]);
        assert_eq!(mesh.face_vertices(bevel.face).len(), 4);
    }

    #[test]
    fn bevel_shortens_the_two_faces_it_split_and_trims_the_two_it_met() {
        let mut mesh = HalfEdgeMesh::create_cube();
        let counts_before: Vec<usize> = (0..mesh.faces.len())
            .map(|index| mesh.face_vertices(FaceId(index)).len())
            .collect();

        let bevel = mesh
            .bevel_edge(EdgeId::between(VertexId(0), VertexId(1)), 0.25)
            .expect("beveling a cube edge should succeed");

        assert_eq!(counts_before, vec![4; 6], "a cube is six quads");
        for &face in bevel.shortened_faces.iter() {
            assert_eq!(
                mesh.face_vertices(face).len(),
                4,
                "a face that gave up a strip of itself keeps its corner count"
            );
        }
        for &face in bevel.trimmed_faces.iter() {
            assert_eq!(
                mesh.face_vertices(face).len(),
                5,
                "a face that met the edge end-on gains the cut point, so a quad becomes a pentagon"
            );
        }
        assert_ne!(
            bevel.shortened_faces[0], bevel.shortened_faces[1],
            "the edge has two distinct sides on a solid"
        );
        assert_ne!(
            bevel.trimmed_faces[0], bevel.trimmed_faces[1],
            "and two distinct neighbours at its ends"
        );
    }

    #[test]
    fn bevel_lands_every_cut_point_the_amount_says_along_its_edge() {
        // The only check here about geometry rather than links, and the one that would catch the two ends
        // of the cut being built off the wrong neighbours: on a unit cube each adjacent corner is exactly
        // one unit away, so an amount of a quarter puts every new point a quarter of the way out.
        let mut mesh = HalfEdgeMesh::create_cube();
        let before: Vec<Point3<f32>> = mesh.vertices.iter().map(|vertex| vertex.position).collect();
        let amount = 0.25;

        let bevel = mesh
            .bevel_edge(EdgeId::between(VertexId(0), VertexId(1)), amount)
            .expect("beveling a cube edge should succeed");

        let fraction_along = |from: Point3<f32>, to: Point3<f32>| {
            (to - from).magnitude() / (before[0] - before[1]).magnitude()
        };
        assert_eq!(
            bevel.corners,
            [
                bevel.corners[0],
                bevel.corners[1],
                bevel.new_vertices[0],
                bevel.new_vertices[1]
            ],
            "the facet is wound through the two near corners then the two new ones, so the order reads \
             around it instead of jumping across"
        );
        for (corner, relocated) in [
            (VertexId(0), bevel.corners[1]),
            (VertexId(1), bevel.corners[0]),
        ] {
            assert_eq!(
                relocated, corner,
                "the near corners are the edge's own endpoints, moved"
            );
            let travelled = fraction_along(before[corner.0], mesh.vertices[corner.0].position);
            assert!(
                near(travelled, amount),
                "vertex {} slid {travelled} of the way along its edge, not {amount}",
                corner.0
            );
        }
        for (&fresh, &corner) in bevel
            .new_vertices
            .iter()
            .zip([VertexId(0), VertexId(1)].iter())
        {
            let out = fraction_along(before[corner.0], mesh.vertices[fresh.0].position);
            assert!(
                near(out, amount),
                "the new vertex beside vertex {} sits {out} out, not {amount}",
                corner.0
            );
        }

        let corners: Vec<Point3<f32>> = bevel
            .corners
            .iter()
            .map(|&corner| mesh.vertices[corner.0].position)
            .collect();
        assert!(
            face_area(&corners) > MIN_FACE_AREA,
            "the facet the four cut points enclose has a real area"
        );
    }

    #[test]
    fn a_beveled_edge_is_reused_as_the_facet_rather_than_removed() {
        // The caveat a bevel leaves for selection, and it is a different one from `split_edge`'s. There the
        // vertex pair vanishes; here it survives, because the two corners on the near side of the cut ARE
        // the original pair, moved. So an [`EdgeId`] held across a bevel stays legal while naming something
        // else - the facet's own edge instead of the corner it used to be. An operation that reports or
        // highlights what it acted on has to take what the primitive returned, not keep the selection.
        let mut mesh = HalfEdgeMesh::create_cube();
        let chosen = EdgeId::between(VertexId(0), VertexId(1));
        let before = mesh.vertices[0].position;

        let bevel = mesh
            .bevel_edge(chosen, 0.25)
            .expect("beveling a cube edge should succeed");

        let present: Vec<EdgeId> = mesh.edges().into_iter().map(|(edge, _)| edge).collect();
        assert!(
            present.contains(&chosen),
            "the pair still runs, now as the near side of the facet"
        );

        let (first, second) = mesh.edge_faces(chosen);
        let bounded: Vec<FaceId> = [first, second].into_iter().flatten().collect();
        assert_eq!(bounded.len(), 2, "the reused edge still bounds two faces");
        assert!(
            bounded.contains(&bevel.face),
            "one of them is the facet it was repurposed as: {bounded:?}"
        );
        assert!(
            bounded.contains(&bevel.shortened_faces[0])
                || bounded.contains(&bevel.shortened_faces[1]),
            "and the other is whichever side kept its own corners: {bounded:?}"
        );
        assert!(
            !mesh.is_boundary_edge(chosen),
            "a reused edge on a closed solid is not on the hull"
        );

        // The geometry moved even though the id did, which is the half of the caveat a count cannot show.
        assert_ne!(
            mesh.vertices[0].position, before,
            "the vertex the selection named is no longer where it was"
        );
    }

    #[test]
    fn bevel_refuses_an_edge_the_mesh_does_not_have() {
        let before = HalfEdgeMesh::create_cube();
        let mut mesh = before.clone();

        let errors = mesh
            .bevel_edge(EdgeId::between(VertexId(0), VertexId(6)), 0.25)
            .expect_err("that pair is a face diagonal, not an edge");
        expect_reason(&errors, "no half edge runs between vertex 0 and vertex 6");
        assert_identical(&mesh, &before);
    }

    #[test]
    fn bevel_refuses_an_edge_that_only_one_face_bounds() {
        let before = open_quad();
        let mut mesh = before.clone();

        let errors = mesh
            .bevel_edge(EdgeId::between(VertexId(0), VertexId(1)), 0.25)
            .expect_err("one side of the edge is open air");
        expect_reason(&errors, "only one face");
        assert_identical(&mesh, &before);
    }

    #[test]
    fn bevel_refuses_a_corner_where_four_faces_meet_and_names_it() {
        let mut mesh = square_pyramid();
        mesh.validate()
            .unwrap_or_else(|errors| panic!("the pyramid itself must be sound: {errors:?}"));

        let apex = EdgeId::between(VertexId(4), VertexId(0));
        let errors = mesh
            .bevel_edge(apex, 0.25)
            .expect_err("four faces meet at the apex, so the cut has no single neighbour to trim");
        expect_reason(&errors, "vertex 4 is shared by 4 faces");
        assert_eq!(
            shape(&mesh),
            (5, 16, 5),
            "five faces of a pyramid is 4 + 3x4 half-edges, and nothing moved"
        );
    }

    #[test]
    fn bevel_takes_a_base_edge_of_the_same_pyramid() {
        // The control for the refusal above: same mesh, degree-three ends, and the bevel works - so a
        // future "this shape is always refused" regression cannot hide behind the apex test.
        let mut mesh = square_pyramid();

        mesh.bevel_edge(EdgeId::between(VertexId(0), VertexId(1)), 0.25)
            .unwrap_or_else(|errors| panic!("a base edge has degree-three ends: {errors:?}"));

        mesh.validate()
            .unwrap_or_else(|errors| panic!("the beveled pyramid should validate: {errors:?}"));
        assert!(mesh.boundary_edges().is_empty());
        assert_eq!(
            shape(&mesh),
            (7, 22, 6),
            "same arithmetic as the cube: eleven edges, so Euler still gives a closed shell"
        );
    }

    #[test]
    fn bevel_refuses_an_amount_outside_the_open_unit_interval() {
        let before = HalfEdgeMesh::create_cube();
        let mut mesh = before.clone();

        for amount in [0.0f32, -0.25, 1.0, 2.0, f32::NAN, f32::INFINITY] {
            let errors = mesh
                .bevel_edge(EdgeId::between(VertexId(0), VertexId(1)), amount)
                .expect_err("{amount} is not a fraction of an edge");
            expect_reason(&errors, "between 0 and 1");
        }
        assert_identical(&mesh, &before);
    }

    #[test]
    fn bevel_refuses_two_faces_that_lie_flat_against_each_other() {
        // A loop cut leaves a seam where two halves of one face still share a plane. Beveling that edge
        // folds a facet shut along it: geometry that renders as a crease, cannot be picked, and about which
        // `validate` has no opinion, since nothing in it breaks a topology rule. The angle is checked before
        // the corners are resolved because a seam like this also lands on a corner of degree four, and the
        // flat hinge is the reason that actually applies to it.
        let mut mesh = HalfEdgeMesh::create_cube();
        mesh.insert_edge_loop(EdgeId::between(VertexId(0), VertexId(1)))
            .expect("a cube takes a loop cut around its four sides");
        let settled = mesh.clone();

        let flat = mesh
            .edges()
            .into_iter()
            .find(|&(edge, _)| {
                let (first, second) = mesh.edge_faces(edge);
                matches!((first, second), (Some(a), Some(b))
                    if mesh.face_normal(a).cross(mesh.face_normal(b)).magnitude()
                        < MIN_BEVEL_DIHEDRAL_SINE)
            })
            .map(|(edge, _)| edge)
            .expect("a loop cut leaves two halves of one face lying in the same plane");

        let errors = mesh
            .bevel_edge(flat, 0.25)
            .expect_err("a hinge with no angle has no facet to open");
        expect_reason(&errors, "lie flat against each other");
        assert_identical(&mesh, &settled);
    }

    #[test]
    fn bevel_refuses_to_build_on_a_mesh_that_is_already_broken() {
        let mut mesh = HalfEdgeMesh::create_cube();
        mesh.half_edges[9].face = None;
        let before = mesh.clone();

        let errors = mesh
            .bevel_edge(EdgeId::between(VertexId(0), VertexId(1)), 0.25)
            .expect_err("damage elsewhere must stop this edit");
        expect_reason(&errors, "belongs to no face");
        expect_reason(&errors, "nothing was installed");
        assert_identical(&mesh, &before);
    }

    #[test]
    fn beveling_two_edges_that_share_a_corner_keeps_the_solid_closed() {
        // The sequence a user reaches for immediately: chamfer one edge, then the next. Each bevel turns
        // the faces it trimmed into pentagons, so the second reads links the first wrote rather than the
        // cube it started from - and both ends stay degree three, which is what lets it work at all.
        let mut mesh = HalfEdgeMesh::create_cube();

        mesh.bevel_edge(EdgeId::between(VertexId(0), VertexId(1)), 0.2)
            .expect("the first bevel should succeed");
        mesh.bevel_edge(EdgeId::between(VertexId(0), VertexId(3)), 0.2)
            .expect("the second bevel, which shares vertex 0 with the first, should also succeed");

        mesh.validate()
            .unwrap_or_else(|errors| panic!("two chamfers should still be a solid: {errors:?}"));
        assert_eq!(mesh.faces.len(), 8, "one facet per bevel");
        assert!(
            mesh.boundary_edges().is_empty(),
            "still closed after both chamfers"
        );
    }

    #[test]
    fn deleting_one_face_of_a_cube_leaves_an_open_box() {
        let mut mesh = HalfEdgeMesh::create_cube();

        let gone = mesh
            .delete_faces(&[FaceId(0)])
            .expect("a cube face is perfectly deletable");

        assert_eq!(mesh.faces.len(), 5);
        assert_eq!(
            mesh.half_edges.len(),
            20,
            "the face took its four edges with it"
        );
        assert_eq!(gone.faces, vec![FaceId(0)]);
        assert_eq!(gone.half_edges, 4);
        assert_eq!(
            gone.vertices, 0,
            "every corner is shared with three other faces"
        );
        mesh.validate()
            .unwrap_or_else(|errors| panic!("a five-sided box should validate: {errors:?}"));

        assert_eq!(mesh.boundary_edges().len(), 4, "the rim where the face was");
        assert_eq!(
            euler_characteristic(&mesh),
            1,
            "a box missing its lid is topologically a disc, not a sphere"
        );
        assert_eq!(
            mesh.generate_render_buffers().indices.len(),
            30,
            "five quads fan to 10 triangles"
        );
    }

    #[test]
    fn the_neighbours_of_a_deleted_face_gain_the_edges_it_left() {
        // The twin pass is the whole reason for rebuilding rather than blanking records: each of the four
        // faces that bounded the deleted one had the relevant half-edge twinned, and after the delete that
        // partner is gone. Leaving the pointer behind would be a dangling id; clearing both sides is what
        // turns a shared edge into a rim.
        //
        // The expectation is measured off the mesh before it changes rather than assumed, because a cube
        // face has one neighbour on the far side that shares nothing - and the rebuild renumbers, so a
        // hardcoded face index would describe whichever face happened to land there.
        let before = HalfEdgeMesh::create_cube();
        let doomed = FaceId(0);

        let shared: Vec<usize> = (1..before.faces.len())
            .map(|index| {
                before
                    .face_half_edges(FaceId(index))
                    .iter()
                    .filter(|&&half_edge| {
                        before.half_edges[half_edge.0]
                            .twin
                            .is_some_and(|twin| before.half_edges[twin.0].face == Some(doomed))
                    })
                    .count()
            })
            .collect();
        assert_eq!(
            shared.iter().sum::<usize>(),
            4,
            "four faces bound the front one"
        );
        assert_eq!(
            shared.iter().filter(|&&count| count == 0).count(),
            1,
            "the face on the far side shares no edge with it: {shared:?}"
        );

        let mut mesh = before;
        let gone = mesh
            .delete_faces(&[doomed])
            .expect("a cube face is perfectly deletable");

        for (old, expected) in shared.iter().enumerate() {
            let survivor = gone
                .remap
                .face(FaceId(old + 1))
                .expect("only face 0 was deleted");
            let open = mesh
                .face_half_edges(survivor)
                .iter()
                .filter(|&&half_edge| mesh.half_edges[half_edge.0].twin.is_none())
                .count();
            assert_eq!(
                open, *expected,
                "face {} should meet the opening along {} of its four edges",
                survivor.0, expected
            );
        }
    }

    #[test]
    fn a_deleted_face_retires_the_corners_nothing_else_bounds() {
        // The far pair belongs to the second face alone, so deleting that face must drop them while the
        // shared pair stays. This is the same question `check_vertices` asks, which is why no separate
        // orphan sweep is needed afterwards - the survivor rule and the validity rule are one rule.
        let mut mesh = joined_quads();
        let before = mesh.clone();

        let gone = mesh
            .delete_faces(&[FaceId(1)])
            .expect("the second of two joined quads is deletable");

        assert_eq!(mesh.faces.len(), 1);
        assert_eq!(mesh.half_edges.len(), 4);
        assert_eq!(gone.half_edges, 4);
        assert_eq!(gone.vertices, 2, "the two corners only that face claimed");
        assert_eq!(mesh.vertices.len(), before.vertices.len() - 2);
        assert_eq!(mesh.boundary_edges().len(), 4, "a lone quad is all rim");
        mesh.validate()
            .unwrap_or_else(|errors| panic!("the lone remaining quad should validate: {errors:?}"));
    }

    #[test]
    fn every_surviving_face_keeps_the_geometry_its_remap_points_at() {
        // Deleting shifts every id above the gap, so the guarantee a caller needs is not "ids are stable"
        // but "the remap is right". Asserted against positions rather than ids, because a rebuild that
        // pointed a surviving face at the wrong corner would still have had the right counts.
        let before = HalfEdgeMesh::create_cube();
        let mut mesh = before.clone();
        let kept: Vec<(FaceId, Vec<[f32; 3]>)> = (0..before.faces.len())
            .filter(|&index| index != 2)
            .map(|index| {
                let corners = before
                    .face_positions(FaceId(index))
                    .iter()
                    .map(|point| [point.x, point.y, point.z])
                    .collect();
                (FaceId(index), corners)
            })
            .collect();

        let gone = mesh
            .delete_faces(&[FaceId(2)])
            .expect("a cube face is perfectly deletable");

        assert_eq!(
            gone.remap.face(FaceId(2)),
            None,
            "the deleted face names nothing"
        );
        for (old, corners) in kept {
            let fresh = gone
                .remap
                .face(old)
                .unwrap_or_else(|| panic!("face {} should have survived", old.0));
            assert_face_at(&mesh, fresh, &corners);
        }

        for index in 0..before.vertices.len() {
            let old = VertexId(index);
            if let Some(fresh) = gone.remap.vertex(old) {
                assert_eq!(
                    mesh.vertices[fresh.0].position, before.vertices[index].position,
                    "corner {} was remapped to a vertex standing somewhere else",
                    old.0
                );
            }
        }
    }

    #[test]
    fn the_open_box_reports_one_rim_running_the_missing_faces_outline() {
        let mesh = open_box();

        let rims = mesh
            .boundary_rims()
            .expect("five cube faces meet in one clean opening");

        assert_eq!(rims.len(), 1, "one opening, therefore one rim");
        assert_eq!(
            rims[0].len(),
            4,
            "the missing bottom quad bounds four edges"
        );

        // The rim has to arrive back where it started, corner for corner: a chain that drifted would cap
        // the wrong opening or none. Each step's destination is the next step's origin, and the last
        // arrives at the first's origin.
        let corners: Vec<usize> = rims[0]
            .iter()
            .map(|&half_edge| mesh.half_edges[half_edge.0].origin.0)
            .collect();
        let destinations: Vec<usize> = rims[0]
            .iter()
            .map(|&half_edge| mesh.half_edge_destination(half_edge).0)
            .collect();
        assert_eq!(
            corners,
            vec![0, 1, 5, 4],
            "one turn of the opening's outline"
        );
        assert_eq!(corners[0], destinations[3], "the rim closes on itself");
        assert_eq!(
            destinations[..3],
            corners[1..],
            "each step arrives at the corner the next one leaves, so the walk never jumps the rim"
        );
    }

    #[test]
    fn filling_an_open_box_grows_the_face_that_was_left_off() {
        let mut mesh = open_box();

        let filled = mesh.fill_holes().expect("an open box is one clean opening");

        assert_eq!(filled.faces, vec![FaceId(5)]);
        assert_eq!(
            filled.corners,
            vec![4],
            "the lid is the quad that was missing"
        );
        assert_eq!(mesh.faces.len(), 6);
        assert!(
            mesh.boundary_edges().is_empty(),
            "nothing is left open, so the solid is closed again"
        );
        mesh.validate()
            .unwrap_or_else(|errors| panic!("a closed box should validate: {errors:?}"));
        assert_eq!(
            euler_characteristic(&mesh),
            2,
            "closed on every side is a sphere again, which is what the disc became"
        );

        // The strongest form of the winding claim: reversing the rim does not merely produce *a* face that
        // twins, it reproduces the missing face's own corner loop - same corners, same order - so the cap
        // carries the outward (-y) normal the bottom had rather than the inward one its neighbour wears.
        assert_eq!(
            mesh.face_vertices(FaceId(5)),
            ids(&CUBE_FACES[5]),
            "the cap should be the cube's bottom face, corner for corner and in the same winding"
        );
        let normal = mesh.face_normal(FaceId(5));
        assert!(
            normal.y < -0.99,
            "the cap points down and out, not up through the box: {normal:?}"
        );
    }

    #[test]
    fn filling_a_closed_mesh_is_refused_because_there_is_nothing_to_close() {
        let before = HalfEdgeMesh::create_cube();
        let mut mesh = before.clone();

        let errors = mesh
            .fill_holes()
            .expect_err("a closed solid has no opening");

        expect_reason(&errors, "no opening to fill");
        assert_identical(&mesh, &before);
    }

    #[test]
    fn closing_a_lone_quad_would_stack_a_second_face_on_the_one_it_closes() {
        // The rim of a single quad bounds an opening, and the one face that could close it is the quad
        // itself flipped. Twinning links perfectly and every other invariant holds, which is exactly the
        // case `check_faces_are_distinct` exists for - so the refusal arrives as that validator's line with
        // the fill's own context in front of it, and nothing installed.
        let before = open_quad();
        let mut mesh = before.clone();

        let errors = mesh
            .fill_holes()
            .expect_err("the only face that could close this rim is already there");

        expect_reason(&errors, "span the same 4 corners");
        expect_reason(&errors, "filling 1 opening");
        assert_identical(&mesh, &before);
    }

    #[test]
    fn every_opening_in_a_mesh_closes_in_one_pass() {
        let mut mesh = HalfEdgeMesh::create_cube();

        // Front and back, leaving a tube: two openings that share nothing.
        mesh.delete_faces(&[FaceId(0), FaceId(2)])
            .expect("two opposite faces of a cube are deletable");
        assert_eq!(mesh.boundary_edges().len(), 8, "two rims of four");

        let rims = mesh
            .boundary_rims()
            .expect("the two rims share no corner, so each is walkable");
        assert_eq!(rims.len(), 2, "one rim per opening, not one merged walk");

        let filled = mesh
            .fill_holes()
            .expect("both ends of the tube should close together");

        assert_eq!(filled.faces.len(), 2, "one cap per opening");
        assert_eq!(filled.corners, vec![4, 4], "both caps are quads");
        assert_eq!(mesh.faces.len(), 6);
        assert!(mesh.boundary_edges().is_empty());
        mesh.validate()
            .unwrap_or_else(|errors| panic!("a closed solid should validate: {errors:?}"));
        assert_eq!(euler_characteristic(&mesh), 2);

        // Each cap covers one of the two faces that were removed, and the two are not the same four
        // corners - measured as sets, because the delete in front of this renumbered the mesh.
        let mut caps: Vec<Vec<usize>> = filled
            .faces
            .iter()
            .map(|&face| {
                let mut corners: Vec<usize> = mesh
                    .face_vertices(face)
                    .iter()
                    .map(|&corner| corner.0)
                    .collect();
                corners.sort_unstable();
                corners
            })
            .collect();
        caps.sort();
        assert_eq!(
            caps,
            vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
            "the caps belong to opposite ends of the tube, not both to one end"
        );
    }

    #[test]
    fn filling_appends_so_every_id_handed_out_before_it_still_names_the_same_face() {
        // The counterpart to the remap a delete has to carry: a cap adds records and removes none, so no id
        // moves and `Scene` needs no remap pass here. Stated as a test because it is what the digest's
        // append-only rule buys, and the day filling is rewritten to rebuild this is what fails first.
        let mut mesh = open_box();
        let before: Vec<Vec<Point3<f32>>> = (0..mesh.faces.len())
            .map(|index| mesh.face_positions(FaceId(index)))
            .collect();

        mesh.fill_holes().expect("an open box is one clean opening");

        for (index, corners) in before.iter().enumerate() {
            assert_eq!(
                mesh.face_positions(FaceId(index)),
                *corners,
                "face {} was standing somewhere else after the fill",
                index
            );
        }
        assert_eq!(
            mesh.faces.len(),
            before.len() + 1,
            "the cap is the only record added"
        );
    }

    /// The lattice index of a point on the plate's 5x5 grid, for corner comparisons in the pinch tests.
    fn lattice(x: i32, y: i32) -> usize {
        ((y + 2) * 5 + (x + 2)) as usize
    }

    /// A flat plate of unit quads spanning [-2,2] with two square holes that touch at exactly one corner.
    ///
    /// The holes are the cells at [-1,0]x[0,1] and [0,1]x[-1,0], so they share the corner (0,0) and no
    /// edge: the pinch. Built from a grid rather than a typed-out face list so that "the two openings share
    /// one corner and nothing else" is a property of the construction rather than of my having counted the
    /// coordinates correctly, and so the plate stays manifold everywhere else.
    fn plate_with_two_touching_holes() -> HalfEdgeMesh {
        let mut positions = Vec::with_capacity(25);
        for y in -2..=2 {
            for x in -2..=2 {
                positions.push(Point3::new(x as f32, y as f32, 0.0));
            }
        }

        let mut faces = Vec::with_capacity(14);
        for y in -2..2 {
            for x in -2..2 {
                // The two cells that are the holes, left out so their four edges become boundaries.
                if (x, y) == (-1, 0) || (x, y) == (0, -1) {
                    continue;
                }
                faces.push(vec![
                    lattice(x, y),
                    lattice(x + 1, y),
                    lattice(x + 1, y + 1),
                    lattice(x, y + 1),
                ]);
            }
        }

        HalfEdgeMesh::from_faces(&positions, &faces)
    }

    #[test]
    fn two_openings_that_touch_at_one_corner_each_get_their_own_cap() {
        let mut mesh = plate_with_two_touching_holes();

        // The pinch is real, and `validate` already refuses it: the faces at (0,0) form two fans rather
        // than one, so nothing can be walked between the two sides. Filling is the operation that *heals*
        // this damage rather than the one it defeats - capping both holes puts a face in each of the two
        // gaps, and the corner's fan closes into a ring.
        let damaged = mesh
            .validate()
            .expect_err("a corner that joins two fans is the damage the fan check exists for");
        assert!(
            damaged
                .iter()
                .any(|reason| reason.contains("separate fans") && reason.contains("vertex 12")),
            "expected the pinch at (0,0), which is lattice index 12, got {damaged:?}"
        );

        // Both rims are walkable, and the shared corner is a choice to make rather than a dead end. The
        // plate's own outside edge is the third opening.
        let rims = mesh
            .boundary_rims()
            .expect("two openings meeting at one corner are still two openings");
        assert_eq!(
            rims.len(),
            3,
            "the two holes plus the plate's outer outline"
        );
        let on_the_pinch = rims
            .iter()
            .filter(|rim| {
                rim.iter()
                    .any(|&half_edge| mesh.half_edges[half_edge.0].origin == VertexId(12))
            })
            .count();
        assert_eq!(
            on_the_pinch, 2,
            "exactly the two holes touch at (0,0); the outline does not"
        );

        let filled = mesh
            .fill_holes()
            .expect("each opening closes on its own, not one face across both");
        assert_eq!(filled.faces.len(), 3, "one cap per opening");
        // Which rim is scanned first only decides the order the caps arrive in, so the sizes are compared
        // sorted: the two holes take a quad each and the plate's outline takes the sixteen corners it has.
        let mut sizes = filled.corners.clone();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![4, 4, 16], "two quads and the outline");
        assert!(mesh.boundary_edges().is_empty());
        mesh.validate().unwrap_or_else(|errors| {
            panic!("capping both holes should heal the pinch: {errors:?}")
        });
        // A flat plate with all three of its outlines closed is a squashed sphere, and Euler says so -
        // which is the check that the two caps did not merge into one figure-eight.
        assert_eq!(euler_characteristic(&mesh), 2);
    }

    #[test]
    fn each_cap_of_a_pinch_covers_its_own_hole_and_faces_the_way_the_plate_does() {
        let mut mesh = plate_with_two_touching_holes();
        let filled = mesh
            .fill_holes()
            .expect("each opening closes on its own, not one face across both");

        // Read the hole caps as corner sets, ignoring the plate's own outline cap, since which rim is
        // scanned first only decides the order the caps come back in.
        let mut caps: Vec<Vec<usize>> = filled
            .faces
            .iter()
            .map(|&face| {
                let mut corners: Vec<usize> = mesh
                    .face_vertices(face)
                    .iter()
                    .map(|&corner| corner.0)
                    .collect();
                corners.sort_unstable();
                corners
            })
            .filter(|corners| corners.len() == 4)
            .collect();
        caps.sort();
        let mut expected = vec![
            vec![lattice(-1, 0), lattice(0, 0), lattice(-1, 1), lattice(0, 1)],
            vec![lattice(0, -1), lattice(1, -1), lattice(0, 0), lattice(1, 0)],
        ];
        for corners in expected.iter_mut() {
            corners.sort_unstable();
        }
        expected.sort();
        assert_eq!(
            caps, expected,
            "the two hole caps must be the two cells that were left out of the grid"
        );

        // Both hole caps face +Z, like every face of the plate. A cap that had been wound the other way
        // would sit in the same place, twin correctly, and still read as a dent from the outside.
        let quads = filled
            .faces
            .iter()
            .filter(|&&face| mesh.face_vertices(face).len() == 4)
            .count();
        assert_eq!(
            quads, 2,
            "the two holes, and the outline cap is not one of them"
        );
        for &face in &filled.faces {
            if mesh.face_vertices(face).len() != 4 {
                continue;
            }
            assert!(
                mesh.face_normal(face).z > 0.9,
                "cap {face:?} must face the way the plate does, not into it"
            );
        }
    }

    #[test]
    fn an_imported_pinch_walks_as_two_openings_despite_their_shared_corner() {
        // The shape `plate_missing` documents: the bottom-left and centre quads left out, so the centre hole
        // and the notch the bottom-left one leaves in the plate's outline meet at corner 5 and share nothing
        // else. `delete_faces` cannot put a mesh into this state - removing the first of the pair leaves
        // corner 5 joining two fans, which `validate` refuses - so a pinch reaches this app through a file
        // import, which is exactly what this fixture authors.
        let mesh = plate_missing(&[0, 4]);
        let damaged = mesh
            .validate()
            .expect_err("corner 5 joins the notch and the hole as two separate fans");
        assert!(
            damaged
                .iter()
                .any(|reason| reason.contains("vertex 5") && reason.contains("separate fans")),
            "expected the pinch to be reported at corner 5, got {damaged:?}"
        );

        let rims = mesh
            .boundary_rims()
            .expect("both openings are walkable despite sharing corner 5");
        assert_eq!(
            rims.len(),
            2,
            "the centre hole and the notched outline, not one merged walk"
        );
        assert_eq!(
            rims.iter()
                .filter(|rim| {
                    rim.iter()
                        .any(|&half_edge| mesh.half_edges[half_edge.0].origin == VertexId(5))
                })
                .count(),
            2,
            "both openings run through the shared corner, which is what makes it a pinch"
        );

        let mut sizes: Vec<usize> = rims.iter().map(|rim| rim.len()).collect();
        sizes.sort_unstable();
        assert_eq!(
            sizes,
            vec![4, 12],
            "the centre hole is a ring of four and the notched outline twelve - and both closed, since an \
             opening that failed to close on itself would have been reported rather than returned"
        );
    }

    #[test]
    fn filling_an_imported_mesh_is_blocked_by_damage_it_did_not_make_and_says_where() {
        // The same fixture, filled - and refused. Not because of the pinch: `plate_missing` hands
        // `from_faces` all sixteen lattice corners including the one only the removed bottom-left quad used,
        // so the mesh arrives with a corner no face reaches. Filling validates the whole document before it
        // installs anything, so that stray corner blocks both openings, and the reason names the corner
        // rather than either hole.
        //
        // Kept as a test rather than smoothed over, because it is the shape of every all-or-nothing rule in
        // this module: an operation is only as good as the document it was handed, and the message has to
        // point somewhere the user can act on. Retiring corners nothing bounds is an operation of its own,
        // and it is listed in the digest's next steps.
        let before = plate_missing(&[0, 4]);
        let mut mesh = before.clone();

        let errors = mesh.fill_holes().expect_err(
            "a corner the mesh does not use blocks the whole document, so nothing fills",
        );

        expect_reason(&errors, "does not validate");
        expect_reason(&errors, "vertex 0 starts no half edge");
        assert_identical(&mesh, &before);
    }

    #[test]
    fn a_figure_eight_across_two_openings_is_still_refused_by_the_corner_it_repeats() {
        // Two triangles touching at a corner is not a pinch that can be closed: their wedges at the shared
        // corner overlap, so the walk that keeps one opening on its right is free to visit the other, and
        // the single rim it produces lists that corner twice. That face is refused for repeating a corner -
        // stated here because the walk now *chooses* at a shared corner, and the reader needs to know the
        // choice ends up checked rather than trusted.
        let positions = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(0.5, -1.0, 0.0),
            Point3::new(1.5, -1.0, 0.0),
        ];
        let before = HalfEdgeMesh::from_faces(&positions, &[vec![0, 1, 2], vec![2, 3, 4]]);
        let mut mesh = before.clone();

        let errors = mesh
            .fill_holes()
            .expect_err("a cap that runs through one corner twice is not a face");

        expect_reason(&errors, "could not be closed");
        assert_identical(&mesh, &before);
    }
}
