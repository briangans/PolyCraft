# PolyCraft - AI Context Digest

## 1. Project Vision
PolyCraft is a trimmed-down Maya clone written in Rust, built to leverage decades of learned muscle memory (e.g., V-key vertex snapping, D-key pivot editing).
- **Core Features:** Polygon modeling, FBX export, pie menus.
- **Explicit Non-Goals:** Advanced rendering (mental ray/Vray), physics, NURBS, Output Window, complex construction history.

## 2. Tech Stack
- **Language:** Rust (Windows / MSVC ABI)
- **Graphics:** `wgpu` (DirectX 12 backend), `winit`, `pollster`, `bytemuck`, `cgmath`
- **UI:** `egui` (Immediate mode, perfect for Pie Menus)
- **Geometry:** Custom Half-Edge mesh structure using Arena/Index vectors (`usize`) to bypass borrow checker cycles.
- **Versions:** Set in `Cargo.toml` (winit 0.29, wgpu 0.20, egui 0.28). These crates change APIs often, so trust `Cargo.toml` over memory.

## 3. Architectural Rules
- **Context Management:** Code is modularized. Only share the specific `src/*.rs` file being actively worked on, plus this digest.
- **State Management:** Strict decoupling. The `State` struct encapsulates all `wgpu` initialization. The `winit` event loop strictly routes OS events.
- **Render Passes:** 3D Scene renders first with Z-buffer depth. `egui` renders second with `LoadOp::Load` to draw on top.
- **Scene Ownership:** `scene.rs` owns the editable document: the mesh, the `EditMode` and the `Selection`. Change them only through the `Scene` API (`mesh_mut()`, `set_mode()`, `select_at()`). Each flags the document dirty; `State::render` then calls `take_dirty()` and `upload_mesh()` to rebuild the GPU buffers.
- **UI Returns Intent:** `ui.rs` never mutates app state. `UiState::draw` returns what the user asked for (e.g. `Option<EditMode>`) and `main.rs` applies it.
- **Pure Logic Stays Testable:** Mesh topology, picking, selection and camera math never touch `wgpu` or `winit`, and have unit tests next to the code (`mod tests`). Keep new logic out of GPU code and `main.rs`; `main.rs` only wires it up.

## 4. File Structure (Current)
```text
polycraft/
├── Cargo.toml
├── CLAUDE.md              # Instructions for Claude Code sessions
├── .claude/settings.json  # Claude Code permissions (edits in src/docs, cargo, read-only git; push denied)
├── docs/
│   ├── AI_DIGEST.md
│   └── ARCHITECTURE.md    # Scope: what is in and out
└── src/
    ├── main.rs       # winit loop, wgpu State, GPU Vertex layout, mesh buffer upload
    ├── mesh.rs       # Half-Edge mesh, component ids, topology queries, triangulation for rendering
    ├── camera.rs     # Orbit camera, Maya-style navigation, view basis for picking
    ├── scene.rs      # Editable document: mesh + EditMode + Selection, dirty flag
    ├── selection.rs  # Selected components per kind, Maya click semantics
    ├── picking.rs    # Screen position -> ray; face/vertex/edge/object hit tests
    ├── edit_mode.rs  # EditMode enum (Object/Vertex/Edge/Face)
    ├── ui.rs         # Spacebar pie menu (flick and release), Tools window
    └── shader.wgsl   # Vertex/fragment shader with flat shading lighting
```

## 5. Conventions & Invariants
- Right-handed, Y-up. Front faces wind counter-clockwise (`FrontFace::Ccw`, back-face culling on).
- Flat shading: one normal per face (Newell's method in `face_normal`). Render vertices are un-welded, and `RenderMesh::vertex_faces` records which face each one came from. Faces are triangulated as fans, so they must be convex.
- `VertexId`, `HalfEdgeId`, `FaceId` are `usize` newtypes indexing the `Vec`s in `HalfEdgeMesh`. `EdgeId` is an undirected pair of vertex indices (smaller first), so both half-edges of an edge give the same id.
- Picking tolerances are in screen pixels (`VERTEX_PICK_RADIUS_PIXELS`, `EDGE_PICK_RADIUS_PIXELS`), so they feel the same at any zoom. Components behind a nearer face are not pickable.
- Selection is one set per kind, and switching `EditMode` drops the other kinds. A plain click replaces the selection, Shift+click toggles, and a plain click on empty space clears it (Maya behavior).
- `Vertex` in `main.rs` (position, normal, color) must match the `@location`s in `shader.wgsl`.

## 6. Status (update at the end of every task)
- **Working (run-tested):** Flat-shaded cube from the half-edge mesh; Maya camera (Alt+LMB orbit, Alt+MMB pan, Alt+RMB/scroll zoom); Space pie menu: hold Space, flick toward a slot and release to set `EditMode`, shown in the "PolyCraft Tools" window. Releasing in the center, outside the window or after losing focus cancels; clicking a slot also works; the tracking line hides when the mouse leaves the window.
- **Implemented and unit-tested, not wired to the app yet:** Picking (ray from a screen position; face, vertex, edge and object hit tests with occlusion), `Selection` with Maya click semantics, and `Scene::select_at`. Nothing calls `select_at`, so clicking selects nothing.
- **Not built yet:** Click-to-select wiring, selection highlight, edge/vertex display, transforms, edit operations, undo, FBX export. `EditMode` is stored but changes nothing visible yet.
- **Known limitations:** `u16` index buffer (about 16k quads max); fan triangulation needs convex faces; the graphics backend is chosen by `Instance::default()`, not forced to DX12; the dirty flag is coarse, so a mode change, a selection change or any `mesh_mut()` call recreates both GPU buffers (before interactive vertex dragging, split it into topology and positions, and use `queue.write_buffer` for positions); `cargo clippy` warns about the unwired selection API and a missing `Default` for `HalfEdgeMesh`.
- **Next (proposed):** (1) Wire clicks: a plain left click (no Alt) in the viewport that egui did not consume becomes a `Ray` via `Ray::through_ndc` and calls `Scene::select_at` (Shift = additive). (2) Show it: tint selected faces using `vertex_faces`, then draw edges and vertices in a new `overlay.rs`. (3) Transform tools.

## 7. Handoff Checklist
1. `cargo check`, `cargo test` and `cargo clippy --all-targets` pass (no new warnings), and the app launches.
2. Update Section 4 (file structure) and Section 6 (status).
3. Commit with a descriptive message.

Roles for now: Claude writes; Gemini reviews each change from a diff plus a review brief.
