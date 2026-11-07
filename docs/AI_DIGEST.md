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
- **Context Management:** Code is heavily modularized. Only share the specific `src/*.rs` file being actively worked on.
- **State Management:** Strict decoupling. The `State` struct encapsulates all `wgpu` initialization. The `winit` event loop strictly routes OS events.
- **Render Passes:** 3D Scene renders first with Z-buffer depth. `egui` renders second with `LoadOp::Load` to draw on top.
- **Scene Ownership:** `scene.rs` owns the editable document (mesh + `EditMode`). The mesh is mutated only through `Scene::mesh_mut()`, which flags it dirty; `State::render` then re-uploads the GPU buffers.
- **UI Returns Intent:** `ui.rs` never mutates app state. `UiState::draw` returns what the user asked for (e.g. `Option<EditMode>`) and `main.rs` applies it.

## 4. File Structure (Current)
```text
polycraft/
├── Cargo.toml      
├── docs/
│   ├── AI_DIGEST.md
│   └── ARCHITECTURE.md
└── src/
    ├── main.rs     # winit loop, wgpu State
    ├── mesh.rs     # Half-Edge Mesh data structure and GPU buffer triangulation
    ├── camera.rs   # Orbit camera math and Maya-style navigation controls
    ├── scene.rs    # Editable document: mesh + EditMode, dirty flag for GPU re-upload
    ├── ui.rs       # Spacebar Pie Menu, absolute radial widget positioning
    └── shader.wgsl # Vertex/fragment shader with flat shading lighting

## 5. Conventions & Invariants
- Right-handed, Y-up. Front faces wind counter-clockwise (`FrontFace::Ccw`, back-face culling on).
- Flat shading: one normal per face; vertices are un-welded when building render buffers.
- `VertexId`, `HalfEdgeId`, `FaceId` are `usize` newtypes indexing the `Vec`s in `HalfEdgeMesh`.

## 6. Status (update at the end of every task)
- **Working:** Flat-shaded cube from the half-edge mesh; Maya camera (Alt+LMB orbit, Alt+MMB pan, Alt+RMB/scroll zoom); Space pie menu UI.
- **Implemented, not yet run-tested:** Pie menu: hold Space, flick toward a slot and release to set `EditMode` (Object/Vertex/Edge/Face); releasing in the center cancels, and clicking a slot also works. The mode is shown in the "PolyCraft Tools" window. `Scene` owns the mesh, with a GPU re-upload path.
- **Not built yet:** Edge/vertex display, picking/selection, transforms, edit operations, undo, FBX export. `EditMode` is stored but does not change behavior yet, and nothing edits the mesh yet.
- **Known limitations:** `u16` index buffer (about 16k quads max); face normals use the first three vertices (wrong for concave or degenerate faces); the graphics backend is chosen by `Instance::default()`, not forced to DX12; every `mesh_mut()` call recreates both GPU buffers, so before interactive vertex dragging, split the dirty flag into topology (recreate buffers) and positions (`queue.write_buffer`).
- **Next (proposed):** Draw edges/vertices per `EditMode`, then picking/selection.

## 7. Handoff Checklist
1. `cargo check` and `cargo test` pass and the app launches.
2. Update Section 4 (file structure) and Section 6 (status).
3. Commit with a descriptive message.

Roles for now: Claude writes; Gemini reviews each change from a diff plus a review brief.
