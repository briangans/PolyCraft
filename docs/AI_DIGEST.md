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
    ├── gizmo.rs      # Transform gizmo math (pure): selection pivot, axis colours, constant-pixel arm length
    ├── overlay.rs    # Wireframe overlay, edge highlights, vertex 3D handles
    ├── ui.rs         # Spacebar pie menu (flick and release), Tools window
    └── shader.wgsl   # Vertex/fragment shader with flat shading and unlit overlay pass
```

## 5. Conventions & Invariants
- Right-handed, Y-up. Front faces wind counter-clockwise (`FrontFace::Ccw`, back-face culling on).
- Flat shading: one normal per face (Newell's method in `face_normal`). Render vertices are un-welded, and `RenderMesh::vertex_faces` records which face each one came from. Faces are triangulated as fans, so they must be convex.
- `VertexId`, `HalfEdgeId`, `FaceId` are `usize` newtypes indexing the `Vec`s in `HalfEdgeMesh`. `EdgeId` is an undirected pair of vertex indices (smaller first), so both half-edges of an edge give the same id.
- Picking tolerances are in screen pixels (`VERTEX_PICK_RADIUS_PIXELS`, `EDGE_PICK_RADIUS_PIXELS`), so they feel the same at any zoom. Components behind a nearer face are not pickable.
- Selection is one set per kind, and switching `EditMode` drops the other kinds. A plain click replaces the selection, Shift+click toggles, and a plain click on empty space clears it (Maya behavior).
- `Vertex` in `main.rs` (position, normal, color) must match the `@location`s in `shader.wgsl`.

## 6. Status (update at the end of every task)
- **Working (run-tested):** Flat-shaded cube from the half-edge mesh; Maya camera (Alt+LMB orbit, Alt+MMB pan, Alt+RMB/scroll zoom); Marking menu: hold RMB (without Alt, true Maya style) or Space, flick toward a slot and release to switch `EditMode` (Edge=Up, Face=Down, Vertex=Left, Object=Right). Viewport click-to-select: plain left click (no Alt) in viewport outside UI casts an NDC picking ray and selects/clears components; Shift+click toggles selection (Maya semantics). Selection visual highlights: selected faces tinted Maya warm orange, selected object tinted Maya green; wireframe and component overlays in `overlay.rs` render clean edges (cyan for selected edges, green for selected object) and screen-aligned flat square vertex dots in Vertex mode (6x6 blue/purple dots for unselected, 9x9 bright yellow dots for selected) with depth testing and bias. "PolyCraft Tools" window displays the current mode, active selection count, and RMB/Space menu hint.
- **Implemented and unit-tested:** All 69 unit tests passing across camera, mesh, picking, scene, selection, ui, overlay, and gizmo. `gizmo.rs` (Iteration 1) provides the pure translate-gizmo math: `pivot()` (selection centroid per mode), `axis_world_length()` (constant on-screen arm derived from a pixel length via the shared `picking::world_per_pixel`), the `AXES` table with Maya axis colours, and `AXIS_LENGTH_PIXELS`. Not yet wired to input, drawing, or dragging.
- **Not built yet:** Transforms/manipulator gizmos (W/E/R) are started (pure math done; no drawing/drag/activation yet), edit operations (extrude, bevel, cut), undo/redo, FBX export.
- **Known limitations:** `u16` index buffer (about 16k quads max); fan triangulation needs convex faces; graphics backend chosen by `Instance::default()`; dirty flag re-uploads both mesh and overlay GPU buffers (split positions before high-frequency drag). `cargo clippy --all-targets` passes with 0 code warnings.
- **Next (gizmo plan, Iteration 1 DONE):** Transform gizmo is being built in small testable steps. LOCKED DECISIONS: (a) the active tool (Move/Rotate/Scale, bound to W/E/R) is a SEPARATE concept from `EditMode` — a tool like `Scene::tool` set alongside selection mode; (b) the gizmo is CONSTANT SCREEN SIZE (its world arm is derived from `AXIS_LENGTH_PIXELS`, hence the `world_per_pixel` dependency). Iteration 1 (this commit): `gizmo.rs` pure math + tests. Iteration 2 (next): DRAW the three coloured axes at `gizmo::pivot` when something is selected, reusing the overlay line pipeline; because the gizmo is constant-screen-size, this likely needs a small `shader.wgsl` change so lines can take a pixel offset like the vertex-dot `vs_dot` pass. Iteration 3: activate with W and drag to translate selected vertices along the hit axis (drag math stays in `gizmo.rs`; needs a `KeyboardInput` handler in `main.rs`, which does not exist yet). Then undo/redo stack.

## 7. Handoff Checklist
1. `cargo check`, `cargo test` and `cargo clippy --all-targets` pass (no new warnings), and the app launches.
2. Update Section 4 (file structure) and Section 6 (status).
3. Commit with a descriptive message.

Roles: Gemini audits code occasionally, and updates documentation if needed, directly adhering to the architecture. Other AI models also author code.