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
    ├── main.rs       # winit loop, wgpu State, GPU Vertex layout, mesh buffer upload, hotkey routing
    ├── mesh.rs       # Half-Edge mesh, component ids, topology queries, triangulation for rendering
    ├── camera.rs     # Orbit camera, Maya-style navigation, view basis for picking
    ├── scene.rs      # Editable document: mesh + EditMode + Selection + Tool, dirty flag
    ├── selection.rs  # Selected components per kind, Maya click semantics
    ├── picking.rs    # Screen position -> ray; face/vertex/edge/object hit tests
    ├── edit_mode.rs  # EditMode enum (Object/Vertex/Edge/Face)
    ├── tool.rs       # Tool enum (Select/Translate/Rotate/Scale), Maya Q/W/E/R shortcuts
    ├── gizmo.rs      # Transform gizmo math (pure): selection pivot, axis colours, constant-pixel arm length
    ├── overlay.rs    # Wireframe overlay, edge highlights, vertex 3D handles, gizmo line vertices
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
- **Working (run-tested):** Flat-shaded cube from the half-edge mesh; Maya camera (Alt+LMB orbit, Alt+MMB pan, Alt+RMB/scroll zoom); Marking menu: hold RMB (without Alt, true Maya style) or Space, flick toward a slot and release to switch `EditMode` (Edge=Up, Face=Down, Vertex=Left, Object=Right). Viewport click-to-select: plain left click (no Alt) in viewport outside UI casts an NDC picking ray and selects/clears components; Shift+click toggles selection (Maya semantics). Selection visual highlights: selected faces tinted Maya warm orange, selected object tinted Maya green; wireframe and component overlays in `overlay.rs` render clean edges (cyan for selected edges, green for selected object) and screen-aligned flat square vertex dots in Vertex mode (6x6 blue/purple dots for unselected, 9x9 bright yellow dots for selected) with depth testing and bias. "PolyCraft Tools" window displays current mode, active tool, selection count, and hotkey hints.
- **Implemented and unit-tested:** All 76 unit tests passing across camera, mesh, picking, scene, selection, tool, ui, overlay, and gizmo.
  - `gizmo.rs` (Iteration 1) provides pure translate-gizmo math: `pivot()` (selection centroid per mode), `axis_world_length()` (constant on-screen arm derived from a pixel length via `world_per_pixel`), the `AXES` table with Maya axis colours (X=Red, Y=Green, Z=Blue), and `translate_arms()`.
  - `tool.rs` and `scene.rs` (Iteration 2): `Tool` enum (`Select`, `Translate`, `Rotate`, `Scale`) decoupled from `EditMode`. Hotkeys `Q` (Select), `W` (Move), `E` (Rotate), and `R` (Scale) wired in `main.rs` via `WindowEvent::KeyboardInput` (safe from UI focus).
  - Overlay & Gizmo rendering (Iteration 2): `gizmo_pipeline` renders constant-pixel-size 3D coordinate axes at `gizmo::pivot` whenever `tool == Tool::Translate` and there is an active selection. Always depth test ensures axes are visible through front faces as Maya does. Axes disappear cleanly in Select mode (`Q`) or when selection is cleared.
- **Not built yet:** Transforms/manipulator dragging (Iteration 3), edit operations (extrude, bevel, cut), undo/redo, FBX export.
- **Known limitations:** `u16` index buffer (about 16k quads max); fan triangulation needs convex faces; graphics backend chosen by `Instance::default()`; dirty flag re-uploads both mesh and overlay GPU buffers (split positions before high-frequency drag). `cargo clippy --all-targets` passes with 0 code warnings.
- **Next (gizmo plan, Iterations 1 & 2 DONE):**
  - **Iteration 3 (next):** Hit-test gizmo axes and drag to translate selected components along the chosen axis.
    1. Pure picking math in `gizmo.rs`: `pick_gizmo_axis(pivot, length, ray, tolerance_pixels, viewport_height) -> Option<usize>` using `picking::ray_segment` (depth test disabled).
    2. Pure drag math in `gizmo.rs`: project ray offset onto chosen axis $\vec{u}$ to produce world-space translation displacement $\Delta \vec{p}$.
    3. Input handling in `main.rs`: on LMB press over an axis, enter active drag state; on mouse move, apply displacement to selected vertices in `mesh_mut()`; on LMB release, complete translation.
  - Then undo/redo stack.

## 7. Handoff Checklist
1. `cargo check`, `cargo test` and `cargo clippy --all-targets` pass (no new warnings), and the app launches.
2. Update Section 4 (file structure) and Section 6 (status).
3. Commit with a descriptive message.

Roles: Gemini audits code occasionally, and updates documentation if needed, directly adhering to the architecture. Other AI models also author code.