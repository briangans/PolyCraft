# PolyCraft - AI Context Digest

## 1. Project Vision
PolyCraft is a trimmed-down Maya clone written in Rust, built to leverage decades of learned muscle memory (e.g., V-key vertex snapping, D-key pivot editing, Q/W/E/R tool switching).
- **Core Features:** Polygon modeling, FBX export, pie menus, manipulator gizmos.
- **Explicit Non-Goals:** Advanced rendering (mental ray/Vray), physics, NURBS, Output Window, complex construction history beyond immediate operations.

## 2. Tech Stack & Dependencies
- **Language:** Rust 2021 edition (Windows / MSVC ABI).
- **Graphics:** `wgpu` (0.20, DirectX 12 backend), `winit` (0.29), `pollster`, `bytemuck`, `cgmath` (0.18).
- **UI:** `egui` / `egui-wgpu` / `egui-winit` (0.28, immediate mode for pie menus and tool hints).
- **Geometry:** Custom Half-Edge mesh structure using Arena/Index vectors (`usize`) to bypass borrow checker cycles.
- **Dependency Coupling:** Versions in `Cargo.toml` are tightly coupled (e.g. `egui` 0.28 pairs strictly with `wgpu` 0.20). Upgrading one usually requires upgrading all. Do not add external crates without checking if the existing stack already covers the requirement.

## 3. Architectural Rules
- **Context Management:** Code is modularized. Keep files focused; prefer creating a new module over letting `main.rs` grow excessively.
- **State Management:** Strict decoupling. The `State` struct encapsulates all `wgpu` initialization and render pipeline dispatch. The `winit` event loop strictly routes OS events.
- **Render Passes:** 3D Scene renders first with Z-buffer depth. `egui` renders second with `LoadOp::Load` to draw on top.
- **Scene Ownership:** `scene.rs` owns the editable document: the mesh, the `EditMode`, `Selection`, and active `Tool`. Change them only through the `Scene` API (`mesh_mut()`, `set_mode()`, `set_tool()`, `select_at()`, `translate_vertices()`). Document mutations flag `dirty = true`; `State::render` calls `take_dirty()` to rebuild GPU buffers.
- **UI Returns Intent:** `ui.rs` never mutates application state directly. `UiState::draw` returns what the user asked for (`UiResponse { mode, tool }`) and `main.rs` applies it.
- **Pure Logic Stays Testable:** Mesh topology, picking, selection, camera math, and gizmo calculations never touch `wgpu` or `winit`, and have unit tests next to the code (`mod tests`).
- **Resilient Error Handling:** Recoverable errors must be handled gracefully rather than panicked; the application must not crash on surface resizes, minimize events, or transient GPU adapter conditions.

## 4. File Structure & Responsibilities

```text
polycraft/
├── Cargo.toml
├── docs/
│   └── AI_DIGEST.md       # Central architecture, rules, workflow, and progress digest
└── src/
    ├── main.rs       # Entry point, winit loop, wgpu State, Vertex layout, buffer upload, hotkey routing
    ├── mesh.rs       # Half-edge mesh (HalfEdgeMesh), topology queries, Newell normals, triangulation
    ├── camera.rs     # Orbit camera, Maya navigation (Alt+LMB/MMB/RMB), view basis for picking
    ├── scene.rs      # Editable document: mesh + EditMode + Selection + Tool, dirty flag, translate_vertices
    ├── selection.rs  # Selected components per kind, Maya click semantics
    ├── picking.rs    # Screen position -> ray; face/vertex/edge/object hit tests, ray_segment
    ├── edit_mode.rs  # EditMode enum (Object/Vertex/Edge/Face)
    ├── tool.rs       # Tool enum (Select/Translate/Rotate/Scale), Maya Q/W/E/R hotkeys
    ├── gizmo.rs      # Transform gizmo math (pure): pivot, constant-pixel arm length, axis picking, drag projection
    ├── overlay.rs    # Wireframe overlay, edge highlights, vertex square dots, gizmo line vertices
    ├── ui.rs         # Spacebar/RMB marking pie menu, PolyCraft Tools window; returns UiResponse
    └── shader.wgsl   # WGSL vertex/fragment shaders with flat shading, vertex dots, and unlit line passes
```

## 5. Conventions & Invariants
- **Coordinate Space:** Right-handed, Y-up. Front faces wind counter-clockwise (`FrontFace::Ccw`, back-face culling on).
- **Flat Shading:** One normal per face (Newell's method in `face_normal`). Render vertices are un-welded, and `RenderMesh::vertex_faces` records which face each one came from. Faces are triangulated as fans (must remain convex).
- **Component IDs:** `VertexId`, `HalfEdgeId`, `FaceId` are `usize` newtypes indexing the `Vec`s in `HalfEdgeMesh`. `EdgeId` is an undirected pair of vertex indices (smaller first), so both half-edges of an edge produce the identical id.
- **Screen-Space Tolerances:** Picking and gizmo arm tolerances are measured in screen pixels (`VERTEX_PICK_RADIUS_PIXELS`, `EDGE_PICK_RADIUS_PIXELS`, `GIZMO_PICK_TOLERANCE_PIXELS`) via `world_per_pixel()`, maintaining consistent feel at any zoom level. Components behind nearer faces are occluded; gizmo arms bypass depth occlusion (`Always` pass).
- **Selection Semantics:** One selection set per component kind; switching `EditMode` drops other kinds. Plain click replaces selection, Shift+click toggles, and plain click on empty space clears (Maya semantics).
- **Tool Independence:** Active tool (`Tool::Select`, `Tool::Translate`, `Tool::Rotate`, `Tool::Scale`) is decoupled from `EditMode`. Switching modes preserves tool; switching tools preserves selection.
- **GPU Vertex Layout Sync:** The GPU `Vertex` struct in `src/main.rs` derives `bytemuck::Pod`/`Zeroable` and must stay strictly in sync with `wgpu::VertexBufferLayout` and `@location(n)` declarations in `shader.wgsl`.

## 6. Build, Verify & Git Workflow

### Essential Commands
| Command | Description |
| --- | --- |
| `cargo check` | Fast feedback while editing |
| `cargo test` | Runs all unit tests across modules |
| `cargo clippy --all-targets` | Linting (keep at 0 warnings) |
| `cargo fmt` | Format codebase before finishing |
| `cargo run` | Launch application viewport (requires display/GPU) |

### Verification Rule
Always run `cargo check` and `cargo test` after modifying Rust code. Do not report a change as complete if tests fail or if there are compiler/clippy warnings.

### Git Rules
- Commit messages must describe the user-visible change or architectural milestone.
- Never force-push or rewrite git history.
- Push operations are strictly performed by the user; agents commit locally.

## 7. Status (update at the end of every task)
- **Working (run-tested):**
  - Flat-shaded cube from the half-edge mesh with Newell normals.
  - Maya camera navigation: Alt+LMB orbit, Alt+MMB pan (consistent screen-space tracking), Alt+RMB / scroll zoom.
  - Marking pie menu: hold RMB (without Alt) or Spacebar, flick toward a slot and release to switch `EditMode` (Edge=Up, Face=Down, Vertex=Left, Object=Right).
  - Viewport click-to-select: plain LMB click casts NDC ray; Shift+click toggles selection; empty click clears.
  - Visual overlays: selected faces tinted Maya warm orange, selected object tinted Maya green; wireframe edges with depth bias (cyan for selected); vertex square dots in Vertex mode (blue for unselected, bright yellow for selected).
  - Tool system: Q (Select), W (Move), E (Rotate), R (Scale) hotkeys and "PolyCraft Tools" egui panel.
  - Transform Gizmo (W key Move tool): constant-pixel-size 3D coordinate axes rendered with `Always` depth-pass at selection pivot. Hovering an axis highlights it in bright Maya yellow. Clicking and dragging an axis with LMB translates selected components (vertices, edges, faces, or object) live along the 3D axis with real-time normal and wireframe recalculation. Releasing LMB commits translation; pressing Escape cancels and restores initial positions.
- **Implemented and unit-tested:** All 85 unit tests passing across camera, mesh, picking, scene, selection, tool, ui, overlay, and gizmo.
  - `gizmo.rs` (Iterations 1 & 3): `GizmoAxis` enum (X, Y, Z), selection centroid (`pivot()`), constant-screen-size arm length (`axis_world_length()`), `translate_arms()` with active yellow highlight (`COLOR_AXIS_ACTIVE`), pure axis picking (`pick_gizmo_axis()`) via `picking::ray_segment`, 1D ray-axis projection (`project_ray_onto_axis()`), and unique component vertex collection (`selected_vertices()`).
  - `tool.rs` and `scene.rs` (Iteration 2): `Tool` enum (`Select`, `Translate`, `Rotate`, `Scale`) decoupled from `EditMode`. Maya hotkeys `Q`, `W`, `E`, `R` wired in `main.rs` via `WindowEvent::KeyboardInput` (safe from UI focus). `Scene::translate_vertices()` applies displacement to vertices and marks document dirty.
  - Viewport & Event loop (Iteration 3): `main.rs` manages `drag_state` and `hovered_axis`. Moving mouse updates hover highlight or translates components along the chosen axis. Escape key restores initial positions.
- **Not built yet:** Undo/redo stack, edit operations (extrude, bevel, cut), FBX export.
- **Known limitations:** `u16` index buffer (about 16k quads max); fan triangulation needs convex faces; graphics backend chosen by `Instance::default()`; dirty flag re-uploads both mesh and overlay GPU buffers (split positions before high-frequency drag). `cargo clippy --all-targets` passes with 0 code warnings.
- **Next Milestone:**
  - **Undo/Redo stack:**
    - Implement a history stack (`UndoStack`) recording actions or mesh snapshots before modifications.
    - Wire `Ctrl+Z` (Undo) and `Ctrl+Y` / `Ctrl+Shift+Z` (Redo) in `WindowEvent::KeyboardInput`.
  - Followed by Polygon Modeling Operations (face extrude, bevel, loop split).

## 8. Handoff Checklist
1. `cargo check`, `cargo test` and `cargo clippy --all-targets` pass (no new warnings), and the app launches.
2. Update Section 4 (file structure) and Section 7 (status).
3. Commit with a descriptive message.

Roles: All AI agents and human contributors adhere strictly to this digest for architecture, conventions, and status updates.
