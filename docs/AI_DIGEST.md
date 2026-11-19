# PolyCraft - AI Context Digest

## 1. Project Vision
PolyCraft is a trimmed-down Maya clone written in Rust, built to leverage decades of learned muscle memory (e.g., V-key vertex snapping, D-key pivot editing, Q/W/E/R tool switching).
- **Core Features:** Polygon modeling, FBX export, pie menus, manipulator gizmos.
- **Explicit Non-Goals:** Advanced rendering (mental ray/Vray), physics, fluids, particles, NURBS, Output Window, complex construction history beyond immediate operations.

## 2. Tech Stack & Dependencies
- **Language:** Rust 2021 edition (Windows / MSVC ABI).
- **Graphics:** `wgpu` (0.20, DirectX 12 backend), `winit` (0.29), `pollster`, `bytemuck`, `cgmath` (0.18).
- **UI:** `egui` / `egui-wgpu` / `egui-winit` (0.28, immediate mode for pie menus and tool hints).
- **Geometry:** Custom Half-Edge mesh structure using Arena/Index vectors (`usize`) to bypass borrow checker cycles.
- **Dependency Coupling:** Versions in `Cargo.toml` are tightly coupled (e.g. `egui` 0.28 pairs strictly with `wgpu` 0.20). Upgrading one usually requires upgrading all. Do not add external crates without checking if the existing stack already covers the requirement.

## 3. Architectural Rules
- **Context Management:** Code is modularized. Keep files focused; prefer creating a new module over letting `main.rs` grow excessively.
- **State Management:** Strict decoupling. `State` (in `renderer.rs`, not `main.rs`) encapsulates all `wgpu` initialization, render pipeline dispatch, and viewport interaction state. The `winit` event loop in `main.rs` strictly routes OS events and never touches `State` fields. Multi-field operations go through `State` interaction seams - `process_mouse_motion`, `process_scroll`, `handle_cursor_moved`, `handle_mouse_input`, `cancel_active_drag`, `undo`, `redo`, `handle_hotkey`, `set_alt_pressed`, `set_shift_pressed`, `set_ctrl_pressed` - because a seam can split-borrow `self.camera_controller` and `self.camera` while two separate accessors each borrow all of `state` and fail to compile (E0499).
- **Scene Ownership:** `scene.rs` owns the editable document: the mesh, the `EditMode`, `Selection`, active `Tool`, and the undo/redo `HistoryStack`. Change them only through the `Scene` API (`mesh_mut()`, `set_mode()`, `set_tool()`, `select_at()`, `translate_vertices()`, `begin_action()`/`end_action()`/`cancel_action()`, `undo()`/`redo()`). Document mutations flag `dirty = true`; `State::render` calls `take_dirty()` to rebuild GPU buffers.
- **UI Returns Intent:** `ui.rs` never mutates application state directly. `UiState::draw` takes what to display as a `UiContext` struct and returns what the user asked for (`UiResponse { mode, tool, undo, redo }`); `renderer.rs` applies it.
- **Hotkeys Stay a Pure Map:** the Maya keymap lives in `renderer.rs::hotkey_for(key, ctrl, shift) -> Option<Hotkey>`, a free function that `State::handle_hotkey` carries out. Binding a key is adding a match arm; asserting one is a unit test - no window, no GPU context. `main.rs` only decides *whether* to offer the key (skipping it when the UI consumed the event or owns keyboard focus).
- **Pure Logic Stays Testable:** Mesh topology, picking, selection, camera math, gizmo calculations, and the `HistoryStack` never touch `wgpu` or `winit`, and have unit tests next to the code (`mod tests`). `HistoryStack<T>` is generic over the snapshot type, so stack semantics are tested with `T = i32`.
- **Render Passes:** 3D Scene renders first with Z-buffer depth. `egui` renders second with `LoadOp::Load` to draw on top.
- **Resilient Error Handling:** Recoverable errors must be handled gracefully rather than panicked; the application must not crash on surface resizes, minimize events, or transient GPU adapter conditions.

## 4. File Structure & Responsibilities

```text
polycraft/
├── Cargo.toml
├── docs/
│   └── AI_DIGEST.md   # Central architecture, rules, workflow, and progress digest
└── src/
    ├── main.rs         # Entry point: winit event loop, routes OS events into State, Maya hotkeys
    ├── renderer.rs     # GPU renderer: wgpu init, State (camera/scene/egui/drag), render passes, event seams
    ├── render_types.rs # Shared GPU layouts: GpuVertex (all passes) + CameraUniform (camera bind group)
    ├── mesh.rs         # Half-edge mesh (HalfEdgeMesh), topology queries, Newell normals, triangulation
    ├── camera.rs       # Orbit camera, Maya navigation (Alt+LMB/MMB/RMB), view basis for picking
    ├── scene.rs        # Editable document: mesh + EditMode + Selection + Tool, dirty flag, undo/redo, HistoryStack
    ├── history.rs      # Undo/redo HistoryStack<T> (pure): begin/commit/cancel action capture, bounded undo/redo stacks
    ├── selection.rs    # Selected components per kind, Maya click semantics
    ├── picking.rs      # Screen position -> ray (Ray::from_screen); face/vertex/edge/object hit tests, ray_segment
    ├── edit_mode.rs    # EditMode enum (Object/Vertex/Edge/Face)
    ├── tool.rs         # Tool enum (Select/Translate/Rotate/Scale), Maya Q/W/E/R hotkeys
    ├── gizmo.rs        # Transform gizmo math (pure): GizmoAxis, pivot, arm length, axis picking, drag projection
    ├── overlay.rs      # Wireframe overlay, edge highlights, vertex square dots, gizmo line vertices
    ├── ui.rs           # Spacebar/RMB marking pie menu, PolyCraft Tools window; returns UiResponse
    └── shader.wgsl     # WGSL vertex/fragment shaders with flat shading, vertex dots, and unlit line passes
```

## 5. Conventions & Invariants
- **Coordinate Space:** Right-handed, Y-up. Front faces wind counter-clockwise (`FrontFace::Ccw`, back-face culling on).
- **Flat Shading:** One normal per face (Newell's method in `face_normal`). Render vertices are un-welded, and `RenderMesh::vertex_faces` records which face each one came from. Faces are triangulated as fans (must remain convex).
- **Component IDs:** `VertexId`, `HalfEdgeId`, `FaceId` are `usize` newtypes indexing the `Vec`s in `HalfEdgeMesh`. `EdgeId` is an undirected pair of vertex indices (smaller first), so both half-edges of an edge produce the identical id.
- **Screen-Space Tolerances:** Picking and gizmo arm tolerances are measured in screen pixels (`VERTEX_PICK_RADIUS_PIXELS`, `EDGE_PICK_RADIUS_PIXELS`, `GIZMO_PICK_TOLERANCE_PIXELS`) via `world_per_pixel()`, maintaining consistent feel at any zoom level. Components behind nearer faces are occluded; gizmo arms bypass depth occlusion (`Always` pass).
- **Selection Semantics:** One selection set per component kind; switching `EditMode` drops other kinds. Plain click replaces selection, Shift+click toggles, and plain click on empty space clears (Maya semantics).
- **Tool Independence:** Active tool (`Tool::Select`, `Tool::Translate`, `Tool::Rotate`, `Tool::Scale`) is decoupled from `EditMode`. Switching modes preserves tool; switching tools preserves selection.
- **GPU Vertex Layout Sync:** `GpuVertex` (and `CameraUniform`) live in `src/render_types.rs` and derive `bytemuck::Pod`/`Zeroable`. One struct serves every pass - `position` plus `normal`, which carries a surface normal for shaded faces but a screen-pixel offset for billboarded vertex dots - so it must stay strictly in sync with the single `wgpu::VertexBufferLayout` and the `@location(n)` declarations in `shader.wgsl`. `overlay.rs` exposes it as `OverlayVertex` via a type alias; do not reintroduce a second vertex struct.

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
  - Undo / Redo: `Ctrl+Z` steps back, `Ctrl+Y` and `Ctrl+Shift+Z` step forward. A whole gizmo drag is a single step; a press-and-release that moved nothing records no step, so `Ctrl+Z` after a stray click lands on the previous real action. Undo restores geometry and selection but never the `EditMode` or `Tool`. The Undo / Redo buttons in the "PolyCraft Tools" panel name the action each would perform and grey out when the history runs out.
- **Implemented and unit-tested:** All 119 unit tests passing across camera, mesh, picking, scene, selection, tool, ui, overlay, gizmo, history, and renderer (keymap). `picking::Ray::from_screen` (top-left pixel origin to NDC ray, used by `State::cursor_ray`) is covered alongside the other pure picking math.
  - `gizmo.rs` (Iterations 1 & 3): `GizmoAxis` enum (X, Y, Z), selection centroid (`pivot()`), constant-screen-size arm length (`axis_world_length()`), `translate_arms()` with active yellow highlight (`COLOR_AXIS_ACTIVE`), pure axis picking (`pick_gizmo_axis()`) via `picking::ray_segment`, 1D ray-axis projection (`project_ray_onto_axis()`), and unique component vertex collection (`selected_vertices()`).
  - `tool.rs` and `scene.rs` (Iteration 2): `Tool` enum (`Select`, `Translate`, `Rotate`, `Scale`) decoupled from `EditMode`. Maya hotkeys `Q`, `W`, `E`, `R` wired in `main.rs` via `WindowEvent::KeyboardInput` (safe from UI focus). `Scene::translate_vertices()` applies displacement to vertices and marks document dirty.
  - Viewport & Event loop (Iteration 3): `renderer.rs` `State` owns `drag_state` and `hovered_axis` (a private `DragState` struct). Moving the mouse updates the hover highlight or translates components along the chosen axis; Escape restores initial positions via `cancel_active_drag()`.
  - Module Extraction (after Iteration 3): the roughly 1000-line `main.rs` was split into `renderer.rs` (wgpu init, `State`, all render passes, and the `process_mouse_motion`/`process_scroll`/`handle_cursor_moved`/`handle_mouse_input` interaction seams) and `render_types.rs` (`GpuVertex` + `CameraUniform`), leaving `main.rs` as the winit event loop and hotkey routing only. `overlay.rs` now aliases `OverlayVertex` to `GpuVertex` instead of declaring its own struct; dead accessors left behind by the split (`camera_mut`, `egui_ctx`) were removed.
  - `history.rs` + `scene.rs` (Undo/Redo iteration): `HistoryStack<T>` is pure and generic over the snapshot type, so stack semantics are tested with `T = i32` - ordering, the bounded depth, redo cleared on a new action, and no-op actions dropped. `Scene` owns a `HistoryStack<DocumentSnapshot>` (mesh + selection; `EditMode` and `Tool` are deliberately not versioned). Actions are captured explicitly with `begin_action(label)` / `end_action()` / `cancel_action()` rather than per mutation, which is what makes one drag one step and keeps snapshot count proportional to actions, not frames. `restore()` re-clips the selection through `retain_mode` plus live index/id bounds so undo can never resurrect a component kind the current mode cannot hold. `HalfEdgeMesh`/`Selection` and their records gained `Clone + PartialEq` (and `Debug`/`Eq` where meaningful) to support snapshotting and no-op detection.
  - `renderer.rs` (Undo/Redo iteration): the Q/W/E/R + Escape + `Ctrl+Z` / `Ctrl+Y` / `Ctrl+Shift+Z` keymap moved out of `main.rs` into the pure `hotkey_for(key, ctrl, shift) -> Option<Hotkey>`, executed by the `State::handle_hotkey` seam. `State` tracks `ctrl_pressed` from `ModifiersChanged`. `State::undo`/`redo` abort an in-flight drag before touching history, since a live drag holds vertex ids and initial positions from the state being rolled back. Drag history opens on LMB press over a gizmo axis (`drag_action_label` yields "Move Vertex" / "Move Edge" / "Move Face" / "Move") and commits on release.
- **Not built yet:** Edit operations (extrude, bevel, cut), FBX export.
- **Known limitations:** `u16` index buffer (about 16k quads max); fan triangulation needs convex faces; graphics backend chosen by `Instance::default()`; dirty flag re-uploads both mesh and overlay GPU buffers (split positions before high-frequency drag). `cargo clippy --all-targets` passes with 0 code warnings.
- **Next Milestone:**
  - **Polygon Modeling Operations:** face extrude, bevel, loop split / edge cut.
    - Build each operation as a single `Scene` method that returns the new topology, wrapped by the caller in `begin_action("Extrude Face")` / `end_action()` so it lands in history as one step for free.
    - Topology edits invalidate component ids, so every operation must decide what the selection becomes afterwards (Maya keeps the newly created faces) - and `Scene::restore` already bounds-checks a restored selection against live ids.
    - `u16` indices bound how deep history can go on a dense mesh; if extrude pushes past the cap, address the index buffer before raising `HISTORY_DEPTH`.
  - Followed by FBX export.

## 8. Handoff Checklist
1. `cargo check`, `cargo test` and `cargo clippy --all-targets` pass (no new warnings), and the app launches.
2. Update Section 4 (file structure) and Section 7 (status).
3. Commit with a descriptive message.

Roles: All AI agents and human contributors adhere strictly to this digest for architecture, conventions, and status updates.
