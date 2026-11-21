# PolyCraft - AI Context Digest

## 0. Model & Harness Configuration
This digest is written to be pasted whole into one prompt for a locally-run coding model. Measured host: **RTX 3070 Ti, 8 GB VRAM** (~2.4 GiB held by the desktop, ~5.5 GiB free) plus **128 GB DDR4-3600** in two channels (~57 GB/s peak, ~46 GB/s effective) and 16 cores / 32 threads.
- **Model:** Qwen3-Coder-30B-A3B-Instruct, `Q4_K_M` GGUF - 30.5B parameters, 3.3B active per token, 48 layers, 262,144 native context. The 18.56 GB file cannot fit 8 GB of VRAM, so most layers run from system RAM: `--n-gpu-layers 5 --flash-attn --ctx-size 65536 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --jinja --mlock --threads 16 --keep 6000 --cache-reuse 256`.
- **Sizing math:** grouped-query attention gives 4 KV heads at `head_dim` 128 over 48 layers, so the KV cache is 2 x 4 x 128 x 48 x 2 bytes = **96 KiB per context token** in FP16, 51 KiB at `q8_0` - 0.8 GiB at 16k, 1.6 GiB at 32k, 3.2 GiB at 65k. Each layer is 17.28 GiB / 48 = **0.36 GiB**. After the desktop, compute buffers, and the viewport (the app itself measures only ~75 MB), roughly 5 GiB is spare, so 65k context leaves room for about five layers on the GPU; `--ctx-size 32768` affords about eight. Raise `--n-gpu-layers` in steps of two while `nvidia-smi` still reports free memory - spilling into shared memory costs far more than the layers gained. The server's context must be at least what the harness is set to, or requests are truncated.
- **Throughput:** generation is bound by DDR4 bandwidth rather than the GPU - roughly 2 GB of active weights per token over ~46 GB/s gives an order-of-magnitude **20-28 tok/s** at Q4_K_M. A smaller quant is proportionally faster (Q3_K_M is ~14 GB) at some quality cost. Measure prompt processing and generation separately before trusting any figure.
- **Context budget: 65,536 tokens.** Keep this digest **byte-identical and at the same position** in every prompt so the prefix KV cache is reused and it is never re-prefilled - on this host a fresh ~4.4k-token prefill each turn is the dominant cost - and `--keep` stops it being evicted. Keep the digest near 6k tokens.
- **Flag spellings drift between llama.cpp builds** (recent builds want `--flash-attn on`), and no llama.cpp binary is currently on this machine's PATH. Confirm every flag with `llama-server --help` against the engine actually in use before treating the line above as runnable.
- **Do not** feed the whole repo to the model. Use the §4 map to name the exact file(s) to read, then supply only those.

## 1. Project Vision
PolyCraft is a trimmed-down Maya clone written in Rust, built to leverage decades of learned muscle memory (e.g., V-key vertex snapping, D-key pivot editing, Q/W/E/R tool switching).
- **Core Features:** Polygon modeling, FBX export, pie menus, manipulator gizmos.
- **Explicit Non-Goals:** Advanced rendering (mental ray/Vray), physics, fluids, particles, NURBS, Output Window, complex construction history beyond immediate operations.

## 2. Tech Stack & Dependencies
- **Language:** Rust 2021 edition (Windows / MSVC ABI). Toolchain in use is rustc 1.95.x; there is **no `rust-version` key in `Cargo.toml`** and no `rust-toolchain.toml`, so the floor is only implied (the `Cargo.lock` is spec `version = 4`, which needs Cargo 1.85+). Do not assume 1.80 will build it.
- **Graphics:** `wgpu` 0.20 (lockfile 0.20.1, DirectX 12 backend), `winit` 0.29 (0.29.15), `pollster` 0.3, `bytemuck` 1.25, `cgmath` 0.18. `naga` sits at 0.20.0 through `wgpu`.
- **UI:** `egui` / `egui-wgpu` / `egui-winit` 0.28 (0.28.1, immediate mode for pie menus and tool hints).
- **Geometry:** A custom half-edge mesh stored as three plain `Vec`s (`vertices`, `half_edges`, `faces`) addressed by `usize` newtypes (`VertexId`, `HalfEdgeId`, `FaceId`). There is **no `Arena`, `slotmap`, or `index_vec`** anywhere in the dependency tree - do not go looking for one. The `HashMap`/`HashSet` in `mesh.rs` are for `EdgeId` grouping only, not for storage.
- **Dependency Coupling:** Versions in `Cargo.toml` are tightly coupled (e.g. `egui` 0.28 pairs strictly with `wgpu` 0.20). Upgrading one usually requires upgrading all. Do not add external crates without checking if the existing stack already covers the requirement.

## 3. Architectural Rules
- **Context Management:** Code is modularized. Keep files focused; prefer creating a new module over letting `main.rs` grow excessively.
- **State Management:** Strict decoupling. `State` (in `renderer.rs`, not `main.rs`) encapsulates all `wgpu` initialization, render pipeline dispatch, and viewport interaction state. The `winit` event loop in `main.rs` strictly routes OS events and touches `State` only through its public surface - `window`, `size`, `on_window_event`, `wants_keyboard_input`, `handle_cursor_moved`, `handle_mouse_input`, `handle_hotkey`, `process_mouse_motion`, `process_scroll`, `set_alt_pressed`, `set_shift_pressed`, `set_ctrl_pressed`, `resize`, `render` - and never a field directly. Multi-field operations go through those interaction seams rather than pairs of accessors because a seam can split-borrow `self.camera_controller` and `self.camera` while two separate accessors each borrow all of `state` and fail to compile (E0499). `undo`/`redo`/`cancel_active_drag` are also `State` methods for the same reason.
- **Scene Ownership:** `scene.rs` owns the editable document: the mesh, the `EditMode`, `Selection`, active `Tool`, and the undo/redo `HistoryStack`. Change them only through the `Scene` API (`mesh_mut()`, `set_mode()`, `set_tool()`, `select_at()`, `translate_vertices()`, `begin_action()`/`end_action()`/`cancel_action()`, `undo()`/`redo()`). Document mutations flag `dirty = true`; `State::render` calls `take_dirty()` to rebuild GPU buffers.
- **UI Returns Intent:** `ui.rs` never mutates application state directly. `UiState::draw` takes what to display as a `UiContext` struct and returns what the user asked for (`UiResponse { mode, tool, undo, redo }`); `renderer.rs` applies it.
- **Hotkeys Stay a Pure Map:** the Maya keymap lives in `renderer.rs::hotkey_for(key, ctrl, shift) -> Option<Hotkey>`, a free function that `State::handle_hotkey` carries out. Binding a key is adding a match arm; asserting one is a unit test - no window, no GPU context. `main.rs` only decides *whether* to offer the key (skipping it when the UI consumed the event or owns keyboard focus).
- **Pure Logic Stays Testable:** Mesh topology, picking, selection, camera math, gizmo calculations, and the `HistoryStack` never touch `wgpu` or `winit`, and have unit tests next to the code (`mod tests`). `HistoryStack<T>` is generic over the snapshot type, so stack semantics are tested with `T = i32`.
- **Render Passes:** 3D Scene renders first with Z-buffer depth. `egui` renders second with `LoadOp::Load` to draw on top.
- **Resilient Error Handling:** Recoverable errors must be handled gracefully rather than panicked; the application must not crash on surface resizes, minimize events, or transient GPU adapter conditions. (Known gap: `State::new` and `main.rs` still `unwrap()` surface/adapter/shader setup and `gizmo::pivot` at drag start - 5 in `renderer.rs` outside `mod tests`, 2 in `main.rs`. Prefer `.expect("what failed and why")` in new code.)

### 3.5 Failure Modes
The gotchas that carry no compile error. The rules they protect are in §3 and §5; only the symptom is listed here.
- **Hand-building shaded vertices** instead of calling `HalfEdgeMesh::generate_render_buffers()` desynchronizes un-welded vertices from `vertex_faces`, so shading and wireframe disagree with the geometry.
- **Narrowing `RenderMesh::indices` from `u32` to `u16`** wraps silently onto the wrong vertices rather than failing, because vertices are un-welded per face and a modest model passes 65,536 of them. The pass binds `IndexFormat::Uint32`.
- **A mutation left out of `begin_action`/`end_action`** still renders correctly but never reaches history, so `Ctrl+Z` skips straight past that operation.
- **A `GpuVertex`/`CameraUniform` change made in only one place** (struct, layout/bind group, or `shader.wgsl` `@location`) yields a garbled or blank viewport with no compile error.
- **The gizmo is translate-only.** `upload_gizmo()` and `handle_mouse_input()` gate on `Tool::Translate`; removing that gate lets a drag start on an axis that was never drawn. Rotate and Scale are labels today.

### 3.6 Test Placement
Tests live in each file's bottom `mod tests`, and pure-module tests never open a window or a GPU context. Put new *logic* in a pure module and test it there. Put new *wiring* in `renderer.rs` and test it the way `hotkey_for` and `drag_action_label` already are: as a free pure function the seam calls, asserted directly.

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
    ├── mesh.rs         # HalfEdgeMesh: read-only topology queries, Newell normals, generate_render_buffers() - no topology-edit ops yet
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
- **Half-edge record shape:** `HalfEdge { origin, twin, next, prev, face }` - the opposite half-edge is `twin`, not `opposite`, and `twin`, `next`, `prev` and `face` are all `Option` (a boundary edge has no twin). `MeshVertex { position, half_edge }` stores one outgoing half-edge, also `Option`. There is no destination field; use `half_edge_destination()`.
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
Write what the app *does now*, not how it got here. The per-iteration narrative lives in `git log`; do not re-grow a build history here - it is the first thing to go stale and the last thing a model needs.
- **Working (run-tested):**
  - Flat-shaded cube from the half-edge mesh with Newell normals.
  - Maya camera navigation: Alt+LMB orbit, Alt+MMB pan (consistent screen-space tracking), Alt+RMB / scroll zoom.
  - Marking pie menu: hold RMB (without Alt) or Spacebar, flick toward a slot and release to switch `EditMode` (Edge=Up, Face=Down, Vertex=Left, Object=Right).
  - Viewport click-to-select: plain LMB click casts NDC ray; Shift+click toggles selection; empty click clears.
  - Visual overlays: selected faces tinted Maya warm orange, selected object tinted Maya green; wireframe edges with depth bias (cyan for selected); vertex square dots in Vertex mode (blue for unselected, bright yellow for selected).
  - Tool system: Q (Select), W (Move), E (Rotate), R (Scale) hotkeys and "PolyCraft Tools" egui panel.
  - Transform Gizmo (W key Move tool): constant-pixel-size 3D coordinate axes rendered with `Always` depth-pass at selection pivot. Hovering an axis highlights it in bright Maya yellow. Clicking and dragging an axis with LMB translates selected components (vertices, edges, faces, or object) live along the 3D axis with real-time normal and wireframe recalculation. Releasing LMB commits translation; pressing Escape cancels and restores initial positions.
  - Undo / Redo: `Ctrl+Z` steps back, `Ctrl+Y` and `Ctrl+Shift+Z` step forward. A whole gizmo drag is a single step; a press-and-release that moved nothing records no step, so `Ctrl+Z` after a stray click lands on the previous real action. Undo restores geometry and selection but never the `EditMode` or `Tool`. The Undo / Redo buttons in the "PolyCraft Tools" panel name the action each would perform and grey out when the history runs out.
- **Implemented and unit-tested:** 121 unit tests pass across camera, mesh, picking, scene, selection, tool, ui, overlay, gizmo, history, and renderer (keymap), and all of them run without a window or GPU. Covered alongside the pure picking math is `picking::Ray::from_screen` (top-left pixel origin to NDC ray, used by `State::cursor_ray`). The gizmo math is pure and tested: `GizmoAxis`, `pivot()`, `axis_world_length()`, `translate_arms()` (active yellow `COLOR_AXIS_ACTIVE`), `pick_gizmo_axis()` via `picking::ray_segment`, `project_ray_onto_axis()`, and `selected_vertices()`. `renderer.rs` seams are tested through the free functions they call - `hotkey_for(key, ctrl, shift) -> Option<Hotkey>` and `drag_action_label(mode, tool)` (yields "Move Vertex" / "Move Edge" / "Move Face" / "Move") - asserted directly with no window.
  - `HistoryStack<T>` is pure and generic, so stack semantics (ordering, bounded depth, redo dropped on a new action, no-op actions dropped) are tested with `T = i32`. `Scene` holds a `HistoryStack<DocumentSnapshot>` of mesh + selection only; `EditMode` and `Tool` are deliberately never versioned. `restore()` re-clips the restored selection through `retain_mode` plus live vertex/face id bounds, so undo can never resurrect a component kind the current mode cannot hold. `HalfEdgeMesh` and `Selection` derive `Clone + PartialEq` (plus `Debug`/`Eq` where meaningful) to support snapshotting and no-op detection.
- **Not built yet:** Rotate and Scale are labels only - the gizmo and drag path implement `Tool::Translate` alone. Edit operations (extrude, bevel, cut), FBX export.
  - **There is no topology-surgery layer to build on yet.** `mesh.rs` exposes only read-only queries (`face_half_edges`, `face_vertices`, `face_normal`, `face_center`, `edge_faces`, `edges`, `half_edge_destination`) plus the `from_faces`/`create_cube` constructors - no insert, split, flip or delete primitive exists. The first edit operation must create that layer, and `Scene::mesh_mut()` is `#[allow(dead_code)]` precisely because it is still waiting for that first caller.
- **Known limitations:** fan triangulation assumes convex faces; the graphics backend is whatever `Instance::default()` picks (D3D12 on this machine, not pinned in code); the single `dirty` flag re-uploads mesh *and* overlay buffers together, so split them before driving very high-frequency drags; `HISTORY_DEPTH` (64) bounds undo by snapshot memory, which is a few hundred KB per step at current model sizes. `cargo check`, `cargo test` (121 passing) and `cargo clippy --all-targets` (0 warnings) all pass as of this update.
- **Next Milestone - Polygon Modeling Operations:** face extrude, bevel, loop split / edge cut.
  - Build each operation as a single `Scene` method that returns the new topology, wrapped by the caller in `begin_action("Extrude Face")` / `end_action()` so it lands in history as one step for free.
  - Topology edits invalidate component ids, so every operation must decide what the selection becomes afterwards (Maya keeps the newly created faces) - and `Scene::restore` already bounds-checks a restored selection against live ids.
  - Index generation no longer caps model size: `RenderMesh::indices` is `Vec<u32>` and the render pass binds `IndexFormat::Uint32`, so extrusion can push past 65,536 un-welded vertices.
  - Followed by FBX export.

## 8. Handoff Checklist
1. `cargo check`, `cargo test` and `cargo clippy --all-targets` pass (no new warnings), and the app launches.
2. Update Section 4 (file structure) and Section 7 (status).
3. Commit with a descriptive message.

Roles: All AI agents and human contributors adhere strictly to this digest for architecture, conventions, and status updates.
