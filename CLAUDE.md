# CLAUDE.md

Guidance for Claude when working in this repository.

## Project

PolyCraft — a Rust desktop 3D modeling app, deliberately modeled on Maya's interaction
conventions (polygon modeling, pie menus, FBX export) but much smaller in scope.

Read `docs/AI_DIGEST.md` first: it has the architecture rules, conventions, current status and
the proposed next steps. Then read `docs/ARCHITECTURE.md` before starting feature work; it lists
what is in scope and, importantly, what is deliberately out of scope (advanced renderers,
physics, NURBS, construction history beyond the current operation).

When you finish a task, update Section 4 (file structure) and Section 6 (status) of the digest.

## Layout

| File | Responsibility |
| --- | --- |
| `src/main.rs` | Entry point, winit event loop, wgpu surface and `State`, the GPU `Vertex` layout, mesh buffer upload; declares the modules |
| `src/mesh.rs` | Half-edge mesh (`HalfEdgeMesh`), component ids, topology queries, triangulation into `RenderMesh` |
| `src/camera.rs` | Orbit camera, Maya-style navigation controls, the view basis picking uses |
| `src/scene.rs` | The editable document: mesh, `EditMode`, `Selection`, dirty flag; `select_at` |
| `src/selection.rs` | Which vertices, edges, faces and object are selected, with Maya click semantics |
| `src/picking.rs` | Screen position to world ray; face, vertex, edge and object hit tests |
| `src/edit_mode.rs` | The `EditMode` enum (Object, Vertex, Edge, Face) |
| `src/ui.rs` | egui: the Space pie menu and the Tools window; returns user intent instead of mutating state |
| `src/shader.wgsl` | WGSL vertex/fragment shaders |

## Build and verify

```
cargo check                  # fast feedback while editing
cargo build                  # full build
cargo test                   # unit tests live in `mod tests` in camera, mesh, picking, scene, selection and ui
cargo fmt                    # format before finishing
cargo clippy --all-targets   # no new warnings (`-D warnings` fails today: the selection API is not wired up yet)
cargo run                    # launch the app (needs a GPU/display)
```

Always run `cargo check` (and `cargo test`/`cargo fmt`/`cargo clippy` where the change warrants it)
after editing Rust code. Do not report a change as done if it does not compile.

## Conventions

- Rust 2021 edition. Dependencies are pinned in `Cargo.toml`: `winit` 0.29, `wgpu` 0.20,
  `egui`/`egui-wgpu`/`egui-winit` 0.28, `cgmath` 0.18, `bytemuck` 1.x. Do not add a crate
  without checking whether the existing stack already covers the need — versions here are
  coupled (egui 0.28 pairs with wgpu 0.20), so upgrading one usually means upgrading all.
- The GPU `Vertex` struct in `src/main.rs` derives `bytemuck::Pod`/`Zeroable` and is described by
  a matching `wgpu::VertexBufferLayout`; keep the two in sync when adding attributes.
- Shader edits in `src/shader.wgsl` must stay in step with the `@location(n)` indices declared
  in `Vertex::ATTRIBS` in `src/main.rs`.
- Change the document (mesh, mode, selection) only through `Scene`; each change flags it dirty so
  the renderer re-uploads the GPU buffers.
- `src/ui.rs` never mutates application state: it returns what the user asked for and `main.rs`
  applies it.
- Keep logic that needs neither the GPU nor a window (topology, picking, selection, camera
  math) in pure functions with unit tests next to them, and add tests with new behavior.
- Recoverable errors should be reported, not panicked away; the app should not crash on a
  missing GPU or a surface resize.
- Prefer adding a new module over growing `main.rs`; keep files focused.

## Git

Commit messages should describe the user-visible change. Never force-push or rewrite
history on this repository; ask before running anything that discards work. `git push` is
denied in `.claude/settings.json`, so leave pushing to the user.
