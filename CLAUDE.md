# CLAUDE.md

Guidance for Claude when working in this repository.

## Project

PolyCraft — a Rust desktop 3D modeling app, deliberately modeled on Maya's interaction
conventions (polygon modeling, pie menus, FBX export) but much smaller in scope.

Read `docs/ARCHITECTURE.md` before starting feature work; it lists what is in scope and,
importantly, what is deliberately out of scope (advanced renderers, physics, NURBS,
construction history beyond the current operation).

## Layout

| File | Responsibility |
| --- | --- |
| `src/main.rs` | Entry point, winit event loop, wgpu surface setup, module wiring (`mod camera; mod mesh; mod scene; mod ui;`) |
| `src/camera.rs` | View/projection matrices, orbit/pan/zoom camera state |
| `src/mesh.rs` | Vertex/index buffers and GPU mesh upload |
| `src/scene.rs` | Scene graph / object storage |
| `src/ui.rs` | egui panels; contains the crate's unit tests in `mod tests` |
| `src/shader.wgsl` | WGSL vertex/fragment shaders |

## Build and verify

```
cargo check                 # fast feedback while editing
cargo build                 # full build
cargo test                  # unit tests (currently in src/ui.rs)
cargo fmt                   # format before finishing
cargo clippy -- -D warnings # lints
cargo run                   # launch the app (needs a GPU/display)
```

Always run `cargo check` (and `cargo fmt`/`cargo clippy` where the change warrants it)
after editing Rust code. Do not report a change as done if it does not compile.

## Conventions

- Rust 2021 edition. Dependencies are pinned in `Cargo.toml`: `winit` 0.29, `wgpu` 0.20,
  `egui`/`egui-wgpu`/`egui-winit` 0.28, `cgmath` 0.18, `bytemuck` 1.x. Do not add a crate
  without checking whether the existing stack already covers the need — versions here are
  coupled (egui 0.28 pairs with wgpu 0.20), so upgrading one usually means upgrading all.
- Vertex structs uploaded to the GPU derive `bytemuck::Pod`/`Zeroable` and are described by
  a matching `wgpu::VertexBufferLayout`; keep the two in sync when adding attributes.
- Shader edits in `src/shader.wgsl` must stay in step with the `@location(n)` indices
  declared in `src/mesh.rs`.
- Recoverable errors should be reported, not panicked away; the app should not crash on a
  missing GPU or a surface resize.
- Prefer adding a new module over growing `main.rs`; keep files focused.

## Git

Commit messages should describe the user-visible change. Never force-push or rewrite
history on this repository; ask before running anything that discards work.
