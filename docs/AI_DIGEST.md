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

## 3. Architectural Rules
- **Context Management:** Code is heavily modularized. Only share the specific `src/*.rs` file being actively worked on.
- **State Management:** Strict decoupling. The `State` struct encapsulates all `wgpu` initialization. The `winit` event loop strictly routes OS events.
- **Render Passes:** 3D Scene renders first with Z-buffer depth. `egui` renders second with `LoadOp::Load` to draw on top.

## 4. File Structure (Current)
```text
polycraft/
├── Cargo.toml      
├── docs/
│   └── AI_DIGEST.md
└── src/
    ├── main.rs     # winit loop, wgpu State, CameraController, and UI hook
    ├── mesh.rs     # Half-Edge Mesh data structure and GPU buffer triangulation
    └── shader.wgsl # Vertex/fragment shader