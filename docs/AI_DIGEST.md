# PolyCraft - AI Context Digest

## 1. Project Vision
PolyCraft is a trimmed-down Maya clone written in Rust, built to leverage decades of learned muscle memory (e.g., V-key vertex snapping, D-key pivot editing).
- **Core Features:** Polygon modeling, FBX export, pie menus.
- **Explicit Non-Goals:** Advanced rendering (mental ray/Vray), physics, NURBS, Output Window, complex construction history. We lean hard on the undo buffer instead.

## 2. Tech Stack
- **Language:** Rust (Windows / MSVC ABI)
- **Graphics:** `wgpu` (DirectX 12 backend), `winit`, `pollster`, `bytemuck`, `cgmath`
- **UI:** `egui` (Immediate mode, perfect for Pie Menus)
- **Geometry (Planned):** Custom Half-Edge mesh structure.
- **Export (Planned):** `fbxcel` or C++ bindings to Autodesk FBX SDK.

## 3. Architectural Rules
- **Viewport:** `wgpu` clears to Maya-gray `(r: 0.196, g: 0.196, b: 0.196)`.
- **State Management:** Strict decoupling. The `State` struct encapsulates all `wgpu` initialization, buffers, and render passes. The `winit` event loop strictly routes OS events (resizes, mouse inputs) to the `State` and `CameraController`.
- **Undo System:** Command pattern. Every mesh operation is a struct implementing `execute()` and `undo()`.

## 4. File Structure (Current)
```text
polycraft/
├── Cargo.toml      # [Dependencies: winit, wgpu, pollster, bytemuck, cgmath]
├── docs/
│   └── AI_DIGEST.md
└── src/
    ├── main.rs     # winit event loop, wgpu State struct, Camera, and CameraController
    └── shader.wgsl # Vertex/fragment shader with Camera projection math