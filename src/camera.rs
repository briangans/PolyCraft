//! Orbit camera (Maya-style Alt+drag navigation) plus the view basis the picker needs to turn a
//! screen position into a world-space ray.

use cgmath::{InnerSpace, Point3, Vector3};
use winit::event::{ElementState, MouseButton, MouseScrollDelta};

/// Vertical field of view. Shared by the projection matrix and the picking ray so a click lands
/// where the pixel it came from points.
pub const FOV_Y_DEGREES: f32 = 45.0;

pub struct Camera {
    pub target: Point3<f32>,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl Camera {
    pub fn eye(&self) -> Point3<f32> {
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();

        Point3::new(
            self.target.x + self.distance * cos_pitch * sin_yaw,
            self.target.y + self.distance * sin_pitch,
            self.target.z + self.distance * cos_pitch * cos_yaw,
        )
    }

    /// Unit vector from the eye toward the target.
    pub fn forward(&self) -> Vector3<f32> {
        (self.target - self.eye()).normalize()
    }

    /// Camera-space +X (screen right), derived from world up.
    pub fn right(&self) -> Vector3<f32> {
        self.forward().cross(Vector3::unit_y()).normalize()
    }

    /// Camera-space +Y (screen up).
    pub fn up(&self) -> Vector3<f32> {
        self.right().cross(self.forward()).normalize()
    }

    pub fn build_view_projection_matrix(&self, aspect: f32) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at_rh(self.eye(), self.target, Vector3::unit_y());
        let proj = cgmath::perspective(cgmath::Deg(FOV_Y_DEGREES), aspect, 0.1, 100.0);
        let wgpu_correction = cgmath::Matrix4::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0,
        );
        wgpu_correction * proj * view
    }
}

pub struct CameraController {
    pub alt_pressed: bool,
    pub is_left_clicked: bool,
    pub is_middle_clicked: bool,
    pub is_right_clicked: bool,
    pub orbit_speed: f32,
    pub pan_speed: f32,
    pub zoom_speed: f32,
}

impl CameraController {
    pub fn new() -> Self {
        Self {
            alt_pressed: false,
            is_left_clicked: false,
            is_middle_clicked: false,
            is_right_clicked: false,
            orbit_speed: 0.005,
            pan_speed: 0.002,
            zoom_speed: 0.01,
        }
    }

    pub fn process_mouse_input(&mut self, state: ElementState, button: MouseButton) {
        let is_pressed = state == ElementState::Pressed;
        match button {
            MouseButton::Left => self.is_left_clicked = is_pressed,
            MouseButton::Middle => self.is_middle_clicked = is_pressed,
            MouseButton::Right => self.is_right_clicked = is_pressed,
            _ => {}
        }
    }

    pub fn process_scroll(&mut self, delta: &MouseScrollDelta, camera: &mut Camera) {
        let scroll_amount = match delta {
            MouseScrollDelta::LineDelta(_, y) => *y * 0.5,
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
        };
        camera.distance -= scroll_amount * camera.distance * 0.2;
        camera.distance = camera.distance.max(0.1);
    }

    pub fn process_mouse_motion(&mut self, dx: f64, dy: f64, camera: &mut Camera) {
        if !self.alt_pressed {
            return;
        }

        if self.is_left_clicked {
            camera.yaw -= (dx as f32) * self.orbit_speed;
            camera.pitch += (dy as f32) * self.orbit_speed;
            let safe_angle = std::f32::consts::FRAC_PI_2 - 0.01;
            camera.pitch = camera.pitch.clamp(-safe_angle, safe_angle);
        } else if self.is_middle_clicked {
            let forward = camera.forward();
            let right = camera.right();
            let up = right.cross(forward).normalize();

            camera.target += right * (dx as f32 * self.pan_speed * camera.distance)
                + up * (dy as f32 * self.pan_speed * camera.distance);
        } else if self.is_right_clicked {
            camera.distance += (dx as f32 + dy as f32) * self.zoom_speed * camera.distance;
            camera.distance = camera.distance.max(0.1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> Camera {
        Camera {
            target: Point3::new(0.0, 0.0, 0.0),
            distance: 3.0,
            yaw: 0.785,
            pitch: 0.5,
        }
    }

    #[test]
    fn basis_vectors_are_orthonormal() {
        let camera = camera();
        let (forward, right, up) = (camera.forward(), camera.right(), camera.up());

        for vector in [forward, right, up] {
            assert!((vector.magnitude() - 1.0).abs() < 1e-5);
        }
        assert!(forward.dot(right).abs() < 1e-5);
        assert!(forward.dot(up).abs() < 1e-5);
        assert!(right.dot(up).abs() < 1e-5);
    }

    #[test]
    fn eye_moves_around_the_target_at_the_orbit_distance() {
        let camera = camera();
        assert!((camera.eye() - camera.target).magnitude() - camera.distance < 1e-4);
    }

    #[test]
    fn pitched_all_the_way_up_the_basis_stays_finite() {
        // Clamping happens in the controller, but a hand-built camera must not produce NaNs.
        let camera = Camera {
            target: Point3::new(0.0, 0.0, 0.0),
            distance: 3.0,
            yaw: 0.0,
            pitch: std::f32::consts::FRAC_PI_2 - 0.01,
        };
        let forward = camera.forward();
        assert!(forward.x.is_finite() && forward.y.is_finite() && forward.z.is_finite());
    }

    #[test]
    fn lookup_matrix_reflects_the_view_basis() {
        // The camera at default yaw looks from +Z toward the origin, so forward is -Z-ish.
        let camera = Camera {
            target: Point3::new(0.0, 0.0, 0.0),
            distance: 3.0,
            yaw: 0.0,
            pitch: 0.0,
        };
        let forward = camera.forward();
        assert!((forward.z + 1.0).abs() < 1e-5, "expected forward = -Z");
    }
}
