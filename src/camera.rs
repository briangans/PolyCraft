use cgmath::InnerSpace;
use winit::event::{ElementState, MouseButton, MouseScrollDelta};

pub struct Camera {
    pub target: cgmath::Point3<f32>,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl Camera {
    pub fn build_view_projection_matrix(&self, aspect: f32) -> cgmath::Matrix4<f32> {
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();

        let eye = cgmath::Point3::new(
            self.target.x + self.distance * cos_pitch * sin_yaw,
            self.target.y + self.distance * sin_pitch,
            self.target.z + self.distance * cos_pitch * cos_yaw,
        );

        let view = cgmath::Matrix4::look_at_rh(eye, self.target, cgmath::Vector3::unit_y());
        let proj = cgmath::perspective(cgmath::Deg(45.0), aspect, 0.1, 100.0);
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
            let (sin_pitch, cos_pitch) = camera.pitch.sin_cos();
            let (sin_yaw, cos_yaw) = camera.yaw.sin_cos();

            let forward = cgmath::Vector3::new(cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw)
                .normalize();
            let right = forward.cross(cgmath::Vector3::unit_y()).normalize();
            let up = right.cross(forward).normalize();

            camera.target += right * (dx as f32 * self.pan_speed * camera.distance)
                + up * (dy as f32 * self.pan_speed * camera.distance);
        } else if self.is_right_clicked {
            camera.distance += (dx as f32 + dy as f32) * self.zoom_speed * camera.distance;
            camera.distance = camera.distance.max(0.1);
        }
    }
}
