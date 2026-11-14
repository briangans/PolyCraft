//! Component picking: turn a screen position into a world-space ray, then find what that ray hits.
//!
//! Faces are picked with Möller–Trumbore. Vertices and edges are picked by proximity to the ray,
//! measured in screen-space pixels so the tolerance feels the same at any zoom, and anything
//! behind a nearer face is treated as occluded (Maya does not let you click through the surface).

use crate::camera::{Camera, FOV_Y_DEGREES};
use crate::mesh::{EdgeId, FaceId, HalfEdgeMesh, VertexId};
use cgmath::{InnerSpace, Point3, Vector3};

/// How far behind a hit surface another component may sit before it counts as occluded.
/// Relative, so it scales with distance from the camera.
const OCCLUSION_TOLERANCE: f32 = 0.01;

/// A world-space ray with a normalized direction.
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Point3<f32>,
    pub direction: Vector3<f32>,
}

impl Ray {
    pub fn new(origin: Point3<f32>, direction: Vector3<f32>) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
        }
    }

    /// The ray through a point in normalized device coordinates: (-1, -1) bottom-left, (1, 1)
    /// top-right, matching what the vertex shader sees before the viewport transform.
    pub fn through_ndc(camera: &Camera, ndc_x: f32, ndc_y: f32, aspect: f32) -> Self {
        let tan_half_fov = (FOV_Y_DEGREES.to_radians() * 0.5).tan();
        let direction = camera.forward()
            + camera.right() * (ndc_x * aspect * tan_half_fov)
            + camera.up() * (ndc_y * tan_half_fov);
        Self::new(camera.eye(), direction)
    }

    pub fn point_at(&self, distance: f32) -> Point3<f32> {
        self.origin + self.direction * distance
    }

    /// Distance from the ray to a point, plus how far along the ray that closest approach happens.
    /// `None` when the point is behind the ray origin.
    fn closest_approach(&self, point: Point3<f32>) -> Option<(f32, f32)> {
        let to_point = point - self.origin;
        let along = to_point.dot(self.direction);
        if along <= 0.0 {
            return None;
        }
        let offset = to_point - self.direction * along;
        Some((offset.magnitude(), along))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceHit {
    pub face: FaceId,
    pub distance: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VertexHit {
    pub vertex: VertexId,
    pub distance: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeHit {
    pub edge: EdgeId,
    pub distance: f32,
}

/// World-space size of one pixel at `distance` from the camera. Also used by the transform
/// gizmo, which scales a constant-on-screen handle by it, so both pick and draw agree on what a
/// "pixel" means. `pub(crate)` rather than duplicated so the projection math has one home.
pub(crate) fn world_per_pixel(distance: f32, viewport_height: f32) -> f32 {
    let tan_half_fov = (FOV_Y_DEGREES.to_radians() * 0.5).tan();
    (2.0 * tan_half_fov * distance) / viewport_height.max(1.0)
}

/// Distance to the nearest front-facing triangle of a polygon, or `None` if the ray misses.
fn ray_triangle(ray: &Ray, a: Point3<f32>, b: Point3<f32>, c: Point3<f32>) -> Option<f32> {
    let edge1 = b - a;
    let edge2 = c - a;
    let p = ray.direction.cross(edge2);
    let determinant = edge1.dot(p);

    // Same winding as the renderer's `FrontFace::Ccw`, so back faces are skipped here too.
    if determinant < 1e-8 {
        return None;
    }

    let inverse_determinant = 1.0 / determinant;
    let to_origin = ray.origin - a;
    let u = to_origin.dot(p) * inverse_determinant;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let q = to_origin.cross(edge1);
    let v = ray.direction.dot(q) * inverse_determinant;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let distance = edge2.dot(q) * inverse_determinant;
    if distance <= 1e-6 {
        return None;
    }

    Some(distance)
}

/// Nearest face the ray hits. Faces are triangulated the same way the renderer fans them.
pub fn pick_face(mesh: &HalfEdgeMesh, ray: &Ray) -> Option<FaceHit> {
    let mut nearest: Option<FaceHit> = None;

    for face_index in 0..mesh.faces.len() {
        let face = FaceId(face_index);
        let positions = mesh.face_positions(face);
        if positions.len() < 3 {
            continue;
        }

        for i in 1..positions.len() - 1 {
            let Some(distance) = ray_triangle(ray, positions[0], positions[i], positions[i + 1])
            else {
                continue;
            };
            let closer = nearest.map(|hit| distance < hit.distance).unwrap_or(true);
            if closer {
                nearest = Some(FaceHit { face, distance });
            }
        }
    }

    nearest
}

/// True when the ray hits any face: used for object-mode selection.
pub fn hits_object(mesh: &HalfEdgeMesh, ray: &Ray) -> bool {
    pick_face(mesh, ray).is_some()
}

fn is_occluded(component_distance: f32, surface: Option<FaceHit>) -> bool {
    match surface {
        Some(hit) => component_distance > hit.distance + hit.distance * OCCLUSION_TOLERANCE,
        None => false,
    }
}

/// Nearest vertex within `radius_pixels` of the ray, ignoring anything behind the surface.
pub fn pick_vertex(
    mesh: &HalfEdgeMesh,
    ray: &Ray,
    radius_pixels: f32,
    viewport_height: f32,
) -> Option<VertexHit> {
    let surface = pick_face(mesh, ray);
    let mut best: Option<(f32, VertexHit)> = None;

    for index in 0..mesh.vertices.len() {
        let vertex = VertexId(index);
        let Some((offset, distance)) = ray.closest_approach(mesh.vertex_position(vertex)) else {
            continue;
        };

        let radius = world_per_pixel(distance, viewport_height) * radius_pixels;
        if offset > radius || is_occluded(distance, surface) {
            continue;
        }

        let candidate = VertexHit { vertex, distance };
        let better = best
            .map(|(best_offset, _)| offset < best_offset)
            .unwrap_or(true);
        if better {
            best = Some((offset, candidate));
        }
    }

    best.map(|(_, hit)| hit)
}

/// Closest approach between a ray and a line segment: `(distance, distance along the ray)`.
fn ray_segment(ray: &Ray, a: Point3<f32>, b: Point3<f32>) -> Option<(f32, f32)> {
    let direction = ray.direction;
    let segment = b - a;
    let to_origin = ray.origin - a;

    let direction_sq = direction.dot(direction);
    let direction_segment = direction.dot(segment);
    let segment_sq = segment.dot(segment);
    let direction_to_origin = direction.dot(to_origin);
    let segment_to_origin = segment.dot(to_origin);

    // Zeros the denominator term below; clamping `s` afterwards keeps the point on the segment.
    let denominator = direction_segment * direction_segment - direction_sq * segment_sq;
    let mut along_segment = if denominator.abs() < 1e-8 {
        0.0
    } else {
        (direction_segment * direction_to_origin - direction_sq * segment_to_origin) / denominator
    };
    along_segment = along_segment.clamp(0.0, 1.0);

    let point_on_segment = a + segment * along_segment;
    let along_ray = (point_on_segment - ray.origin).dot(direction);
    if along_ray <= 0.0 {
        return None;
    }

    let point_on_ray = ray.origin + direction * along_ray;
    Some(((point_on_segment - point_on_ray).magnitude(), along_ray))
}

/// Nearest edge within `radius_pixels` of the ray, ignoring anything behind the surface.
pub fn pick_edge(
    mesh: &HalfEdgeMesh,
    ray: &Ray,
    radius_pixels: f32,
    viewport_height: f32,
) -> Option<EdgeHit> {
    let surface = pick_face(mesh, ray);
    let mut best: Option<(f32, EdgeHit)> = None;

    for (edge, half_edge) in mesh.edges() {
        let start = mesh.vertex_position(mesh.half_edges[half_edge.0].origin);
        let end = mesh.vertex_position(mesh.half_edge_destination(half_edge));
        let Some((offset, distance)) = ray_segment(ray, start, end) else {
            continue;
        };

        let radius = world_per_pixel(distance, viewport_height) * radius_pixels;
        if offset > radius || is_occluded(distance, surface) {
            continue;
        }

        let candidate = EdgeHit { edge, distance };
        let better = best
            .map(|(best_offset, _)| offset < best_offset)
            .unwrap_or(true);
        if better {
            best = Some((offset, candidate));
        }
    }

    best.map(|(_, hit)| hit)
}
#[cfg(test)]
mod tests {
    use super::*;

    const VIEWPORT_HEIGHT: f32 = 600.0;

    /// A camera at (0, 0, 3) looking down -Z at the unit cube, with no tilt.
    fn front_camera() -> Camera {
        Camera {
            target: Point3::new(0.0, 0.0, 0.0),
            distance: 3.0,
            yaw: 0.0,
            pitch: 0.0,
        }
    }

    fn cube() -> HalfEdgeMesh {
        HalfEdgeMesh::create_cube()
    }

    /// A ray aimed straight at a world point from the front camera's eye.
    fn ray_at(point: Point3<f32>) -> Ray {
        Ray::new(front_camera().eye(), point - front_camera().eye())
    }

    /// Projects a world point with the camera's own matrix, so the ray and the renderer are
    /// checked against each other instead of against a hand-written expectation.
    fn project(camera: &Camera, point: Point3<f32>) -> (f32, f32) {
        let aspect = 1.0;
        let clip = camera.build_view_projection_matrix(aspect) * point.to_homogeneous();
        (clip.x / clip.w, clip.y / clip.w)
    }

    #[test]
    fn ndc_ray_passes_through_the_point_it_was_projected_from() {
        let camera = front_camera();
        for point in [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.5, 0.5, 0.5),
            Point3::new(-0.5, 0.25, -0.5),
        ] {
            let (ndc_x, ndc_y) = project(&camera, point);
            let ray = Ray::through_ndc(&camera, ndc_x, ndc_y, 1.0);
            let to_point = (point - camera.eye()).normalize();
            let cosine = ray.direction.dot(to_point);
            assert!(
                cosine > 0.9999,
                "ray missed its own projection: cos = {cosine}"
            );
        }
    }

    #[test]
    fn centre_ray_looks_at_the_target() {
        let camera = front_camera();
        let ray = Ray::through_ndc(&camera, 0.0, 0.0, 1.0);
        assert!((ray.direction - camera.forward()).magnitude() < 1e-5);
        assert!((ray.origin - camera.eye()).magnitude() < 1e-5);
    }

    #[test]
    fn ray_origin_is_the_eye_and_direction_is_normalized() {
        let ray = Ray::new(Point3::new(1.0, 2.0, 3.0), Vector3::new(0.0, 0.0, -5.0));
        assert_eq!(ray.origin, Point3::new(1.0, 2.0, 3.0));
        assert!((ray.direction.magnitude() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn picking_the_front_face_reports_the_front_of_the_cube() {
        let mesh = cube();
        let hit = pick_face(&mesh, &ray_at(Point3::new(0.0, 0.0, 0.0))).expect("front face");
        // Eye at z = 3, front face at z = 0.5.
        assert!(
            (hit.distance - 2.5).abs() < 1e-4,
            "distance {}",
            hit.distance
        );
        assert_eq!(hit.face, FaceId(0), "front face of the cube");
    }

    #[test]
    fn picking_from_behind_hits_the_back_face() {
        let camera = Camera {
            target: Point3::new(0.0, 0.0, 0.0),
            distance: 3.0,
            yaw: std::f32::consts::PI,
            pitch: 0.0,
        };
        let ray = Ray::new(camera.eye(), camera.forward());
        let hit = pick_face(&cube(), &ray).expect("back face");
        assert_eq!(hit.face, FaceId(2), "back face of the cube");
    }

    #[test]
    fn a_ray_missing_the_cube_hits_nothing() {
        let mesh = cube();
        let ray = ray_at(Point3::new(5.0, 5.0, 0.0));
        assert!(pick_face(&mesh, &ray).is_none());
        assert!(!hits_object(&mesh, &ray));
    }

    #[test]
    fn picking_a_vertex_at_its_own_pixel_finds_it() {
        let mesh = cube();
        let vertex = Point3::new(-0.5, -0.5, 0.5);
        let hit = pick_vertex(&mesh, &ray_at(vertex), 5.0, VIEWPORT_HEIGHT).expect("front vertex");
        assert_eq!(mesh.vertex_position(hit.vertex), vertex);
    }

    #[test]
    fn a_back_vertex_is_not_pickable_through_the_surface() {
        let mesh = cube();
        let behind = Point3::new(-0.5, -0.5, -0.5);
        assert!(
            pick_vertex(&mesh, &ray_at(behind), 5.0, VIEWPORT_HEIGHT).is_none(),
            "the front face blocks it"
        );
    }

    #[test]
    fn vertex_tolerance_is_measured_in_pixels() {
        let mesh = cube();
        let vertex = Point3::new(-0.5, -0.5, 0.5);
        // Aim slightly to the side of the vertex.
        let nudged = vertex + Vector3::new(0.05, 0.0, 0.0);
        let ray = ray_at(nudged);

        assert!(
            pick_vertex(&mesh, &ray, 5.0, VIEWPORT_HEIGHT).is_none(),
            "5 px must not reach a vertex 20+ px away"
        );
        let hit = pick_vertex(&mesh, &ray, 60.0, VIEWPORT_HEIGHT).expect("wide radius reaches it");
        assert_eq!(mesh.vertex_position(hit.vertex), vertex);
    }

    #[test]
    fn world_per_pixel_grows_with_distance() {
        let near = world_per_pixel(1.0, VIEWPORT_HEIGHT);
        let far = world_per_pixel(10.0, VIEWPORT_HEIGHT);
        assert!(far > near * 9.0 && far < near * 11.0);
    }

    #[test]
    fn picking_the_middle_of_a_front_edge_finds_that_edge() {
        let mesh = cube();
        // Midpoint of the front-bottom edge, from vertex 0 to vertex 1.
        let midpoint = Point3::new(0.0, -0.5, 0.5);
        let hit = pick_edge(&mesh, &ray_at(midpoint), 5.0, VIEWPORT_HEIGHT).expect("front edge");
        assert_eq!(hit.edge, EdgeId(0, 1));
    }

    #[test]
    fn a_back_edge_is_not_pickable_through_the_surface() {
        let mesh = cube();
        // Midpoint of the back-bottom edge, from vertex 4 to vertex 5.
        let midpoint = Point3::new(0.0, -0.5, -0.5);
        assert!(pick_edge(&mesh, &ray_at(midpoint), 5.0, VIEWPORT_HEIGHT).is_none());
    }

    #[test]
    fn ray_segment_distance_matches_hand_computed_values() {
        let ray = Ray::new(Point3::new(0.0, 0.0, 5.0), Vector3::new(0.0, 0.0, -1.0));

        // Straight through the middle of the segment.
        let (distance, along) =
            ray_segment(&ray, Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)).unwrap();
        assert!(distance.abs() < 1e-5, "distance {distance}");
        assert!((along - 5.0).abs() < 1e-5);

        // Aimed past the segment's end: clamped to the nearer endpoint, 2 units aside.
        let (distance, _) =
            ray_segment(&ray, Point3::new(2.0, 0.0, 0.0), Point3::new(3.0, 0.0, 0.0)).unwrap();
        assert!((distance - 2.0).abs() < 1e-4, "distance {distance}");
    }

    #[test]
    fn a_segment_behind_the_ray_is_ignored() {
        let ray = Ray::new(Point3::new(0.0, 0.0, 5.0), Vector3::new(0.0, 0.0, -1.0));
        assert!(
            ray_segment(&ray, Point3::new(0.0, 0.0, 8.0), Point3::new(1.0, 0.0, 8.0)).is_none()
        );
    }
}
