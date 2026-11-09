struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    out.color = model.color;
    out.normal = model.normal; 
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // A fixed directional light shining down and to the left
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.8));
    
    // Lambertian diffuse lighting (dot product of surface normal and light vector)
    let diff = max(dot(in.normal, light_dir), 0.0);
    
    // Combine base ambient light with the directional diffuse
    let ambient = 0.2;
    let lighting = ambient + diff * 0.8;
    
    let final_color = in.color * lighting;
    
    return vec4<f32>(final_color, 1.0);
}

@vertex
fn vs_overlay(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    out.color = model.color;
    out.normal = model.normal;
    return out;
}

@fragment
fn fs_overlay(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
