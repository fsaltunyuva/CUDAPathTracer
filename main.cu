#include <iostream>
#include <curand_kernel.h>
#include <vector>
#include <chrono>

// Toggle Next Event Estimation
#define USE_NEE 1

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "vec3.cuh"
#include "ray.cuh"
#include "hittable.cuh"
#include "camera.cuh"

__device__ bool world_hit(Shape* world, int num_shapes, const ray& r, float t_min, float t_max, HitRecord& rec, const aabb& scene_box) {
    // Check bounding box first
    if (!scene_box.hit(r, t_min, t_max))
        return false;

    HitRecord temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (int i = 0; i < num_shapes; i++) {
        if (world[i].hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

__device__ color ray_color(ray r, const color& background, Shape* world, int num_shapes, int* area_light_indices,
    int num_area_lights, PointLight* point_lights, int num_point_lights, Material* materials, curandState *local_rand_state,
    int max_depth, const aabb& scene_box) {

    color final_color(0, 0, 0);
    color throughput(1, 1, 1);
    bool is_indirect = false;

    for (int depth = 0; depth < max_depth; depth++) {
        HitRecord rec;
        if (!world_hit(world, num_shapes, r, 0.001f, 10000.0f, rec, scene_box)) { // check each object (depending on shape's own formula)
            final_color += throughput * background;
            break;
        }

        Material mat = materials[rec.mat_index];

        color emitted(0,0,0);
        if (mat.type == MatType::LIGHT && rec.front_face) {
            emitted = mat.emit;
        }

        // Avoid double counting for area lights if this is an indirect diffuse bounce
#if USE_NEE
        if (is_indirect) emitted = color(0,0,0); 
#endif

        if (mat.type == MatType::LIGHT) { // if we hit a light source, add its emission and stop tracing
            final_color += throughput * emitted;
            break;
        }

        if (mat.type == MatType::METAL) { // if we hit a metal, reflect the ray with fuzz and continue tracing
            vec3 reflected = reflect(unit_vector(r.direction()), rec.normal);
            r = ray(rec.p, reflected + mat.fuzz * random_in_unit_sphere(local_rand_state));
            
            if (dot(r.direction(), rec.normal) <= 0.0f) break;
            
            throughput *= mat.albedo;
            final_color += throughput * emitted;
            is_indirect = false; 
            continue;
        }

        if (mat.type == MatType::LAMBERTIAN) { // if we hit a diffuse surface and if NEE is off, add emission immediately, otherwise add it after calculating direct lighting to avoid double counting
            final_color += throughput * emitted;

            color direct_light(0,0,0);

            // Point Lights
#if USE_NEE
            for (int i = 0; i < num_point_lights; i++) {
                vec3 light_dir = point_lights[i].pos - rec.p;
                float dist = light_dir.length();
                light_dir = unit_vector(light_dir);
                
                HitRecord shadow_rec;
                if (!world_hit(world, num_shapes, ray(rec.p, light_dir), 0.001f, dist - 0.001f, shadow_rec, scene_box)) {
                    float cosine = dot(rec.normal, light_dir);
                    float scattering_pdf = cosine < 0.0f ? 0.0f : cosine / PI;
                    // Inverse square falloff for point light
                    color li = point_lights[i].emit / (dist * dist);
                    direct_light += mat.albedo * scattering_pdf * li * PI; 
                }
            }

            // Area Lights
            for (int i = 0; i < num_area_lights; i++) {
                Shape& light_shape = world[area_light_indices[i]];
                vec3 light_dir = light_shape.random(rec.p, local_rand_state);
                float pdf_val = light_shape.pdf_value(rec.p, light_dir);

                if (pdf_val > 0.0f) {
                    ray shadow_ray(rec.p, unit_vector(light_dir));
                    HitRecord shadow_rec;

                    if (world_hit(world, num_shapes, shadow_ray, 0.001f, 10000.0f, shadow_rec, scene_box)) {
                        Material shadow_mat = materials[shadow_rec.mat_index];

                        if (shadow_mat.type == MatType::LIGHT && shadow_rec.front_face) {
                            float cosine = dot(rec.normal, shadow_ray.direction());

                            float scattering_pdf = cosine < 0.0f ? 0.0f : cosine / PI;
                            direct_light += (mat.albedo * scattering_pdf * shadow_mat.emit) / pdf_val;
                        }
                    }
                }
            }
#endif

            final_color += throughput * direct_light;

            vec3 direction = rec.normal + random_unit_vector(local_rand_state);

            if (direction.near_zero())
                direction = rec.normal;
            
            r = ray(rec.p, unit_vector(direction));
            
            throughput *= mat.albedo;
            is_indirect = true;
        }
    }

    return final_color;
}

__device__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y))
        return;

    int pixel_index = j * max_x + i;
    
    curand_init(1234 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render_init_kernel(int max_x, int max_y, curandState *rand_state) {
    render_init(max_x, max_y, rand_state);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int samples, int max_depth, camera *cam, Shape* world, int num_shapes, int* area_light_indices, int num_area_lights,
    PointLight* point_lights, int num_point_lights, Material* materials, curandState *rand_state,aabb* scene_box) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    
    curandState local_rand_state = rand_state[pixel_index];
    color pixel_color(0,0,0);
    color background(0,0,0);

    for (int s = 0; s < samples; s++) {
        // anti aliasing
        float u = float (i + curand_uniform(&local_rand_state)) / float (max_x - 1);
        float v = float (j + curand_uniform(&local_rand_state)) / float (max_y - 1);

        ray r = cam->get_ray(u, v);
        pixel_color += ray_color(r, background, world, num_shapes, area_light_indices, num_area_lights, point_lights, num_point_lights, materials, &local_rand_state, max_depth, *scene_box);
    }
    
    rand_state[pixel_index] = local_rand_state; // Update global state to avoid divergence in future calls

    // Average the samples
    pixel_color /= float (samples);

    // Gamma correction
    pixel_color.e[0] = sqrtf(pixel_color.e[0]);
    pixel_color.e[1] = sqrtf(pixel_color.e[1]);
    pixel_color.e[2] = sqrtf(pixel_color.e[2]);
    
    fb[pixel_index] = pixel_color;
}

// Same as in the CPU version, but I initialized the world on the GPU to avoid unnecessary data transfers
__global__ void create_world(Shape* world, Material* materials, PointLight* point_lights, int* area_light_indices, camera* cam, float aspect_ratio, aabb* scene_box) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        materials[0] = Material(MatType::LAMBERTIAN, color(0.65f, 0.05f, 0.05f), 0.0f, color(0,0,0)); 
        materials[1] = Material(MatType::LAMBERTIAN, color(0.73f, 0.73f, 0.73f), 0.0f, color(0,0,0)); 
        materials[2] = Material(MatType::LAMBERTIAN, color(0.12f, 0.45f, 0.15f), 0.0f, color(0,0,0)); 
        materials[3] = Material(MatType::LIGHT, color(0,0,0), 0.0f, color(15.0f, 15.0f, 15.0f)); 
        materials[4] = Material(MatType::METAL, color(0.8f, 0.85f, 0.88f), 0.0f, color(0,0,0)); 
        
        world[0].init_quad(point3(555,0,0), vec3(0,555,0), vec3(0,0,555), 2); 
        world[1].init_quad(point3(0,0,0), vec3(0,555,0), vec3(0,0,555), 0); 
        world[2].init_quad(point3(0,0,0), vec3(555,0,0), vec3(0,0,555), 1); 
        world[3].init_quad(point3(555,555,555), vec3(-555,0,0), vec3(0,0,-555), 1); 
        world[4].init_quad(point3(0,0,555), vec3(555,0,0), vec3(0,555,0), 1); 
        
        world[5].init_quad(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), 3); 
        area_light_indices[0] = 5;

        world[6].init_sphere(point3(190, 90, 190), 90.0f, 4); 
        world[7].init_sphere(point3(365, 90, 365), 90.0f, 1); 

        point_lights[0] = {point3(278, 278, 278), color(10000.0f, 10000.0f, 10000.0f)};

        point3 lookfrom(278, 278, -800);
        point3 lookat(278, 278, 0);
        vec3 vup(0, 1, 0);
        float vfov = 38.0f;
        cam->init(lookfrom, lookat, vup, vfov, aspect_ratio);

        // Calculate global scene bounding box
        aabb temp_box;
        world[0].bounding_box(temp_box);
        aabb total_box = temp_box;

        for (int i = 1; i < 8; i++) {
            world[i].bounding_box(temp_box);
            total_box = surrounding_box(total_box, temp_box);
        }

        *scene_box = total_box;
    }
}

int main() {
    int nx = 400; // render width
    int ny = 400; // render height
    int tx = 8; // block size in x dimension
    int ty = 8; // block size in y dimension
    int samples = 100; // spp
    int max_depth = 50;
    
    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    vec3 *fb;
    cudaMallocManaged((void **)&fb, fb_size);

    curandState *d_rand_state;
    cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState));

    Shape *d_world;
    Material *d_materials;
    int *d_area_light_indices;
    PointLight *d_point_lights;
    camera *d_cam;
    aabb *d_scene_box;

    int num_shapes = 8;
    int num_materials = 5;
    int num_area_lights = 1;
    int num_point_lights = 1;

    cudaMallocManaged((void **)&d_world, num_shapes * sizeof(Shape));
    cudaMallocManaged((void **)&d_materials, num_materials * sizeof(Material));
    cudaMallocManaged((void **)&d_area_light_indices, num_area_lights * sizeof(int));
    cudaMallocManaged((void **)&d_point_lights, num_point_lights * sizeof(PointLight));
    cudaMallocManaged((void **)&d_cam, sizeof(camera));
    cudaMallocManaged((void **)&d_scene_box, sizeof(aabb));

    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);

    std::cout << "Rendering on GPU..." << std::endl;
    
    render_init_kernel<<<blocks, threads>>>(nx, ny, d_rand_state);
    cudaGetLastError();
    cudaDeviceSynchronize();

    create_world<<<1, 1>>>(d_world, d_materials, d_point_lights, d_area_light_indices, d_cam, float(nx)/float(ny), d_scene_box);
    cudaGetLastError();
    cudaDeviceSynchronize();

    auto start_time = std::chrono::high_resolution_clock::now();

    render<<<blocks, threads>>>(fb, nx, ny, samples, max_depth, d_cam, d_world, num_shapes, d_area_light_indices, num_area_lights, d_point_lights, num_point_lights,
        d_materials, d_rand_state, d_scene_box);
    cudaGetLastError();
    cudaDeviceSynchronize();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end_time - start_time;
    std::cout << "Rendering completed in " << duration.count() << " ms (" << duration.count() / 1000.0f << " seconds)." << std::endl;

    std::cout << "Saving image..." << std::endl;
    std::vector<unsigned char> image_data(nx * ny * 3);
    for (int j = ny - 1; j >= 0; --j) {
        for (int i = 0; i < nx; ++i) {
            size_t pixel_index = j * nx + i;
            float r = fb[pixel_index].x();
            float g = fb[pixel_index].y();
            float b = fb[pixel_index].z();
            
            int png_j = ny - 1 - j;
            int idx = (png_j * nx + i) * 3;
            
            image_data[idx + 0] = static_cast<int>(256 * fminf(r, 0.999f));
            image_data[idx + 1] = static_cast<int>(256 * fminf(g, 0.999f));
            image_data[idx + 2] = static_cast<int>(256 * fminf(b, 0.999f));
        }
    }

    stbi_write_png("image_cuda.png", nx, ny, 3, image_data.data(), nx * 3);

    cudaFree(fb);
    cudaFree(d_rand_state);
    cudaFree(d_world);
    cudaFree(d_materials);
    cudaFree(d_area_light_indices);
    cudaFree(d_point_lights);
    cudaFree(d_cam);
    cudaFree(d_scene_box);

    std::cout << "Done." << std::endl;
    return 0;
}
