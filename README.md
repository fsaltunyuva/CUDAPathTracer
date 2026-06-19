# CUDA Path Tracer
Basic GPU Path Tracer developed to compare the speedup against CPU Path Tracer for the term project of the MMI713 (Applied Parallel Programming on GPU) course.

[ground truth reference image]

*Ground truth reference render of the custom Cornell Box scene. The image was rendered at a resolution of 800x800 with 10,000 samples per pixel (SPP), a maximum ray depth of 100, and Next Event Estimation (NEE) enabled (Rendered on my custom GPU Path Tracer in 54.19 seconds).*

It is an iterative GPU Accelerated Path Tracer that uses CUDA to accelerate the rendering process. The project aims to demonstrate the performance benefits of using GPU for path tracing compared to traditional CPU implementations. Therefore, it lacks features like BVH acceleration structure, glass/dielectric materials (Snell's law, Fresnel equations), texture mapping, depth of field, motion blur, and Multiple Importance Sampling (MIS) for area lights.

It includes AABB Scene Bounding Box, Sphere and Quad primitive types, and Next Event Estimation (NEE). The renderer supports diffuse, specular, and emissive materials.

It only renders a hard coded custom Cornell Box scene consisting of diffuse walls, a specular metal sphere, a white sphere, an area light, a point light (supplementary), and a camera with a 38.0° vertical FOV value.

I also used this path tracer's renders on Non-Local Means (NLM) Denoiser to test the possible denoising capabilities of the NLM algorithm for noisy path tracing renders. The NLM denoiser was also implemented in CUDA and is available in [this repository](https://github.com/fsaltunyuva/CUDANLMDenoiser).