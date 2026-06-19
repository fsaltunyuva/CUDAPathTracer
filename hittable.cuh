#pragma once

#include "ray.cuh"
#include <curand_kernel.h>

enum class MatType { LAMBERTIAN, METAL, LIGHT };

struct Material {
    MatType type;
    color albedo;
    float fuzz;
    color emit;

    __device__ Material() {}
    __device__ Material(MatType t, color a, float f, color e) : type(t), albedo(a), fuzz(f), emit(e) {}
};

struct HitRecord {
    point3 p;
    vec3 normal;
    float t;
    bool front_face;
    int mat_index;

    __device__ void set_face_normal(const ray& r, const vec3& outward_normal) {
        // Determine if the ray is hitting the front face or the back face of the surface
        front_face = dot(r.direction(), outward_normal) < 0.0f;

        if (front_face)
            normal = outward_normal;
        else
            normal = -outward_normal;
    }
};

struct aabb {
    point3 minimum;
    point3 maximum;

    __device__ aabb() {}
    __device__ aabb(const point3& a, const point3& b) {
        minimum = point3(fminf(a.x(), b.x()), fminf(a.y(), b.y()), fminf(a.z(), b.z()));
        maximum = point3(fmaxf(a.x(), b.x()), fmaxf(a.y(), b.y()), fmaxf(a.z(), b.z()));
    }

    __device__ bool hit(const ray& r, float t_min, float t_max) const {
        for (int a = 0; a < 3; a++) {
            auto invD = 1.0f / r.direction()[a];
            auto t0 = (minimum[a] - r.origin()[a]) * invD;
            auto t1 = (maximum[a] - r.origin()[a]) * invD;

            if (invD < 0.0f) {
                float temp = t0;
                t0 = t1;
                t1 = temp;
            }

            if (t0 > t_min)
                t_min = t0;

            if (t1 < t_max)
                t_max = t1;

            if (t_max <= t_min)
                return false;
        }
        return true;
    }
};

__device__ inline aabb surrounding_box(aabb box0, aabb box1) {
    point3 small(fminf(box0.minimum.x(), box1.minimum.x()),
                 fminf(box0.minimum.y(), box1.minimum.y()),
                 fminf(box0.minimum.z(), box1.minimum.z()));

    point3 big(fmaxf(box0.maximum.x(), box1.maximum.x()),
               fmaxf(box0.maximum.y(), box1.maximum.y()),
               fmaxf(box0.maximum.z(), box1.maximum.z()));

    return aabb(small, big);
}

enum class ShapeType { SPHERE, QUAD };

struct Shape {
    ShapeType type;
    int mat_index;
    aabb bbox;

    // Sphere data
    point3 center;
    float radius;

    // Quad data
    point3 Q;
    vec3 u, v;
    vec3 normal;
    float D;
    vec3 w;
    float area;

    __device__ Shape() {}

    // Constructor for Sphere
    __device__ void init_sphere(point3 cen, float r, int m) {
        type = ShapeType::SPHERE;
        center = cen;
        radius = r;
        mat_index = m;
        bbox = aabb(center - vec3(radius, radius, radius), center + vec3(radius, radius, radius));
    }

    // Constructor for Quad
    __device__ void init_quad(point3 _Q, vec3 _u, vec3 _v, int m) {
        type = ShapeType::QUAD;
        Q = _Q;
        u = _u;
        v = _v;
        mat_index = m;

        vec3 n = cross(u, v);
        normal = unit_vector(n);
        D = dot(normal, Q);
        w = n / dot(n, n);
        area = n.length();

        // Calculate bounding box
        aabb box0(Q, Q + u + v);
        aabb box1(Q + u, Q + v);
        bbox = surrounding_box(box0, box1);

        // Add padding to AABB if it's perfectly flat to avoid issues with ray-AABB intersection
        float pad = 0.0001f;
        if (fabsf(bbox.maximum.x() - bbox.minimum.x()) < pad) {
            bbox.minimum.e[0] -= pad;
            bbox.maximum.e[0] += pad;
        }
        if (fabsf(bbox.maximum.y() - bbox.minimum.y()) < pad) {
            bbox.minimum.e[1] -= pad;
            bbox.maximum.e[1] += pad;
        }
        if (fabsf(bbox.maximum.z() - bbox.minimum.z()) < pad) {
            bbox.minimum.e[2] -= pad;
            bbox.maximum.e[2] += pad;
        }
    }

    __device__ bool hit(const ray& r, float t_min, float t_max, HitRecord& rec) const {
        if (type == ShapeType::SPHERE) {
            vec3 oc = r.origin() - center;
            auto a = r.direction().length_squared();
            auto half_b = dot(oc, r.direction());
            auto c = oc.length_squared() - radius*radius;

            auto discriminant = half_b*half_b - a*c;
            if (discriminant < 0.0f) return false;
            auto sqrtd = sqrtf(discriminant);

            auto root = (-half_b - sqrtd) / a;
            if (root < t_min || t_max < root) {
                root = (-half_b + sqrtd) / a;
                if (root < t_min || t_max < root)
                    return false;
            }

            rec.t = root;
            rec.p = r.at(rec.t);
            vec3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);
            rec.mat_index = mat_index;
            return true;
        }

        if (type == ShapeType::QUAD) {
            auto denom = dot(normal, r.direction());
            if (fabsf(denom) < 1e-8f)
                return false;

            auto t = (D - dot(normal, r.origin())) / denom;
            if (t < t_min || t > t_max)
                return false;

            auto intersection = r.at(t);
            vec3 planar_hitpt_vector = intersection - Q;
            auto alpha = dot(w, cross(planar_hitpt_vector, v));
            auto beta = dot(w, cross(u, planar_hitpt_vector));

            if ((alpha < 0.0f) || (1.0f < alpha) || (beta < 0.0f) || (1.0f < beta))
                return false;

            rec.t = t;
            rec.p = intersection;
            rec.mat_index = mat_index;
            rec.set_face_normal(r, normal);
            return true;
        }

        return false;
    }

    __device__ bool bounding_box(aabb& output_box) const {
        output_box = bbox;
        return true;
    }

    __device__ float pdf_value(const point3& origin, const vec3& v_dir) const {
        if (type == ShapeType::QUAD) {
            HitRecord rec;
            if (!this->hit(ray(origin, v_dir), 0.001f, 10000.0f, rec))
                return 0.0f;

            auto distance_squared = rec.t * rec.t * v_dir.length_squared();
            auto cosine = fabsf(dot(v_dir, rec.normal) / v_dir.length());

            return distance_squared / (cosine * area);
        }
        return 0.0f;
    }

    __device__ vec3 random(const point3& origin, curandState* local_rand_state) const {
        if (type == ShapeType::QUAD) {
            auto p = Q + (curand_uniform(local_rand_state) * u) + (curand_uniform(local_rand_state) * v);
            return p - origin;
        }
        return vec3(1,0,0);
    }
};

struct PointLight {
    point3 pos;
    color emit;
};
