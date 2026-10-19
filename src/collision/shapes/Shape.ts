import type { Transform } from "../../math/Transform.js";
import type { Vec3 } from "../../math/Vec3.js";
import type { AABB } from "../AABB.js";

/** Discriminant for the concrete convex shapes implemented in this directory. */
export const ShapeType = {
  Sphere: 0,
  Box: 1,
  Capsule: 2,
  Plane: 3,
  ConvexPolytope: 4,
} as const;

export type ShapeType = (typeof ShapeType)[keyof typeof ShapeType];

/**
 * A convex collision geometry described through the **support mapping**
 * interface `S_A(d) = argmax_{p ∈ A} (p · d)` used by GJK/EPA (Section 3 of
 * the specification).
 *
 * Both methods accept an optional `out` storage object so the narrowphase can
 * operate allocation-free; when omitted a fresh object is returned.
 *
 * `Plane` is the one unbounded member: its AABB is infinite and it only
 * supports the zero direction (see `Plane.getSupport`).
 */
export abstract class Shape {
  /** Concrete shape discriminant, useful for analytic narrowphase dispatch. */
  public abstract readonly type: ShapeType;

  /** Computes the tight world-space AABB of the shape under `transform`. */
  public abstract computeAABB(transform: Transform, out?: AABB): AABB;

  /**
   * Support point of the shape in world space for `direction` (need not be
   * unit length; the zero direction is handled through a shape-specific
   * fallback so the mapping stays total).
   */
  public abstract getSupport(direction: Vec3, transform: Transform, out?: Vec3): Vec3;
}
