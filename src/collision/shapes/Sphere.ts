import { EPSILON_SQ } from "../../math/MathUtils.js";
import { Mat3 } from "../../math/Mat3.js";
import type { Transform } from "../../math/Transform.js";
import { Vec3 } from "../../math/Vec3.js";
import { AABB } from "../AABB.js";
import { Shape, ShapeType } from "./Shape.js";

/**
 * Solid sphere of radius `r` centered on the transform origin.
 *
 * Support mapping (Section 3.1): `S(d) = x + r * d / |d|`, falling back to
 * `+x` for a degenerate (zero) query direction.
 *
 * Local inertia (Section 2.3): `I = (2/5) m r² * I₃`.
 */
export class Sphere extends Shape {
  public readonly type: ShapeType = ShapeType.Sphere;
  public readonly radius: number;

  constructor(radius: number) {
    super();
    if (!(radius > 0)) {
      throw new RangeError(`Sphere: radius must be positive, got ${radius}`);
    }
    this.radius = radius;
  }

  override computeAABB(transform: Transform, out: AABB = new AABB()): AABB {
    const { x, y, z } = transform.position;
    const r = this.radius;
    out.min.set(x - r, y - r, z - r);
    out.max.set(x + r, y + r, z + r);
    return out;
  }

  override getSupport(direction: Vec3, transform: Transform, out: Vec3 = new Vec3()): Vec3 {
    const lengthSq = direction.lengthSq();
    if (lengthSq < EPSILON_SQ) {
      out.set(this.radius, 0, 0);
    } else {
      const scale = this.radius / Math.sqrt(lengthSq);
      out.set(direction.x * scale, direction.y * scale, direction.z * scale);
    }
    return out.add(transform.position);
  }

  /** `I = (2/5) m r²` about every axis through the center. */
  computeLocalInertia(mass: number, out: Mat3 = new Mat3()): Mat3 {
    if (!(mass > 0)) {
      throw new RangeError(`Sphere.computeLocalInertia: mass must be positive, got ${mass}`);
    }
    const i = 0.4 * mass * this.radius * this.radius;
    return out.set(i, 0, 0, 0, i, 0, 0, 0, i);
  }

  /** `V = (4/3) π r³`. */
  volume(): number {
    const r = this.radius;
    return (4 / 3) * Math.PI * r * r * r;
  }
}
