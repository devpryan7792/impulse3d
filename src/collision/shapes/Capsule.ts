import { Mat3 } from "../../math/Mat3.js";
import { Quat } from "../../math/Quat.js";
import type { Transform } from "../../math/Transform.js";
import { Vec3 } from "../../math/Vec3.js";
import { AABB } from "../AABB.js";
import { Shape, ShapeType } from "./Shape.js";

/** Scratch storage for the segment endpoints / support direction (single-threaded use). */
const scratchA = new Vec3();
const scratchB = new Vec3();
const scratchDirection = new Vec3();
const scratchMat3 = new Mat3();
const fallbackDirection = new Vec3(1, 0, 0);

/**
 * Capsule: the Minkowski sum of the segment `[a, b]` (local `+Y` axis,
 * `a = (0, -h, 0)`, `b = (0, h, 0)` with `halfHeight = h`) and a ball of
 * radius `r`. Total height is `2h + 2r`.
 *
 * Support mapping (Section 3.1):
 * `S(d) = (d·a > d·b ? a : b) + r * d / |d|`, evaluating `a`/`b` in world
 * space under the given transform.
 *
 * Local inertia (Section 2.3): composite of a solid cylinder and two solid
 * hemispheres, each term shifted to the capsule center with the parallel axis
 * theorem.
 */
export class Capsule extends Shape {
  public readonly type: ShapeType = ShapeType.Capsule;
  public readonly radius: number;
  public readonly halfHeight: number;

  constructor(radius: number, halfHeight: number) {
    super();
    if (!(radius > 0)) {
      throw new RangeError(`Capsule: radius must be positive, got ${radius}`);
    }
    if (!(halfHeight > 0)) {
      throw new RangeError(`Capsule: half height must be positive, got ${halfHeight}`);
    }
    this.radius = radius;
    this.halfHeight = halfHeight;
  }

  override computeAABB(transform: Transform, out: AABB = new AABB()): AABB {
    const rotation = Quat.toMat3(scratchMat3, transform.rotation);
    const h = this.halfHeight;
    const r = this.radius;
    const ex = Math.abs(rotation.m01) * h + r;
    const ey = Math.abs(rotation.m11) * h + r;
    const ez = Math.abs(rotation.m21) * h + r;
    const { x, y, z } = transform.position;
    out.min.set(x - ex, y - ey, z - ez);
    out.max.set(x + ex, y + ey, z + ez);
    return out;
  }

  override getSupport(direction: Vec3, transform: Transform, out: Vec3 = new Vec3()): Vec3 {
    const h = this.halfHeight;
    scratchA.set(0, -h, 0);
    transform.transformPoint(scratchA, scratchA);
    scratchB.set(0, h, 0);
    transform.transformPoint(scratchB, scratchB);
    const base = direction.dot(scratchA) > direction.dot(scratchB) ? scratchA : scratchB;
    Vec3.normalize(scratchDirection, direction, fallbackDirection);
    const r = this.radius;
    out.x = base.x + scratchDirection.x * r;
    out.y = base.y + scratchDirection.y * r;
    out.z = base.z + scratchDirection.z * r;
    return out;
  }

  /**
   * Composite inertia: solid cylinder of height `2 * halfHeight` plus two
   * hemispheres, mass split by volume fraction and each sub-body shifted to
   * the capsule center with the parallel axis theorem.
   */
  computeLocalInertia(mass: number, out: Mat3 = new Mat3()): Mat3 {
    if (!(mass > 0)) {
      throw new RangeError(`Capsule.computeLocalInertia: mass must be positive, got ${mass}`);
    }
    const r = this.radius;
    const h = 2 * this.halfHeight;
    const k = this.halfHeight;
    const r2 = r * r;

    const cylinderVolume = Math.PI * r2 * h;
    const capsVolume = (4 / 3) * Math.PI * r2 * r;
    const massCylinder = (mass * cylinderVolume) / (cylinderVolume + capsVolume);
    const massCaps = mass - massCylinder;
    const massHemisphere = 0.5 * massCaps;

    const transverseCylinder = (massCylinder * (3 * r2 + h * h)) / 12;
    const transverseCaps = 2 * massHemisphere * (0.4 * r2 + k * k + 0.75 * r * k);
    const axialCylinder = 0.5 * massCylinder * r2;
    const axialCaps = 2 * 0.4 * massHemisphere * r2;
    const transverse = transverseCylinder + transverseCaps;
    const axial = axialCylinder + axialCaps;
    return out.set(transverse, 0, 0, 0, axial, 0, 0, 0, transverse);
  }

  /** `V = π r² (2 · halfHeight) + (4/3) π r³`. */
  volume(): number {
    const r2 = this.radius * this.radius;
    return Math.PI * r2 * 2 * this.halfHeight + (4 / 3) * Math.PI * r2 * this.radius;
  }
}
