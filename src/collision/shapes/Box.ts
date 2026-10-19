import { Mat3 } from "../../math/Mat3.js";
import { Quat } from "../../math/Quat.js";
import type { Transform } from "../../math/Transform.js";
import { Vec3 } from "../../math/Vec3.js";
import { AABB } from "../AABB.js";
import { Shape, ShapeType } from "./Shape.js";

/** Scratch storage for the rotation matrix / local support point (single-threaded use). */
const scratchMat3 = new Mat3();
const scratchVec3 = new Vec3();

/** `sgn(component) * halfExtent` with `sgn(0) = 0`. */
function signedExtent(component: number, halfExtent: number): number {
  if (component > 0) {
    return halfExtent;
  }
  if (component < 0) {
    return -halfExtent;
  }
  return 0;
}

/**
 * Oriented box defined by positive half extents `[hx, hy, hz]` in its local
 * frame.
 *
 * Support mapping (Section 3.1): the query direction is rotated into the local
 * frame, the extreme corner is picked by component-wise sign, and the corner
 * is rotated back:
 *
 * ```
 * d_local = Rᵀ d
 * s_local = [sgn(dx) hx, sgn(dy) hy, sgn(dz) hz]
 * S(d)   = c + R s_local
 * ```
 *
 * Local inertia (Section 2.3):
 * `I = (1/12) m · diag((2hy)² + (2hz)², (2hx)² + (2hz)², (2hx)² + (2hy)²)`.
 */
export class Box extends Shape {
  public readonly type: ShapeType = ShapeType.Box;
  public readonly halfExtents: Vec3;

  constructor(halfExtents: Vec3) {
    super();
    const { x, y, z } = halfExtents;
    if (!(x > 0) || !(y > 0) || !(z > 0)) {
      throw new RangeError(`Box: half extents must be positive, got ${halfExtents.toString()}`);
    }
    this.halfExtents = halfExtents.clone();
  }

  override computeAABB(transform: Transform, out: AABB = new AABB()): AABB {
    const rotation = Quat.toMat3(scratchMat3, transform.rotation);
    const { x: hx, y: hy, z: hz } = this.halfExtents;
    const ex =
      Math.abs(rotation.m00) * hx + Math.abs(rotation.m01) * hy + Math.abs(rotation.m02) * hz;
    const ey =
      Math.abs(rotation.m10) * hx + Math.abs(rotation.m11) * hy + Math.abs(rotation.m12) * hz;
    const ez =
      Math.abs(rotation.m20) * hx + Math.abs(rotation.m21) * hy + Math.abs(rotation.m22) * hz;
    const { x, y, z } = transform.position;
    out.min.set(x - ex, y - ey, z - ez);
    out.max.set(x + ex, y + ey, z + ez);
    return out;
  }

  override getSupport(direction: Vec3, transform: Transform, out: Vec3 = new Vec3()): Vec3 {
    const rotation = Quat.toMat3(scratchMat3, transform.rotation);
    Mat3.transformVectorTranspose(scratchVec3, rotation, direction);
    const { x: hx, y: hy, z: hz } = this.halfExtents;
    scratchVec3.set(
      signedExtent(scratchVec3.x, hx),
      signedExtent(scratchVec3.y, hy),
      signedExtent(scratchVec3.z, hz),
    );
    Quat.rotateVector(out, transform.rotation, scratchVec3);
    return out.add(transform.position);
  }

  /** `I = (1/12) m · diag((2hy)² + (2hz)², (2hx)² + (2hz)², (2hx)² + (2hy)²)`. */
  computeLocalInertia(mass: number, out: Mat3 = new Mat3()): Mat3 {
    if (!(mass > 0)) {
      throw new RangeError(`Box.computeLocalInertia: mass must be positive, got ${mass}`);
    }
    const { x: hx, y: hy, z: hz } = this.halfExtents;
    const ex = 2 * hx;
    const ey = 2 * hy;
    const ez = 2 * hz;
    const factor = mass / 12;
    return out.set(
      factor * (ey * ey + ez * ez),
      0,
      0,
      0,
      factor * (ex * ex + ez * ez),
      0,
      0,
      0,
      factor * (ex * ex + ey * ey),
    );
  }

  /** `V = 8 hx hy hz`. */
  volume(): number {
    const { x, y, z } = this.halfExtents;
    return 8 * x * y * z;
  }
}
