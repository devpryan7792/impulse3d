import { EPSILON } from "../../math/MathUtils.js";
import type { Transform } from "../../math/Transform.js";
import { Vec3 } from "../../math/Vec3.js";
import { AABB } from "../AABB.js";
import { Shape, ShapeType } from "./Shape.js";

/** Scratch normal for {@link Plane.getWorldOffset} (single-threaded use). */
const scratchNormal = new Vec3();

/**
 * Infinite plane (a half-space boundary) defined in its local frame by
 * `normal · p = offset`, with `normal` unit length. The default plane is the
 * XZ ground plane `y = 0`.
 *
 * A plane is unbounded: {@link Plane.computeAABB} reports an infinite box, the
 * broadphase treats it as a node overlapping every query, and
 * {@link Plane.getSupport} rejects every non-zero query direction because the
 * support mapping of an unbounded set is divergent. Plane collisions must be
 * resolved by the analytic plane narrowphase rather than GJK.
 */
export class Plane extends Shape {
  public readonly type: ShapeType = ShapeType.Plane;
  public readonly localNormal: Vec3;
  public readonly offset: number;

  constructor(localNormal: Vec3 = new Vec3(0, 1, 0), offset: number = 0) {
    super();
    const length = localNormal.length();
    if (!(length > EPSILON)) {
      throw new RangeError(`Plane: normal must be non-zero, got ${localNormal.toString()}`);
    }
    if (!Number.isFinite(offset)) {
      throw new RangeError(`Plane: offset must be finite, got ${offset}`);
    }
    this.localNormal = localNormal.clone().scale(1 / length);
    this.offset = offset;
  }

  /** An infinite box: planes are unbounded in every direction. */
  override computeAABB(_transform: Transform, out: AABB = new AABB()): AABB {
    out.min.set(-Infinity, -Infinity, -Infinity);
    out.max.set(Infinity, Infinity, Infinity);
    return out;
  }

  /**
   * @throws RangeError for every non-zero direction: the support mapping of an
   * infinite plane is unbounded (`argmax (p · d)` does not exist). For the
   * zero direction every plane point is a maximizer, so the point of the plane
   * closest to the world origin is returned.
   */
  override getSupport(direction: Vec3, transform: Transform, out: Vec3 = new Vec3()): Vec3 {
    if (direction.lengthSq() > 0) {
      throw new RangeError(
        "Plane.getSupport: the support mapping of an infinite plane is unbounded; " +
          "use the analytic plane narrowphase instead",
      );
    }
    this.getWorldNormal(transform, out);
    return out.scale(this.getWorldOffset(transform));
  }

  /** Unit world-space normal `R(q) n_local`. */
  getWorldNormal(transform: Transform, out: Vec3 = new Vec3()): Vec3 {
    return transform.transformDirection(out, this.localNormal);
  }

  /**
   * Plane constant `C` in world space such that a point `p` lies on the plane
   * iff `n_world · p = C`.
   */
  getWorldOffset(transform: Transform): number {
    return this.getWorldNormal(transform, scratchNormal).dot(transform.position) + this.offset;
  }

  override toString(): string {
    return `Plane(normal=${this.localNormal.toString()}, offset=${this.offset})`;
  }
}
