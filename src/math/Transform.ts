import { Mat4 } from "./Mat4.js";
import { Quat } from "./Quat.js";
import { Vec3 } from "./Vec3.js";

/**
 * Scratch quaternion used by the inverse rotation paths. Safe because every
 * `Transform` method is synchronous and the scratch is never live across a
 * callback boundary.
 */
const scratchConjugate = new Quat();

/**
 * Rigid transform: a translation `position` composed with a unit-quaternion
 * `rotation` (rotation applied first, then translation):
 *
 * ```
 * T(p) = R(q) p + x
 * ```
 *
 * The class is a plain value carrier: `RigidBody` (Phase 4) will own the
 * canonical `position`/`quaternion` state and construct `Transform` views from
 * it for the narrowphase.
 */
export class Transform {
  public readonly position: Vec3;
  public readonly rotation: Quat;

  constructor(position: Vec3 = new Vec3(), rotation: Quat = new Quat()) {
    this.position = position;
    this.rotation = rotation;
  }

  set(position: Vec3, rotation: Quat): this {
    this.position.copy(position);
    this.rotation.copy(rotation);
    return this;
  }

  setIdentity(): this {
    this.position.setZero();
    this.rotation.identity();
    return this;
  }

  copy(transform: Transform): this {
    return this.set(transform.position, transform.rotation);
  }

  clone(): Transform {
    return new Transform().copy(this);
  }

  /** `out = R(q) p + x`, i.e. rotation followed by translation. */
  transformPoint(out: Vec3, point: Vec3): Vec3 {
    Quat.rotateVector(out, this.rotation, point);
    return out.add(this.position);
  }

  /** `out = R(q) d` (translation ignored). */
  transformDirection(out: Vec3, direction: Vec3): Vec3 {
    return Quat.rotateVector(out, this.rotation, direction);
  }

  /** `out = R*(q) (p - x)`, the exact inverse of {@link Transform.transformPoint}. */
  inverseTransformPoint(out: Vec3, point: Vec3): Vec3 {
    out.copy(point).sub(this.position);
    Quat.conjugate(scratchConjugate, this.rotation);
    return Quat.rotateVector(out, scratchConjugate, out);
  }

  /** `out = R*(q) d`, the exact inverse of {@link Transform.transformDirection}. */
  inverseTransformDirection(out: Vec3, direction: Vec3): Vec3 {
    Quat.conjugate(scratchConjugate, this.rotation);
    return Quat.rotateVector(out, scratchConjugate, direction);
  }

  /** Builds the affine matrix `[R(q) | x]` of this transform. */
  toMat4(out: Mat4 = new Mat4()): Mat4 {
    return Mat4.fromRotationTranslation(out, this.rotation, this.position);
  }

  toString(): string {
    return `Transform(${this.position.toString()}, ${this.rotation.toString()})`;
  }
}
