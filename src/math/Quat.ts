import { clamp, EPSILON_SQ } from "./MathUtils.js";
import { Mat3 } from "./Mat3.js";
import { Vec3 } from "./Vec3.js";

/**
 * Unit quaternion `q = [w, x, y, z]` in Hamilton convention, where `w` is the
 * scalar part (noted `s` in the engine specification) and `[x, y, z]` the
 * vector part.
 *
 * ## Convention
 * A quaternion represents the **active** rotation that maps a vector `v` to
 * `q ⊗ v ⊗ q*`. With `w = cos(θ/2)` and `[x, y, z] = sin(θ/2) * axis` this is
 * a counter-clockwise rotation by `θ` about `axis` in a right-handed frame,
 * matching the rotation matrix of Section 2.1 of the specification.
 *
 * ## Composition direction
 * `multiply(q)` means `this = this ⊗ q`, i.e. the rotation `q` is applied
 * *first*, then `this`. This matches the column-vector convention
 * `R(a ⊗ b) = R(a) R(b)`.
 *
 * ## Angular velocity integration
 * {@link Quat.integrateAngularVelocity} implements the specification formula
 * `q(t + dt) = normalize(q + 0.5 * dt * Ω ⊗ q)` with `Ω = [0, ω]`, where `ω`
 * is the angular velocity expressed in the **world** frame.
 *
 * ## Storage
 * Components are four contiguous scalars in `[w, x, y, z]` order for WASM/GPU
 * interop (see {@link Quat.writeTo}).
 */
export class Quat {
  public w: number;
  public x: number;
  public y: number;
  public z: number;

  /** Constructs the identity rotation `[1, 0, 0, 0]`. */
  constructor() {
    this.w = 1;
    this.x = 0;
    this.y = 0;
    this.z = 0;
  }

  /** Returns the identity rotation `[1, 0, 0, 0]`. */
  static identity(): Quat {
    return new Quat();
  }

  /** Returns `[w, x, y, z]` unnormalized, as given. */
  static fromValues(w: number, x: number, y: number, z: number): Quat {
    const q = new Quat();
    return q.set(w, x, y, z);
  }

  /**
   * Rotation of `angle` radians about `axis` (need not be unit length).
   * @throws RangeError when `axis` is the zero vector.
   */
  static fromAxisAngle(axis: Vec3, angle: number, out = new Quat()): Quat {
    const lengthSq = axis.lengthSq();
    if (lengthSq < EPSILON_SQ) {
      throw new RangeError("Quat.fromAxisAngle: axis must be non-zero");
    }
    const half = angle * 0.5;
    const s = Math.sin(half) / Math.sqrt(lengthSq);
    out.w = Math.cos(half);
    out.x = axis.x * s;
    out.y = axis.y * s;
    out.z = axis.z * s;
    return out;
  }

  /** Reads four consecutive scalars `[w, x, y, z]` starting at `offset`. */
  static fromArray(array: ArrayLike<number>, offset = 0, out = new Quat()): Quat {
    if (offset < 0 || offset + 4 > array.length) {
      throw new RangeError(
        `Quat.fromArray: reading [${offset}, ${offset + 4}) exceeds length ${array.length}`,
      );
    }
    out.w = array[offset];
    out.x = array[offset + 1];
    out.y = array[offset + 2];
    out.z = array[offset + 3];
    return out;
  }

  /** Writes `[w, x, y, z]` as four consecutive scalars starting at `offset`. */
  writeTo(array: Float32Array | Float64Array, offset = 0): void {
    if (offset < 0 || offset + 4 > array.length) {
      throw new RangeError(
        `Quat.writeTo: writing [${offset}, ${offset + 4}) exceeds length ${array.length}`,
      );
    }
    array[offset] = this.w;
    array[offset + 1] = this.x;
    array[offset + 2] = this.y;
    array[offset + 3] = this.z;
  }

  set(w: number, x: number, y: number, z: number): this {
    this.w = w;
    this.x = x;
    this.y = y;
    this.z = z;
    return this;
  }

  /** Resets to the identity rotation. */
  identity(): this {
    this.w = 1;
    this.x = 0;
    this.y = 0;
    this.z = 0;
    return this;
  }

  copy(q: Quat): this {
    this.w = q.w;
    this.x = q.x;
    this.y = q.y;
    this.z = q.z;
    return this;
  }

  clone(): Quat {
    const q = new Quat();
    return q.copy(this);
  }

  /**
   * Hamilton product `this = this ⊗ q` (rotate by `q` first, then by `this`).
   */
  multiply(q: Quat): this {
    Quat.multiply(this, this, q);
    return this;
  }

  /** `this = this* = [w, -x, -y, -z]` */
  conjugate(): this {
    Quat.conjugate(this, this);
    return this;
  }

  /** `this /= |this|`, preserving the rotation encoded by `this`. */
  normalize(): this {
    Quat.normalize(this, this);
    return this;
  }

  /** Inner product, `|dot| -> 1` for aligned unit rotations. */
  dot(q: Quat): number {
    return this.w * q.w + this.x * q.x + this.y * q.y + this.z * q.z;
  }

  /** Squared norm `q . q`. */
  lengthSq(): number {
    return this.w * this.w + this.x * this.x + this.y * this.y + this.z * this.z;
  }

  /** Euclidean norm `|q|`, `1` for a unit rotation. */
  length(): number {
    return Math.sqrt(this.lengthSq());
  }

  /** Rotates `v` by `this`, writing into `out` (`out` may alias `v`). */
  rotateVector(v: Vec3, out = new Vec3()): Vec3 {
    return Quat.rotateVector(out, this, v);
  }

  /** Writes the rotation matrix of this quaternion into `out`. */
  toMat3(out = new Mat3()): Mat3 {
    return Quat.toMat3(out, this);
  }

  /** Advances `this` by world-frame angular velocity `omega` over `dt`. */
  integrateAngularVelocity(omega: Vec3, dt: number): this {
    Quat.integrateAngularVelocity(this, this, omega, dt);
    return this;
  }

  /** Component-wise equality within `tolerance` (default exact). */
  equals(q: Quat, tolerance: number = 0): boolean {
    return (
      Math.abs(this.w - q.w) <= tolerance &&
      Math.abs(this.x - q.x) <= tolerance &&
      Math.abs(this.y - q.y) <= tolerance &&
      Math.abs(this.z - q.z) <= tolerance
    );
  }

  toString(): string {
    return `Quat(${this.w}, ${this.x}, ${this.y}, ${this.z})`;
  }

  // ------------------------------------------------------------------
  // Static operations (out-parameter, alias-safe)
  // ------------------------------------------------------------------

  /** `out = a ⊗ b` (Hamilton product, rotate by `b` first). */
  static multiply(out: Quat, a: Quat, b: Quat): Quat {
    const w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
    const x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
    const y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;
    const z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;
    out.w = w;
    out.x = x;
    out.y = y;
    out.z = z;
    return out;
  }

  /** `out = a*` (negated vector part; the inverse for unit quaternions). */
  static conjugate(out: Quat, a: Quat): Quat {
    out.w = a.w;
    out.x = -a.x;
    out.y = -a.y;
    out.z = -a.z;
    return out;
  }

  /** `out = a⁻¹ = a* / |a|²`. */
  static inverse(out: Quat, a: Quat): Quat {
    const lengthSq = a.lengthSq();
    if (lengthSq < EPSILON_SQ) {
      throw new RangeError("Quat.inverse: cannot invert a zero quaternion");
    }
    const inverseLengthSq = 1 / lengthSq;
    out.w = a.w * inverseLengthSq;
    out.x = -a.x * inverseLengthSq;
    out.y = -a.y * inverseLengthSq;
    out.z = -a.z * inverseLengthSq;
    return out;
  }

  /** `out = a / |a|`. @throws RangeError when `|a|² < EPSILON_SQ`. */
  static normalize(out: Quat, a: Quat): Quat {
    const lengthSq = a.lengthSq();
    if (lengthSq < EPSILON_SQ) {
      throw new RangeError("Quat.normalize: cannot normalize a zero quaternion");
    }
    const inverseLength = 1 / Math.sqrt(lengthSq);
    out.w = a.w * inverseLength;
    out.x = a.x * inverseLength;
    out.y = a.y * inverseLength;
    out.z = a.z * inverseLength;
    return out;
  }

  /** Inner product of `a` and `b`. */
  static dot(a: Quat, b: Quat): number {
    return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
  }

  /** Squared Euclidean norm of `a`. */
  static lengthSq(a: Quat): number {
    return a.w * a.w + a.x * a.x + a.y * a.y + a.z * a.z;
  }

  /**
   * Spherical linear interpolation from `a` to `b`, taking the **shortest
   * arc** (negating `b` when the rotations are more than 180° apart in
   * `SO(3)`). `t` is clamped to `[0, 1]`; both inputs are assumed unit length.
   */
  static slerp(out: Quat, a: Quat, b: Quat, t: number): Quat {
    const clampedT = clamp(t, 0, 1);
    let bx = b.x;
    let by = b.y;
    let bz = b.z;
    let bw = b.w;
    let cosHalfTheta = a.dot(b);
    if (cosHalfTheta < 0) {
      cosHalfTheta = -cosHalfTheta;
      bw = -bw;
      bx = -bx;
      by = -by;
      bz = -bz;
    }
    if (cosHalfTheta > 0.9995) {
      out.w = a.w + clampedT * (bw - a.w);
      out.x = a.x + clampedT * (bx - a.x);
      out.y = a.y + clampedT * (by - a.y);
      out.z = a.z + clampedT * (bz - a.z);
      return Quat.normalize(out, out);
    }
    const halfTheta = Math.acos(clamp(cosHalfTheta, -1, 1));
    const sinHalfTheta = Math.sqrt(1 - cosHalfTheta * cosHalfTheta);
    const weightA = Math.sin((1 - clampedT) * halfTheta) / sinHalfTheta;
    const weightB = Math.sin(clampedT * halfTheta) / sinHalfTheta;
    out.w = weightA * a.w + weightB * bw;
    out.x = weightA * a.x + weightB * bx;
    out.y = weightA * a.y + weightB * by;
    out.z = weightA * a.z + weightB * bz;
    return out;
  }

  /**
   * Rotates `v` by `q` using the streak-optimized form
   * `v' = v + 2 q_w (q_v x v) + 2 q_v x (q_v x v)`, which avoids building the
   * full `q ⊗ v ⊗ q*` product. `q` is assumed unit length.
   */
  static rotateVector(out: Vec3, q: Quat, v: Vec3): Vec3 {
    const tx = 2 * (q.y * v.z - q.z * v.y);
    const ty = 2 * (q.z * v.x - q.x * v.z);
    const tz = 2 * (q.x * v.y - q.y * v.x);
    const x = v.x + q.w * tx + q.y * tz - q.z * ty;
    const y = v.y + q.w * ty + q.z * tx - q.x * tz;
    const z = v.z + q.w * tz + q.x * ty - q.y * tx;
    out.x = x;
    out.y = y;
    out.z = z;
    return out;
  }

  /** Rotation matrix `R(q)` as defined in Section 2.1 of the specification. */
  static toMat3(out: Mat3, q: Quat): Mat3 {
    const x = q.x;
    const y = q.y;
    const z = q.z;
    const w = q.w;
    const x2 = x + x;
    const y2 = y + y;
    const z2 = z + z;
    const xx = x * x2;
    const xy = x * y2;
    const xz = x * z2;
    const yy = y * y2;
    const yz = y * z2;
    const zz = z * z2;
    const wx = w * x2;
    const wy = w * y2;
    const wz = w * z2;
    out.m00 = 1 - (yy + zz);
    out.m01 = xy - wz;
    out.m02 = xz + wy;
    out.m10 = xy + wz;
    out.m11 = 1 - (xx + zz);
    out.m12 = yz - wx;
    out.m20 = xz - wy;
    out.m21 = yz + wx;
    out.m22 = 1 - (xx + yy);
    return out;
  }

  /**
   * Advance `q` by a constant world-frame angular velocity `omega` over `dt`:
   * `out = normalize(q + 0.5 * dt * (Ω ⊗ q))` with `Ω = [0, omega]`.
   *
   * The un-normalized step is exact for a rotation about a fixed axis
   * (the normalized result equals `exp(Ω dt / 2) ⊗ q` up to the O(dt³)
   * angle error of the chord approximation), and `out` may alias `q`.
   */
  static integrateAngularVelocity(out: Quat, q: Quat, omega: Vec3, dt: number): Quat {
    const { x: wx, y: wy, z: wz } = omega;
    const qw = q.w;
    const qx = q.x;
    const qy = q.y;
    const qz = q.z;

    const dw = -(wx * qx + wy * qy + wz * qz);
    const dx = qw * wx + wy * qz - wz * qy;
    const dy = qw * wy - wx * qz + wz * qx;
    const dz = qw * wz + wx * qy - wy * qx;

    const halfDt = 0.5 * dt;
    out.w = qw + halfDt * dw;
    out.x = qx + halfDt * dx;
    out.y = qy + halfDt * dy;
    out.z = qz + halfDt * dz;
    return Quat.normalize(out, out);
  }
}
