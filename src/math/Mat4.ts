import { EPSILON } from "./MathUtils.js";
import { Mat3 } from "./Mat3.js";
import { Quat } from "./Quat.js";
import { Vec3 } from "./Vec3.js";

/**
 * Double-precision 4x4 matrix stored as sixteen contiguous scalars in
 * **row-major** order.
 *
 * With the column-vector convention (`p' = M p`) the translation lives in the
 * last column (`m03, m13, m23`) and `m33 = 1` for affine transforms, matching
 * the renderer/projection pipeline the physics core must interoperate with.
 *
 * Instance methods mutate `this`; static methods write into an explicit `out`
 * matrix and are alias-safe.
 */
export class Mat4 {
  public m00: number;
  public m01: number;
  public m02: number;
  public m03: number;
  public m10: number;
  public m11: number;
  public m12: number;
  public m13: number;
  public m20: number;
  public m21: number;
  public m22: number;
  public m23: number;
  public m30: number;
  public m31: number;
  public m32: number;
  public m33: number;

  /** Constructs the identity matrix. */
  constructor() {
    this.m00 = 1;
    this.m01 = 0;
    this.m02 = 0;
    this.m03 = 0;
    this.m10 = 0;
    this.m11 = 1;
    this.m12 = 0;
    this.m13 = 0;
    this.m20 = 0;
    this.m21 = 0;
    this.m22 = 1;
    this.m23 = 0;
    this.m30 = 0;
    this.m31 = 0;
    this.m32 = 0;
    this.m33 = 1;
  }

  /** Returns the sixteen components in row-major order, unmodified. */
  static fromValues(
    m00: number,
    m01: number,
    m02: number,
    m03: number,
    m10: number,
    m11: number,
    m12: number,
    m13: number,
    m20: number,
    m21: number,
    m22: number,
    m23: number,
    m30: number,
    m31: number,
    m32: number,
    m33: number,
  ): Mat4 {
    return new Mat4().set(
      m00, m01, m02, m03,
      m10, m11, m12, m13,
      m20, m21, m22, m23,
      m30, m31, m32, m33,
    );
  }

  /**
   * Embeds a 3x3 linear map in the upper-left block; the remaining components
   * form the identity (`m33 = 1`, zero translation).
   */
  static fromMat3(out: Mat4, m: Mat3): Mat4 {
    return out.set(
      m.m00, m.m01, m.m02, 0,
      m.m10, m.m11, m.m12, 0,
      m.m20, m.m21, m.m22, 0,
      0, 0, 0, 1,
    );
  }

  /**
   * Affine transform `M = T(translation) · R(rotation)` built from a unit
   * quaternion and a translation vector.
   */
  static fromRotationTranslation(out: Mat4, rotation: Quat, translation: Vec3): Mat4 {
    const x = rotation.x;
    const y = rotation.y;
    const z = rotation.z;
    const w = rotation.w;
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
    out.m03 = translation.x;
    out.m10 = xy + wz;
    out.m11 = 1 - (xx + zz);
    out.m12 = yz - wx;
    out.m13 = translation.y;
    out.m20 = xz - wy;
    out.m21 = yz + wx;
    out.m22 = 1 - (xx + yy);
    out.m23 = translation.z;
    out.m30 = 0;
    out.m31 = 0;
    out.m32 = 0;
    out.m33 = 1;
    return out;
  }

  /** Reads sixteen consecutive row-major scalars starting at `offset`. */
  static fromArray(array: ArrayLike<number>, offset = 0, out = new Mat4()): Mat4 {
    if (offset < 0 || offset + 16 > array.length) {
      throw new RangeError(
        `Mat4.fromArray: reading [${offset}, ${offset + 16}) exceeds length ${array.length}`,
      );
    }
    out.m00 = array[offset];
    out.m01 = array[offset + 1];
    out.m02 = array[offset + 2];
    out.m03 = array[offset + 3];
    out.m10 = array[offset + 4];
    out.m11 = array[offset + 5];
    out.m12 = array[offset + 6];
    out.m13 = array[offset + 7];
    out.m20 = array[offset + 8];
    out.m21 = array[offset + 9];
    out.m22 = array[offset + 10];
    out.m23 = array[offset + 11];
    out.m30 = array[offset + 12];
    out.m31 = array[offset + 13];
    out.m32 = array[offset + 14];
    out.m33 = array[offset + 15];
    return out;
  }

  /** Writes sixteen row-major scalars starting at `offset`. */
  writeTo(array: Float32Array | Float64Array, offset = 0): void {
    if (offset < 0 || offset + 16 > array.length) {
      throw new RangeError(
        `Mat4.writeTo: writing [${offset}, ${offset + 16}) exceeds length ${array.length}`,
      );
    }
    array[offset] = this.m00;
    array[offset + 1] = this.m01;
    array[offset + 2] = this.m02;
    array[offset + 3] = this.m03;
    array[offset + 4] = this.m10;
    array[offset + 5] = this.m11;
    array[offset + 6] = this.m12;
    array[offset + 7] = this.m13;
    array[offset + 8] = this.m20;
    array[offset + 9] = this.m21;
    array[offset + 10] = this.m22;
    array[offset + 11] = this.m23;
    array[offset + 12] = this.m30;
    array[offset + 13] = this.m31;
    array[offset + 14] = this.m32;
    array[offset + 15] = this.m33;
  }

  set(
    m00: number, m01: number, m02: number, m03: number,
    m10: number, m11: number, m12: number, m13: number,
    m20: number, m21: number, m22: number, m23: number,
    m30: number, m31: number, m32: number, m33: number,
  ): this {
    this.m00 = m00;
    this.m01 = m01;
    this.m02 = m02;
    this.m03 = m03;
    this.m10 = m10;
    this.m11 = m11;
    this.m12 = m12;
    this.m13 = m13;
    this.m20 = m20;
    this.m21 = m21;
    this.m22 = m22;
    this.m23 = m23;
    this.m30 = m30;
    this.m31 = m31;
    this.m32 = m32;
    this.m33 = m33;
    return this;
  }

  /** Resets to the identity matrix. */
  identity(): this {
    return this.set(
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    );
  }

  copy(m: Mat4): this {
    return this.set(
      m.m00, m.m01, m.m02, m.m03,
      m.m10, m.m11, m.m12, m.m13,
      m.m20, m.m21, m.m22, m.m23,
      m.m30, m.m31, m.m32, m.m33,
    );
  }

  clone(): Mat4 {
    return new Mat4().copy(this);
  }

  /** `this = this · m` */
  multiply(m: Mat4): this {
    Mat4.multiply(this, this, m);
    return this;
  }

  /** `this = thisᵀ` */
  transpose(): this {
    Mat4.transpose(this, this);
    return this;
  }

  /** `this = this⁻¹`; returns false and zeroes `this` when singular. */
  invert(): boolean {
    return Mat4.invert(this, this);
  }

  /** Determinant `det(this)`. */
  determinant(): number {
    return Mat4.determinant(this);
  }

  /** True when every component is finite. */
  isFinite(): boolean {
    return (
      Number.isFinite(this.m00) &&
      Number.isFinite(this.m01) &&
      Number.isFinite(this.m02) &&
      Number.isFinite(this.m03) &&
      Number.isFinite(this.m10) &&
      Number.isFinite(this.m11) &&
      Number.isFinite(this.m12) &&
      Number.isFinite(this.m13) &&
      Number.isFinite(this.m20) &&
      Number.isFinite(this.m21) &&
      Number.isFinite(this.m22) &&
      Number.isFinite(this.m23) &&
      Number.isFinite(this.m30) &&
      Number.isFinite(this.m31) &&
      Number.isFinite(this.m32) &&
      Number.isFinite(this.m33)
    );
  }

  /** Component-wise equality within `tolerance` (default exact). */
  equals(m: Mat4, tolerance: number = 0): boolean {
    return (
      Math.abs(this.m00 - m.m00) <= tolerance &&
      Math.abs(this.m01 - m.m01) <= tolerance &&
      Math.abs(this.m02 - m.m02) <= tolerance &&
      Math.abs(this.m03 - m.m03) <= tolerance &&
      Math.abs(this.m10 - m.m10) <= tolerance &&
      Math.abs(this.m11 - m.m11) <= tolerance &&
      Math.abs(this.m12 - m.m12) <= tolerance &&
      Math.abs(this.m13 - m.m13) <= tolerance &&
      Math.abs(this.m20 - m.m20) <= tolerance &&
      Math.abs(this.m21 - m.m21) <= tolerance &&
      Math.abs(this.m22 - m.m22) <= tolerance &&
      Math.abs(this.m23 - m.m23) <= tolerance &&
      Math.abs(this.m30 - m.m30) <= tolerance &&
      Math.abs(this.m31 - m.m31) <= tolerance &&
      Math.abs(this.m32 - m.m32) <= tolerance &&
      Math.abs(this.m33 - m.m33) <= tolerance
    );
  }

  toString(): string {
    return (
      `Mat4[${this.m00}, ${this.m01}, ${this.m02}, ${this.m03}; ` +
      `${this.m10}, ${this.m11}, ${this.m12}, ${this.m13}; ` +
      `${this.m20}, ${this.m21}, ${this.m22}, ${this.m23}; ` +
      `${this.m30}, ${this.m31}, ${this.m32}, ${this.m33}]`
    );
  }

  // ------------------------------------------------------------------
  // Static operations (out-parameter, alias-safe)
  // ------------------------------------------------------------------

  /** `out = a · b` (apply `b` first, then `a`). */
  static multiply(out: Mat4, a: Mat4, b: Mat4): Mat4 {
    const a00 = a.m00;
    const a01 = a.m01;
    const a02 = a.m02;
    const a03 = a.m03;
    const a10 = a.m10;
    const a11 = a.m11;
    const a12 = a.m12;
    const a13 = a.m13;
    const a20 = a.m20;
    const a21 = a.m21;
    const a22 = a.m22;
    const a23 = a.m23;
    const a30 = a.m30;
    const a31 = a.m31;
    const a32 = a.m32;
    const a33 = a.m33;
    const b00 = b.m00;
    const b01 = b.m01;
    const b02 = b.m02;
    const b03 = b.m03;
    const b10 = b.m10;
    const b11 = b.m11;
    const b12 = b.m12;
    const b13 = b.m13;
    const b20 = b.m20;
    const b21 = b.m21;
    const b22 = b.m22;
    const b23 = b.m23;
    const b30 = b.m30;
    const b31 = b.m31;
    const b32 = b.m32;
    const b33 = b.m33;

    out.m00 = a00 * b00 + a01 * b10 + a02 * b20 + a03 * b30;
    out.m01 = a00 * b01 + a01 * b11 + a02 * b21 + a03 * b31;
    out.m02 = a00 * b02 + a01 * b12 + a02 * b22 + a03 * b32;
    out.m03 = a00 * b03 + a01 * b13 + a02 * b23 + a03 * b33;
    out.m10 = a10 * b00 + a11 * b10 + a12 * b20 + a13 * b30;
    out.m11 = a10 * b01 + a11 * b11 + a12 * b21 + a13 * b31;
    out.m12 = a10 * b02 + a11 * b12 + a12 * b22 + a13 * b32;
    out.m13 = a10 * b03 + a11 * b13 + a12 * b23 + a13 * b33;
    out.m20 = a20 * b00 + a21 * b10 + a22 * b20 + a23 * b30;
    out.m21 = a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31;
    out.m22 = a20 * b02 + a21 * b12 + a22 * b22 + a23 * b32;
    out.m23 = a20 * b03 + a21 * b13 + a22 * b23 + a23 * b33;
    out.m30 = a30 * b00 + a31 * b10 + a32 * b20 + a33 * b30;
    out.m31 = a30 * b01 + a31 * b11 + a32 * b21 + a33 * b31;
    out.m32 = a30 * b02 + a31 * b12 + a32 * b22 + a33 * b32;
    out.m33 = a30 * b03 + a31 * b13 + a32 * b23 + a33 * b33;
    return out;
  }

  /** `out = aᵀ` */
  static transpose(out: Mat4, a: Mat4): Mat4 {
    const m00 = a.m00;
    const m01 = a.m01;
    const m02 = a.m02;
    const m03 = a.m03;
    const m10 = a.m10;
    const m11 = a.m11;
    const m12 = a.m12;
    const m13 = a.m13;
    const m20 = a.m20;
    const m21 = a.m21;
    const m22 = a.m22;
    const m23 = a.m23;
    const m30 = a.m30;
    const m31 = a.m31;
    const m32 = a.m32;
    const m33 = a.m33;
    out.m00 = m00;
    out.m01 = m10;
    out.m02 = m20;
    out.m03 = m30;
    out.m10 = m01;
    out.m11 = m11;
    out.m12 = m21;
    out.m13 = m31;
    out.m20 = m02;
    out.m21 = m12;
    out.m22 = m22;
    out.m23 = m32;
    out.m30 = m03;
    out.m31 = m13;
    out.m32 = m23;
    out.m33 = m33;
    return out;
  }

  /** Determinant via cofactor expansion along the first row. */
  static determinant(a: Mat4): number {
    const b00 = a.m00 * a.m11 - a.m01 * a.m10;
    const b01 = a.m00 * a.m12 - a.m02 * a.m10;
    const b02 = a.m00 * a.m13 - a.m03 * a.m10;
    const b03 = a.m01 * a.m12 - a.m02 * a.m11;
    const b04 = a.m01 * a.m13 - a.m03 * a.m11;
    const b05 = a.m02 * a.m13 - a.m03 * a.m12;
    const b06 = a.m20 * a.m31 - a.m21 * a.m30;
    const b07 = a.m20 * a.m32 - a.m22 * a.m30;
    const b08 = a.m20 * a.m33 - a.m23 * a.m30;
    const b09 = a.m21 * a.m32 - a.m22 * a.m31;
    const b10 = a.m21 * a.m33 - a.m23 * a.m31;
    const b11 = a.m22 * a.m33 - a.m23 * a.m32;
    return b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  }

  /**
   * `out = a⁻¹` via the classical adjoint (cofactor) construction.
   *
   * @returns `true` on success. When `|det(a)| < EPSILON` the matrix is
   * numerically singular: `out` is zeroed and `false` is returned.
   */
  static invert(out: Mat4, a: Mat4): boolean {
    const a00 = a.m00;
    const a01 = a.m01;
    const a02 = a.m02;
    const a03 = a.m03;
    const a10 = a.m10;
    const a11 = a.m11;
    const a12 = a.m12;
    const a13 = a.m13;
    const a20 = a.m20;
    const a21 = a.m21;
    const a22 = a.m22;
    const a23 = a.m23;
    const a30 = a.m30;
    const a31 = a.m31;
    const a32 = a.m32;
    const a33 = a.m33;

    const b00 = a00 * a11 - a01 * a10;
    const b01 = a00 * a12 - a02 * a10;
    const b02 = a00 * a13 - a03 * a10;
    const b03 = a01 * a12 - a02 * a11;
    const b04 = a01 * a13 - a03 * a11;
    const b05 = a02 * a13 - a03 * a12;
    const b06 = a20 * a31 - a21 * a30;
    const b07 = a20 * a32 - a22 * a30;
    const b08 = a20 * a33 - a23 * a30;
    const b09 = a21 * a32 - a22 * a31;
    const b10 = a21 * a33 - a23 * a31;
    const b11 = a22 * a33 - a23 * a32;

    const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (Math.abs(det) < EPSILON) {
      Mat4.zero(out);
      return false;
    }
    const inverseDet = 1 / det;

    out.m00 = (a11 * b11 - a12 * b10 + a13 * b09) * inverseDet;
    out.m01 = (a02 * b10 - a01 * b11 - a03 * b09) * inverseDet;
    out.m02 = (a31 * b05 - a32 * b04 + a33 * b03) * inverseDet;
    out.m03 = (a22 * b04 - a21 * b05 - a23 * b03) * inverseDet;
    out.m10 = (a12 * b08 - a10 * b11 - a13 * b07) * inverseDet;
    out.m11 = (a00 * b11 - a02 * b08 + a03 * b07) * inverseDet;
    out.m12 = (a32 * b02 - a30 * b05 - a33 * b01) * inverseDet;
    out.m13 = (a20 * b05 - a22 * b02 + a23 * b01) * inverseDet;
    out.m20 = (a10 * b10 - a11 * b08 + a13 * b06) * inverseDet;
    out.m21 = (a01 * b08 - a00 * b10 - a03 * b06) * inverseDet;
    out.m22 = (a30 * b04 - a31 * b02 + a33 * b00) * inverseDet;
    out.m23 = (a21 * b02 - a20 * b04 - a23 * b00) * inverseDet;
    out.m30 = (a11 * b07 - a10 * b09 - a12 * b06) * inverseDet;
    out.m31 = (a00 * b09 - a01 * b07 + a02 * b06) * inverseDet;
    out.m32 = (a31 * b01 - a30 * b03 - a32 * b00) * inverseDet;
    out.m33 = (a20 * b03 - a21 * b01 + a22 * b00) * inverseDet;
    return true;
  }

  /** Zeroes every component of `out`. */
  static zero(out: Mat4): Mat4 {
    return out.set(
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
    );
  }

  /**
   * Transforms a homogeneous **point** (`w = 1`) and performs the perspective
   * divide when `w ≠ 1`.
   */
  static transformPoint(out: Vec3, m: Mat4, p: Vec3): Vec3 {
    const x = p.x;
    const y = p.y;
    const z = p.z;
    const w = m.m30 * x + m.m31 * y + m.m32 * z + m.m33;
    const inverseW = 1 / w;
    out.x = (m.m00 * x + m.m01 * y + m.m02 * z + m.m03) * inverseW;
    out.y = (m.m10 * x + m.m11 * y + m.m12 * z + m.m13) * inverseW;
    out.z = (m.m20 * x + m.m21 * y + m.m22 * z + m.m23) * inverseW;
    return out;
  }

  /**
   * Transforms a homogeneous **direction** (`w = 0`), ignoring translation and
   * any projective row.
   */
  static transformDirection(out: Vec3, m: Mat4, d: Vec3): Vec3 {
    const x = d.x;
    const y = d.y;
    const z = d.z;
    out.x = m.m00 * x + m.m01 * y + m.m02 * z;
    out.y = m.m10 * x + m.m11 * y + m.m12 * z;
    out.z = m.m20 * x + m.m21 * y + m.m22 * z;
    return out;
  }
}
