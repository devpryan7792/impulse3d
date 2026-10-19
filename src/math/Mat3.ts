import { EPSILON } from "./MathUtils.js";
import { Vec3 } from "./Vec3.js";

/**
 * Double-precision 3x3 matrix stored as nine contiguous scalars in
 * **row-major** order (`m00 m01 m02 / m10 m11 m12 / m20 m21 m22`).
 *
 * Matrices act on column vectors (`out = M v`), so composition reads
 * left-to-right: `Mat3.multiply(out, a, b)` represents "apply `b`, then `a`".
 *
 * The class provides the linear-algebra primitives required by rigid body
 * dynamics: the skew-symmetric cross-product operator `[r]×`
 * ({@link Mat3.skew}), rotation matrices, and world-space inertia tensor
 * transformation `R I Rᵀ` ({@link Mat3.transformInertia}).
 *
 * Instance methods mutate `this`; static methods write into an explicit `out`
 * matrix and are alias-safe (`out` may be any input).
 */
export class Mat3 {
  public m00: number;
  public m01: number;
  public m02: number;
  public m10: number;
  public m11: number;
  public m12: number;
  public m20: number;
  public m21: number;
  public m22: number;

  /** Constructs the identity matrix. */
  constructor() {
    this.m00 = 1;
    this.m01 = 0;
    this.m02 = 0;
    this.m10 = 0;
    this.m11 = 1;
    this.m12 = 0;
    this.m20 = 0;
    this.m21 = 0;
    this.m22 = 1;
  }

  /** Returns `[m00 ... m22]` in row-major order, unmodified. */
  static fromValues(
    m00: number,
    m01: number,
    m02: number,
    m10: number,
    m11: number,
    m12: number,
    m20: number,
    m21: number,
    m22: number,
  ): Mat3 {
    return new Mat3().set(m00, m01, m02, m10, m11, m12, m20, m21, m22);
  }

  /** Returns `diag(x, y, z)`, e.g. a body-space diagonal inertia tensor. */
  static fromDiagonal(x: number, y: number, z: number): Mat3 {
    return new Mat3().set(x, 0, 0, 0, y, 0, 0, 0, z);
  }

  /** Reads nine consecutive row-major scalars starting at `offset`. */
  static fromArray(array: ArrayLike<number>, offset = 0, out = new Mat3()): Mat3 {
    if (offset < 0 || offset + 9 > array.length) {
      throw new RangeError(
        `Mat3.fromArray: reading [${offset}, ${offset + 9}) exceeds length ${array.length}`,
      );
    }
    out.m00 = array[offset];
    out.m01 = array[offset + 1];
    out.m02 = array[offset + 2];
    out.m10 = array[offset + 3];
    out.m11 = array[offset + 4];
    out.m12 = array[offset + 5];
    out.m20 = array[offset + 6];
    out.m21 = array[offset + 7];
    out.m22 = array[offset + 8];
    return out;
  }

  /** Writes nine row-major scalars starting at `offset`. */
  writeTo(array: Float32Array | Float64Array, offset = 0): void {
    if (offset < 0 || offset + 9 > array.length) {
      throw new RangeError(
        `Mat3.writeTo: writing [${offset}, ${offset + 9}) exceeds length ${array.length}`,
      );
    }
    array[offset] = this.m00;
    array[offset + 1] = this.m01;
    array[offset + 2] = this.m02;
    array[offset + 3] = this.m10;
    array[offset + 4] = this.m11;
    array[offset + 5] = this.m12;
    array[offset + 6] = this.m20;
    array[offset + 7] = this.m21;
    array[offset + 8] = this.m22;
  }

  set(
    m00: number,
    m01: number,
    m02: number,
    m10: number,
    m11: number,
    m12: number,
    m20: number,
    m21: number,
    m22: number,
  ): this {
    this.m00 = m00;
    this.m01 = m01;
    this.m02 = m02;
    this.m10 = m10;
    this.m11 = m11;
    this.m12 = m12;
    this.m20 = m20;
    this.m21 = m21;
    this.m22 = m22;
    return this;
  }

  /** Resets to the identity matrix. */
  identity(): this {
    return this.set(1, 0, 0, 0, 1, 0, 0, 0, 1);
  }

  copy(m: Mat3): this {
    return this.set(m.m00, m.m01, m.m02, m.m10, m.m11, m.m12, m.m20, m.m21, m.m22);
  }

  clone(): Mat3 {
    return new Mat3().copy(this);
  }

  /** `this = this · m` */
  multiply(m: Mat3): this {
    Mat3.multiply(this, this, m);
    return this;
  }

  /** `this = thisᵀ` */
  transpose(): this {
    Mat3.transpose(this, this);
    return this;
  }

  /** `this = this⁻¹`; returns false and zeroes `this` when singular. */
  invert(): boolean {
    return Mat3.invert(this, this);
  }

  /** `this *= s` */
  multiplyScalar(s: number): this {
    Mat3.multiplyScalar(this, this, s);
    return this;
  }

  /** Determinant `det(this)`. */
  determinant(): number {
    return Mat3.determinant(this);
  }

  /** True when every component is finite. */
  isFinite(): boolean {
    return (
      Number.isFinite(this.m00) &&
      Number.isFinite(this.m01) &&
      Number.isFinite(this.m02) &&
      Number.isFinite(this.m10) &&
      Number.isFinite(this.m11) &&
      Number.isFinite(this.m12) &&
      Number.isFinite(this.m20) &&
      Number.isFinite(this.m21) &&
      Number.isFinite(this.m22)
    );
  }

  /** Component-wise equality within `tolerance` (default exact). */
  equals(m: Mat3, tolerance: number = 0): boolean {
    return (
      Math.abs(this.m00 - m.m00) <= tolerance &&
      Math.abs(this.m01 - m.m01) <= tolerance &&
      Math.abs(this.m02 - m.m02) <= tolerance &&
      Math.abs(this.m10 - m.m10) <= tolerance &&
      Math.abs(this.m11 - m.m11) <= tolerance &&
      Math.abs(this.m12 - m.m12) <= tolerance &&
      Math.abs(this.m20 - m.m20) <= tolerance &&
      Math.abs(this.m21 - m.m21) <= tolerance &&
      Math.abs(this.m22 - m.m22) <= tolerance
    );
  }

  toString(): string {
    return (
      `Mat3[${this.m00}, ${this.m01}, ${this.m02}; ` +
      `${this.m10}, ${this.m11}, ${this.m12}; ` +
      `${this.m20}, ${this.m21}, ${this.m22}]`
    );
  }

  // ------------------------------------------------------------------
  // Static operations (out-parameter, alias-safe)
  // ------------------------------------------------------------------

  /** `out = a · b` (apply `b` first, then `a`). */
  static multiply(out: Mat3, a: Mat3, b: Mat3): Mat3 {
    const a00 = a.m00;
    const a01 = a.m01;
    const a02 = a.m02;
    const a10 = a.m10;
    const a11 = a.m11;
    const a12 = a.m12;
    const a20 = a.m20;
    const a21 = a.m21;
    const a22 = a.m22;
    const b00 = b.m00;
    const b01 = b.m01;
    const b02 = b.m02;
    const b10 = b.m10;
    const b11 = b.m11;
    const b12 = b.m12;
    const b20 = b.m20;
    const b21 = b.m21;
    const b22 = b.m22;

    out.m00 = a00 * b00 + a01 * b10 + a02 * b20;
    out.m01 = a00 * b01 + a01 * b11 + a02 * b21;
    out.m02 = a00 * b02 + a01 * b12 + a02 * b22;
    out.m10 = a10 * b00 + a11 * b10 + a12 * b20;
    out.m11 = a10 * b01 + a11 * b11 + a12 * b21;
    out.m12 = a10 * b02 + a11 * b12 + a12 * b22;
    out.m20 = a20 * b00 + a21 * b10 + a22 * b20;
    out.m21 = a20 * b01 + a21 * b11 + a22 * b21;
    out.m22 = a20 * b02 + a21 * b12 + a22 * b22;
    return out;
  }

  /** `out = aᵀ` */
  static transpose(out: Mat3, a: Mat3): Mat3 {
    const m00 = a.m00;
    const m01 = a.m01;
    const m02 = a.m02;
    const m10 = a.m10;
    const m11 = a.m11;
    const m12 = a.m12;
    const m20 = a.m20;
    const m21 = a.m21;
    const m22 = a.m22;
    out.m00 = m00;
    out.m01 = m10;
    out.m02 = m20;
    out.m10 = m01;
    out.m11 = m11;
    out.m12 = m21;
    out.m20 = m02;
    out.m21 = m12;
    out.m22 = m22;
    return out;
  }

  /** `out = a * s` */
  static multiplyScalar(out: Mat3, a: Mat3, s: number): Mat3 {
    if (out !== a) {
      out.copy(a);
    }
    out.m00 *= s;
    out.m01 *= s;
    out.m02 *= s;
    out.m10 *= s;
    out.m11 *= s;
    out.m12 *= s;
    out.m20 *= s;
    out.m21 *= s;
    out.m22 *= s;
    return out;
  }

  /** Determinant via cofactor expansion along the first row. */
  static determinant(a: Mat3): number {
    return (
      a.m00 * (a.m11 * a.m22 - a.m12 * a.m21) -
      a.m01 * (a.m10 * a.m22 - a.m12 * a.m20) +
      a.m02 * (a.m10 * a.m21 - a.m11 * a.m20)
    );
  }

  /**
   * `out = a⁻¹`.
   *
   * @returns `true` on success. When `|det(a)| < EPSILON` the matrix is
   * numerically singular: `out` is zeroed and `false` is returned.
   */
  static invert(out: Mat3, a: Mat3): boolean {
    const m00 = a.m00;
    const m01 = a.m01;
    const m02 = a.m02;
    const m10 = a.m10;
    const m11 = a.m11;
    const m12 = a.m12;
    const m20 = a.m20;
    const m21 = a.m21;
    const m22 = a.m22;

    const c00 = m11 * m22 - m12 * m21;
    const c01 = m12 * m20 - m10 * m22;
    const c02 = m10 * m21 - m11 * m20;
    const det = m00 * c00 + m01 * c01 + m02 * c02;
    if (Math.abs(det) < EPSILON) {
      Mat3.zero(out);
      return false;
    }
    const inverseDet = 1 / det;
    out.m00 = c00 * inverseDet;
    out.m01 = (m02 * m21 - m01 * m22) * inverseDet;
    out.m02 = (m01 * m12 - m02 * m11) * inverseDet;
    out.m10 = c01 * inverseDet;
    out.m11 = (m00 * m22 - m02 * m20) * inverseDet;
    out.m12 = (m02 * m10 - m00 * m12) * inverseDet;
    out.m20 = c02 * inverseDet;
    out.m21 = (m01 * m20 - m00 * m21) * inverseDet;
    out.m22 = (m00 * m11 - m01 * m10) * inverseDet;
    return true;
  }

  /** Zeroes every component of `out`. */
  static zero(out: Mat3): Mat3 {
    return out.set(0, 0, 0, 0, 0, 0, 0, 0, 0);
  }

  /**
   * `out = M v` where `v` is a column vector.
   */
  static transformVector(out: Vec3, m: Mat3, v: Vec3): Vec3 {
    const x = m.m00 * v.x + m.m01 * v.y + m.m02 * v.z;
    const y = m.m10 * v.x + m.m11 * v.y + m.m12 * v.z;
    const z = m.m20 * v.x + m.m21 * v.y + m.m22 * v.z;
    out.x = x;
    out.y = y;
    out.z = z;
    return out;
  }

  /**
   * `out = Mᵀ v`. For a rotation matrix `R` this is the world-to-local map,
   * used by support mappings (`d_local = Rᵀ d`).
   */
  static transformVectorTranspose(out: Vec3, m: Mat3, v: Vec3): Vec3 {
    const x = m.m00 * v.x + m.m10 * v.y + m.m20 * v.z;
    const y = m.m01 * v.x + m.m11 * v.y + m.m21 * v.z;
    const z = m.m02 * v.x + m.m12 * v.y + m.m22 * v.z;
    out.x = x;
    out.y = y;
    out.z = z;
    return out;
  }

  /**
   * Skew-symmetric cross-product operator `[r]×` such that
   * `[r]× v = r x v` for every `v`:
   *
   * ```
   * [ 0   -rz   ry ]
   * [ rz   0   -rx ]
   * [-ry   rx   0  ]
   * ```
   *
   * This is the Jacobian block used by the sequential impulse solver
   * (`J = [-I | [r]× | I | -[r]×]`).
   */
  static skew(out: Mat3, r: Vec3): Mat3 {
    return out.set(0, -r.z, r.y, r.z, 0, -r.x, -r.y, r.x, 0);
  }

  /**
   * Rotates an inertia tensor into the world frame:
   * `out = R · I_body · Rᵀ`.
   *
   * The result stays symmetric positive-definite whenever `rotation` is
   * orthonormal and `inertia` is positive-definite.
   */
  static transformInertia(out: Mat3, rotation: Mat3, inertia: Mat3): Mat3 {
    const r00 = rotation.m00;
    const r01 = rotation.m01;
    const r02 = rotation.m02;
    const r10 = rotation.m10;
    const r11 = rotation.m11;
    const r12 = rotation.m12;
    const r20 = rotation.m20;
    const r21 = rotation.m21;
    const r22 = rotation.m22;
    const i00 = inertia.m00;
    const i01 = inertia.m01;
    const i02 = inertia.m02;
    const i10 = inertia.m10;
    const i11 = inertia.m11;
    const i12 = inertia.m12;
    const i20 = inertia.m20;
    const i21 = inertia.m21;
    const i22 = inertia.m22;

    // t = R · I
    const t00 = r00 * i00 + r01 * i10 + r02 * i20;
    const t01 = r00 * i01 + r01 * i11 + r02 * i21;
    const t02 = r00 * i02 + r01 * i12 + r02 * i22;
    const t10 = r10 * i00 + r11 * i10 + r12 * i20;
    const t11 = r10 * i01 + r11 * i11 + r12 * i21;
    const t12 = r10 * i02 + r11 * i12 + r12 * i22;
    const t20 = r20 * i00 + r21 * i10 + r22 * i20;
    const t21 = r20 * i01 + r21 * i11 + r22 * i21;
    const t22 = r20 * i02 + r21 * i12 + r22 * i22;

    // out = t · Rᵀ
    out.m00 = t00 * r00 + t01 * r01 + t02 * r02;
    out.m01 = t00 * r10 + t01 * r11 + t02 * r12;
    out.m02 = t00 * r20 + t01 * r21 + t02 * r22;
    out.m10 = t10 * r00 + t11 * r01 + t12 * r02;
    out.m11 = t10 * r10 + t11 * r11 + t12 * r12;
    out.m12 = t10 * r20 + t11 * r21 + t12 * r22;
    out.m20 = t20 * r00 + t21 * r01 + t22 * r02;
    out.m21 = t20 * r10 + t21 * r11 + t22 * r12;
    out.m22 = t20 * r20 + t21 * r21 + t22 * r22;
    return out;
  }
}
