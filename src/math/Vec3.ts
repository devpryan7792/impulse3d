import { EPSILON, EPSILON_SQ } from "./MathUtils.js";

/**
 * Double-precision 3-component vector.
 *
 * ## Layout & SIMD strategy
 * Components are three contiguous `number` fields. Every operation is written
 * as straight-line scalar math over local variables so that JIT engines can
 * vectorize or hoist it, and so the class can be mechanically translated to
 * `Float64Array`/WASM SIMD storage later (see {@link Vec3.writeTo} and
 * {@link Vec3.fromArray}, which use exactly the same component order).
 *
 * ## Mutation model
 * Instance methods mutate `this` and return `this` (fluent, allocation-free).
 * Static methods write into an explicit `out` vector; all statics are
 * **alias-safe**: `out` may be the same object as any input.
 */
export class Vec3 {
  public x: number;
  public y: number;
  public z: number;

  constructor(x = 0, y = 0, z = 0) {
    this.x = x;
    this.y = y;
    this.z = z;
  }

  /** Returns `[0, 0, 0]`. */
  static zero(): Vec3 {
    return new Vec3(0, 0, 0);
  }

  /** Returns `[x, y, z]`. */
  static fromValues(x: number, y: number, z: number): Vec3 {
    return new Vec3(x, y, z);
  }

  /** Reads three consecutive scalars starting at `offset`. */
  static fromArray(array: ArrayLike<number>, offset = 0, out = new Vec3()): Vec3 {
    if (offset < 0 || offset + 3 > array.length) {
      throw new RangeError(
        `Vec3.fromArray: reading [${offset}, ${offset + 3}) exceeds length ${array.length}`,
      );
    }
    out.x = array[offset];
    out.y = array[offset + 1];
    out.z = array[offset + 2];
    return out;
  }

  /** Writes `[x, y, z]` as three consecutive scalars starting at `offset`. */
  writeTo(array: Float32Array | Float64Array, offset = 0): void {
    if (offset < 0 || offset + 3 > array.length) {
      throw new RangeError(
        `Vec3.writeTo: writing [${offset}, ${offset + 3}) exceeds length ${array.length}`,
      );
    }
    array[offset] = this.x;
    array[offset + 1] = this.y;
    array[offset + 2] = this.z;
  }

  set(x: number, y: number, z: number): this {
    this.x = x;
    this.y = y;
    this.z = z;
    return this;
  }

  setZero(): this {
    this.x = 0;
    this.y = 0;
    this.z = 0;
    return this;
  }

  copy(v: Vec3): this {
    this.x = v.x;
    this.y = v.y;
    this.z = v.z;
    return this;
  }

  clone(): Vec3 {
    return new Vec3(this.x, this.y, this.z);
  }

  // ------------------------------------------------------------------
  // Instance arithmetic (mutating, fluent)
  // ------------------------------------------------------------------

  /** `this += v` */
  add(v: Vec3): this {
    this.x += v.x;
    this.y += v.y;
    this.z += v.z;
    return this;
  }

  /** `this -= v` */
  sub(v: Vec3): this {
    this.x -= v.x;
    this.y -= v.y;
    this.z -= v.z;
    return this;
  }

  /** `this *= s` */
  scale(s: number): this {
    this.x *= s;
    this.y *= s;
    this.z *= s;
    return this;
  }

  /** `this += v * s` (fused multiply-accumulate form). */
  addScaled(v: Vec3, s: number): this {
    this.x += v.x * s;
    this.y += v.y * s;
    this.z += v.z * s;
    return this;
  }

  /** `this = -this` */
  negate(): this {
    this.x = -this.x;
    this.y = -this.y;
    this.z = -this.z;
    return this;
  }

  /** `this = this x v` (right-handed cross product). */
  cross(v: Vec3): this {
    Vec3.cross(this, this, v);
    return this;
  }

  /**
   * `this /= |this|`. If `|this|² < EPSILON_SQ` the vector is replaced by
   * `[0, 0, 0]` (a zero vector has no direction).
   */
  normalize(): this {
    Vec3.normalize(this, this);
    return this;
  }

  // ------------------------------------------------------------------
  // Queries
  // ------------------------------------------------------------------

  /** Inner product `this . v`. */
  dot(v: Vec3): number {
    return this.x * v.x + this.y * v.y + this.z * v.z;
  }

  /** Squared Euclidean norm. */
  lengthSq(): number {
    return this.x * this.x + this.y * this.y + this.z * this.z;
  }

  /** Euclidean norm. */
  length(): number {
    return Math.sqrt(this.lengthSq());
  }

  /** Euclidean distance to `v`. */
  distanceTo(v: Vec3): number {
    return Vec3.distance(this, v);
  }

  /** True when every component is finite (not `NaN`, not `±Infinity`). */
  isFinite(): boolean {
    return Number.isFinite(this.x) && Number.isFinite(this.y) && Number.isFinite(this.z);
  }

  /** Component-wise equality within `tolerance`. */
  equals(v: Vec3, tolerance: number = 0): boolean {
    return (
      Math.abs(this.x - v.x) <= tolerance &&
      Math.abs(this.y - v.y) <= tolerance &&
      Math.abs(this.z - v.z) <= tolerance
    );
  }

  toString(): string {
    return `Vec3(${this.x}, ${this.y}, ${this.z})`;
  }

  // ------------------------------------------------------------------
  // Static arithmetic (out-parameter, alias-safe)
  // ------------------------------------------------------------------

  /** `out = a + b` */
  static add(out: Vec3, a: Vec3, b: Vec3): Vec3 {
    const x = a.x + b.x;
    const y = a.y + b.y;
    const z = a.z + b.z;
    out.x = x;
    out.y = y;
    out.z = z;
    return out;
  }

  /** `out = a - b` */
  static sub(out: Vec3, a: Vec3, b: Vec3): Vec3 {
    const x = a.x - b.x;
    const y = a.y - b.y;
    const z = a.z - b.z;
    out.x = x;
    out.y = y;
    out.z = z;
    return out;
  }

  /** `out = a * s` */
  static scale(out: Vec3, a: Vec3, s: number): Vec3 {
    const x = a.x * s;
    const y = a.y * s;
    const z = a.z * s;
    out.x = x;
    out.y = y;
    out.z = z;
    return out;
  }

  /** `out = a + b * s` (fused multiply-accumulate form). */
  static addScaled(out: Vec3, a: Vec3, b: Vec3, s: number): Vec3 {
    const x = a.x + b.x * s;
    const y = a.y + b.y * s;
    const z = a.z + b.z * s;
    out.x = x;
    out.y = y;
    out.z = z;
    return out;
  }

  /** `out = -a` */
  static negate(out: Vec3, a: Vec3): Vec3 {
    return Vec3.scale(out, a, -1);
  }

  /** Inner product `a . b`. */
  static dot(a: Vec3, b: Vec3): number {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }

  /**
   * `out = a x b` using the right-handed convention
   * `x̂ x ŷ = ẑ`, `ŷ x ẑ = x̂`, `ẑ x x̂ = ŷ`.
   */
  static cross(out: Vec3, a: Vec3, b: Vec3): Vec3 {
    const x = a.y * b.z - a.z * b.y;
    const y = a.z * b.x - a.x * b.z;
    const z = a.x * b.y - a.y * b.x;
    out.x = x;
    out.y = y;
    out.z = z;
    return out;
  }

  /**
   * Vector triple product `out = a x (b x c)`.
   *
   * Evaluated through the Lagrange identity `a x (b x c) = b(a . c) - c(a . b)`
   * which is both cheaper and numerically tighter than two chained crosses.
   */
  static tripleProduct(out: Vec3, a: Vec3, b: Vec3, c: Vec3): Vec3 {
    const ac = Vec3.dot(a, c);
    const ab = Vec3.dot(a, b);
    const x = b.x * ac - c.x * ab;
    const y = b.y * ac - c.y * ab;
    const z = b.z * ac - c.z * ab;
    out.x = x;
    out.y = y;
    out.z = z;
    return out;
  }

  /** `out = a + t * (b - a)`, unclamped for extrapolation. */
  static lerp(out: Vec3, a: Vec3, b: Vec3, t: number): Vec3 {
    const x = a.x + t * (b.x - a.x);
    const y = a.y + t * (b.y - a.y);
    const z = a.z + t * (b.z - a.z);
    out.x = x;
    out.y = y;
    out.z = z;
    return out;
  }

  /** Euclidean distance between `a` and `b`. */
  static distance(a: Vec3, b: Vec3): number {
    return Math.sqrt(Vec3.distanceSq(a, b));
  }

  /** Squared Euclidean distance between `a` and `b`. */
  static distanceSq(a: Vec3, b: Vec3): number {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    const dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
  }

  /**
   * `out = a / |a|`.
   *
   * When `|a|² < EPSILON_SQ`, `fallback` is copied into `out` if provided,
   * otherwise `out` becomes the zero vector. Support mappings use this to stay
   * total on degenerate query directions.
   */
  static normalize(out: Vec3, a: Vec3, fallback: Vec3 | null = null): Vec3 {
    const lengthSq = a.x * a.x + a.y * a.y + a.z * a.z;
    if (lengthSq < EPSILON_SQ) {
      if (fallback === null) {
        return out.setZero();
      }
      return out.copy(fallback);
    }
    const inverseLength = 1 / Math.sqrt(lengthSq);
    return Vec3.scale(out, a, inverseLength);
  }

  /**
   * Mirror reflection of `v` about the plane through the origin with normal
   * `n`: `out = v - 2 * (v . n) * n`. `n` is assumed to be unit length.
   */
  static reflect(out: Vec3, v: Vec3, n: Vec3): Vec3 {
    const scale = 2 * Vec3.dot(v, n);
    const x = v.x - scale * n.x;
    const y = v.y - scale * n.y;
    const z = v.z - scale * n.z;
    out.x = x;
    out.y = y;
    out.z = z;
    return out;
  }

  /** Component-wise minimum. */
  static min(out: Vec3, a: Vec3, b: Vec3): Vec3 {
    const x = a.x < b.x ? a.x : b.x;
    const y = a.y < b.y ? a.y : b.y;
    const z = a.z < b.z ? a.z : b.z;
    out.x = x;
    out.y = y;
    out.z = z;
    return out;
  }

  /** Component-wise maximum. */
  static max(out: Vec3, a: Vec3, b: Vec3): Vec3 {
    const x = a.x > b.x ? a.x : b.x;
    const y = a.y > b.y ? a.y : b.y;
    const z = a.z > b.z ? a.z : b.z;
    out.x = x;
    out.y = y;
    out.z = z;
    return out;
  }

  /** True when every component of `v` is finite. */
  static isFinite(v: Vec3): boolean {
    return Number.isFinite(v.x) && Number.isFinite(v.y) && Number.isFinite(v.z);
  }

  /** Component-wise equality within `tolerance` (default `EPSILON`). */
  static equals(a: Vec3, b: Vec3, tolerance: number = EPSILON): boolean {
    return a.equals(b, tolerance);
  }
}
