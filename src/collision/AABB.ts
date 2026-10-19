import { Vec3 } from "../math/Vec3.js";

/**
 * Axis-aligned bounding box defined by two opposite corners `min <= max`.
 *
 * Overlap predicates are **inclusive** (`min == max` and face-to-face
 * touching count as overlapping): the broadphase must never drop a pair that
 * could still produce a zero-depth contact, it must only ever be conservative.
 *
 * An AABB is *empty* when any `max` component lies below its `min` component;
 * empty boxes overlap nothing and have zero surface area. Non-finite bounds
 * (`±Infinity`) denote unbounded regions (e.g. an infinite `Plane`) and are
 * supported by every predicate here.
 */
export class AABB {
  public readonly min: Vec3;
  public readonly max: Vec3;

  constructor(min?: Vec3, max?: Vec3) {
    this.min = min === undefined ? new Vec3(0, 0, 0) : min.clone();
    this.max = max === undefined ? new Vec3(0, 0, 0) : max.clone();
  }

  /** Copies the corners of `aabb` into a new box. */
  static from(aabb: AABB): AABB {
    return aabb.clone();
  }

  /**
   * Union of two boxes. `out` may alias either input.
   * An empty input is ignored (identity element of the union).
   */
  static merge(out: AABB, a: AABB, b: AABB): AABB {
    if (a.isEmpty()) {
      return out.copy(b);
    }
    if (b.isEmpty()) {
      return out.copy(a);
    }
    out.min.x = a.min.x < b.min.x ? a.min.x : b.min.x;
    out.min.y = a.min.y < b.min.y ? a.min.y : b.min.y;
    out.min.z = a.min.z < b.min.z ? a.min.z : b.min.z;
    out.max.x = a.max.x > b.max.x ? a.max.x : b.max.x;
    out.max.y = a.max.y > b.max.y ? a.max.y : b.max.y;
    out.max.z = a.max.z > b.max.z ? a.max.z : b.max.z;
    return out;
  }

  /**
   * Velocity-fattened broadphase envelope of a tight box (Section 4.1):
   *
   * ```
   * AABB_fat = AABB_tight + v * dt * alpha + beta
   * ```
   *
   * The velocity term expands the box **along the motion direction** (leading
   * edge only), while the static margin `beta` grows it symmetrically on every
   * side so that slow bodies never force a tree update. `out` may alias
   * `tight`.
   */
  static fatten(
    out: AABB,
    tight: AABB,
    velocity: Vec3,
    dt: number,
    alpha: number,
    beta: number,
  ): AABB {
    const travelX = velocity.x * dt * alpha;
    const travelY = velocity.y * dt * alpha;
    const travelZ = velocity.z * dt * alpha;

    out.min.x = tight.min.x - beta + (travelX < 0 ? travelX : 0);
    out.min.y = tight.min.y - beta + (travelY < 0 ? travelY : 0);
    out.min.z = tight.min.z - beta + (travelZ < 0 ? travelZ : 0);
    out.max.x = tight.max.x + beta + (travelX > 0 ? travelX : 0);
    out.max.y = tight.max.y + beta + (travelY > 0 ? travelY : 0);
    out.max.z = tight.max.z + beta + (travelZ > 0 ? travelZ : 0);
    return out;
  }

  /** Copies `min` and `max` from the given corners. */
  set(min: Vec3, max: Vec3): this {
    this.min.copy(min);
    this.max.copy(max);
    return this;
  }

  /** Rebuilds the box from a center and non-negative half extents. */
  setFromCenterAndHalfExtents(center: Vec3, halfExtents: Vec3): this {
    this.min.set(
      center.x - halfExtents.x,
      center.y - halfExtents.y,
      center.z - halfExtents.z,
    );
    this.max.set(
      center.x + halfExtents.x,
      center.y + halfExtents.y,
      center.z + halfExtents.z,
    );
    return this;
  }

  copy(aabb: AABB): this {
    this.min.copy(aabb.min);
    this.max.copy(aabb.max);
    return this;
  }

  clone(): AABB {
    return new AABB(this.min, this.max);
  }

  /** True when some `max` component is strictly below its `min` component. */
  isEmpty(): boolean {
    return this.max.x < this.min.x || this.max.y < this.min.y || this.max.z < this.min.z;
  }

  /** True when all six bounds are finite. */
  isFinite(): boolean {
    return this.min.isFinite() && this.max.isFinite();
  }

  /** True when any of the six bounds is `NaN` (`Math.max` propagates `NaN`). */
  hasNaN(): boolean {
    return Number.isNaN(
      Math.max(this.min.x, this.min.y, this.min.z, this.max.x, this.max.y, this.max.z),
    );
  }

  /** Writes the box center into `out`. */
  getCenter(out = new Vec3()): Vec3 {
    out.set(
      0.5 * (this.min.x + this.max.x),
      0.5 * (this.min.y + this.max.y),
      0.5 * (this.min.z + this.max.z),
    );
    return out;
  }

  /** Writes the (possibly negative for empty boxes) half extents into `out`. */
  getHalfExtents(out = new Vec3()): Vec3 {
    out.set(
      0.5 * (this.max.x - this.min.x),
      0.5 * (this.max.y - this.min.y),
      0.5 * (this.max.z - this.min.z),
    );
    return out;
  }

  /**
   * Surface area `2(hx·hy + hy·hz + hz·hx)`, the SAH cost metric of the
   * dynamic tree. Returns `0` for empty boxes.
   */
  surfaceArea(): number {
    if (this.isEmpty()) {
      return 0;
    }
    const hx = this.max.x - this.min.x;
    const hy = this.max.y - this.min.y;
    const hz = this.max.z - this.min.z;
    return 2 * (hx * hy + hy * hz + hz * hx);
  }

  /** Inclusive overlap test, used as the broadphase pair filter. Empty boxes overlap nothing. */
  overlaps(aabb: AABB): boolean {
    if (this.isEmpty() || aabb.isEmpty()) {
      return false;
    }
    return (
      this.min.x <= aabb.max.x &&
      aabb.min.x <= this.max.x &&
      this.min.y <= aabb.max.y &&
      aabb.min.y <= this.max.y &&
      this.min.z <= aabb.max.z &&
      aabb.min.z <= this.max.z
    );
  }

  /** True when `aabb` lies entirely inside this box. */
  contains(aabb: AABB): boolean {
    return (
      aabb.min.x >= this.min.x &&
      aabb.min.y >= this.min.y &&
      aabb.min.z >= this.min.z &&
      aabb.max.x <= this.max.x &&
      aabb.max.y <= this.max.y &&
      aabb.max.z <= this.max.z
    );
  }

  /** True when `point` lies inside the closed box. */
  containsPoint(point: Vec3): boolean {
    return (
      point.x >= this.min.x &&
      point.x <= this.max.x &&
      point.y >= this.min.y &&
      point.y <= this.max.y &&
      point.z >= this.min.z &&
      point.z <= this.max.z
    );
  }

  /** Grows the box by `margin` on every side. Negative margins shrink it. */
  expand(margin: number): this {
    this.min.x -= margin;
    this.min.y -= margin;
    this.min.z -= margin;
    this.max.x += margin;
    this.max.y += margin;
    this.max.z += margin;
    return this;
  }

  /** Grows the box by `extents` symmetrically on every axis. */
  expandBy(extents: Vec3): this {
    this.min.x -= extents.x;
    this.min.y -= extents.y;
    this.min.z -= extents.z;
    this.max.x += extents.x;
    this.max.y += extents.y;
    this.max.z += extents.z;
    return this;
  }

  /** Grows this box to include `aabb`. */
  merge(aabb: AABB): this {
    AABB.merge(this, this, aabb);
    return this;
  }

  /** Component-wise equality within `tolerance` (default exact). */
  equals(aabb: AABB, tolerance: number = 0): boolean {
    return this.min.equals(aabb.min, tolerance) && this.max.equals(aabb.max, tolerance);
  }

  toString(): string {
    return `AABB(min=${this.min.toString()}, max=${this.max.toString()})`;
  }
}
