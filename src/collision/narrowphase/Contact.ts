import { Vec3 } from "../../math/Vec3.js";

/**
 * One point of a contact manifold.
 *
 * `position` is the world-space contact position (the midpoint of the witness
 * points on the two shapes is used by the narrowphase), `penetration` is the
 * non-negative overlap depth along the manifold normal, and `id` is a stable
 * feature hash used to match cached impulses for warm starting.
 */
export interface ContactPoint {
  position: Vec3;
  penetration: number;
  id: number;
}

/**
 * A contact manifold between two convex shapes.
 *
 * `normal` is a unit vector pointing **from shape A towards shape B**: applying
 * a positive impulse to B along `normal` (and the opposite impulse to A)
 * separates the pair. `points` contains at most four world-space contacts.
 */
export interface ContactManifold {
  normal: Vec3;
  points: ContactPoint[];
}

/** Clones a contact point so manifolds never share mutable state. */
export function copyContactPoint(point: ContactPoint): ContactPoint {
  return {
    position: point.position.clone(),
    penetration: point.penetration,
    id: point.id,
  };
}

/** Returns a manifold with the normal reversed, i.e. swapped A/B roles. */
export function flipManifold(manifold: ContactManifold): ContactManifold {
  return {
    normal: manifold.normal.clone().negate(),
    points: manifold.points.map(copyContactPoint),
  };
}

/** Constructs a contact point from an owned position. */
export function makeContactPoint(position: Vec3, penetration: number, id: number): ContactPoint {
  return { position, penetration, id };
}

/** Stable 32-bit-ish feature hash for a pair of feature ids. */
export function hashContactId(a: number, b: number): number {
  let hash = (a * 0x1f1f1f1f) ^ (b * 0x9e3779b1);
  hash = Math.imul(hash ^ (hash >>> 16), 0x85ebca6b);
  hash = Math.imul(hash ^ (hash >>> 13), 0xc2b2ae35);
  return (hash ^ (hash >>> 16)) >>> 0;
}
