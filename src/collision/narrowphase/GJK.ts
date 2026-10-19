import { EPSILON_SQ } from "../../math/MathUtils.js";
import type { Transform } from "../../math/Transform.js";
import { Vec3 } from "../../math/Vec3.js";
import type { Shape } from "../shapes/Shape.js";

/**
 * A vertex of the Minkowski difference polytope `A ⊖ B`.
 *
 * `point = witnessA - witnessB`, where `witnessA` lies on shape A and
 * `witnessB` lies on shape B. Keeping the witnesses lets EPA reconstruct the
 * contact positions by barycentric interpolation.
 */
export interface SupportVertex {
  point: Vec3;
  witnessA: Vec3;
  witnessB: Vec3;
}

/** Result of a GJK query. `simplex` contains the enclosing tetrahedron on a hit. */
export interface GJKResult {
  intersects: boolean;
  simplex: SupportVertex[];
}

/**
 * Support mapping of the Minkowski difference (Section 5):
 *
 * ```
 * S_{A⊖B}(d) = S_A(d) - S_B(-d)
 * ```
 *
 * The returned vertex owns its vectors; the caller may retain it across
 * iterations.
 */
export function minkowskiSupport(
  shapeA: Shape,
  transformA: Transform,
  shapeB: Shape,
  transformB: Transform,
  direction: Vec3,
): SupportVertex {
  const witnessA = shapeA.getSupport(direction, transformA);
  const witnessB = shapeB.getSupport(
    new Vec3(-direction.x, -direction.y, -direction.z),
    transformB,
  );
  return {
    point: witnessA.clone().sub(witnessB),
    witnessA,
    witnessB,
  };
}

/**
 * GJK intersection test using the classic line/triangle/tetrahedron simplex
 * solvers (Section 5.1).
 *
 * The return value does not distinguish touching from deep penetration: both
 * report `intersects: true`. When the origin is enclosed, the four vertices of
 * the tetrahedron are returned for {@link epa}.
 *
 * The implementation is allocation-per-iteration rather than scratch-pooled:
 * callers that need allocation-free narrowphase can pool at a higher level.
 */
export function gjkIntersects(
  shapeA: Shape,
  transformA: Transform,
  shapeB: Shape,
  transformB: Transform,
): GJKResult {
  const simplex: SupportVertex[] = [];
  const direction = transformB.position.clone().sub(transformA.position);
  if (direction.lengthSq() < EPSILON_SQ) {
    direction.set(1, 0, 0);
  }

  const first = minkowskiSupport(shapeA, transformA, shapeB, transformB, direction);
  simplex.push(first);
  direction.copy(first.point).negate();

  for (;;) {
    if (direction.lengthSq() < EPSILON_SQ) {
      return { intersects: true, simplex };
    }

    const support = minkowskiSupport(shapeA, transformA, shapeB, transformB, direction);
    if (support.point.dot(direction) < 0) {
      return { intersects: false, simplex };
    }

    // A repeated support point means the iteration cannot make progress; the
    // origin cannot be strictly enclosed by a non-degenerate simplex. This is a
    // boundary-contact guard: with a strictly convex support mapping the search
    // direction always makes the previous support points non-extremal, so the
    // branch is defensive and not reachable by the public GJK loop.
    /* v8 ignore start */
    for (const vertex of simplex) {
      if (Vec3.distanceSq(vertex.point, support.point) < EPSILON_SQ) {
        return { intersects: false, simplex };
      }
    }
    /* v8 ignore stop */

    simplex.push(support);
    if (doSimplex(simplex, direction)) {
      return { intersects: true, simplex };
    }
  }
}

/**
 * Advances the simplex towards the origin. Returns `true` when a tetrahedron
 * encloses the origin. Mutates `simplex` (keeping only the supporting feature)
 * and `direction` (the next search direction).
 *
 * @internal Exposed so the simplex solvers can be unit-tested with crafted
 * geometric configurations that are unreachable through the public GJK loop.
 */
export function doSimplex(simplex: SupportVertex[], direction: Vec3): boolean {
  if (simplex.length === 2) {
    return solveLine(simplex, direction);
  }
  if (simplex.length === 3) {
    return solveTriangle(simplex, direction);
  }
  return solveTetrahedron(simplex, direction);
}

/**
 * Closest feature of a segment `[a, b]` to the origin, where `a = simplex[0]`
 * is the newest vertex.
 */
function solveLine(simplex: SupportVertex[], direction: Vec3): boolean {
  const a = simplex[0];
  const b = simplex[1];
  const ab = b.point.clone().sub(a.point);
  const ao = a.point.clone().negate();
  if (ab.dot(ao) > 0) {
    // d = AB x AO x AB, the component of AO perpendicular to AB.
    direction.copy(ab).cross(ao).cross(ab);
  } else {
    simplex.length = 1;
    direction.copy(ao);
  }
  return false;
}

/**
 * Closest feature of a triangle `[a, b, c]` to the origin. `a` is the newest
 * vertex, so only the two edges incident to it and the face interior need to be
 * classified.
 */
function solveTriangle(simplex: SupportVertex[], direction: Vec3): boolean {
  const a = simplex[0];
  const b = simplex[1];
  const c = simplex[2];
  const ab = b.point.clone().sub(a.point);
  const ac = c.point.clone().sub(a.point);
  const ao = a.point.clone().negate();
  const abc = ab.clone().cross(ac);

  if (abc.clone().cross(ac).dot(ao) > 0) {
    // Origin is outside edge AC.
    if (ac.dot(ao) > 0) {
      simplex[1] = c;
      simplex.length = 2;
      return solveLine(simplex, direction);
    }
    simplex.length = 2;
    return solveLine(simplex, direction);
  }

  // The remaining edge is the previous simplex base. The new support point was
  // chosen on the origin side of that base, so the origin can never lie outside
  // it; only the face region is left to classify.
  if (abc.dot(ao) > 0) {
    direction.copy(abc);
  } else {
    simplex[1] = c;
    simplex[2] = b;
    direction.copy(abc).negate();
  }
  return false;
}

/**
 * Tetrahedron case. For every face, checks whether the origin lies on the
 * opposite side of the face plane from the remaining vertex; if so the simplex
 * reduces to that face. Orientation-independent: it compares signed plane
 * distances rather than assuming a winding.
 *
 * A non-degenerate tetrahedron that is not cut by any face plane contains the
 * origin (including on its boundary, which counts as touching).
 */
function solveTetrahedron(simplex: SupportVertex[], direction: Vec3): boolean {
  const a = simplex[0];
  const b = simplex[1];
  const c = simplex[2];
  const d = simplex[3];

  const faces: Array<[SupportVertex, SupportVertex, SupportVertex, SupportVertex]> = [
    [b, c, d, a],
    [a, c, d, b],
    [a, b, d, c],
    [a, b, c, d],
  ];

  for (const [v0, v1, v2, opposite] of faces) {
    const normal = v1.point
      .clone()
      .sub(v0.point)
      .cross(v2.point.clone().sub(v0.point));
    const h0 = normal.dot(v0.point);
    const hOpposite = normal.dot(opposite.point);
    if (h0 * (hOpposite - h0) > 0) {
      simplex.length = 0;
      simplex.push(v0, v1, v2);
      return solveTriangle(simplex, direction);
    }
  }

  return true;
}
