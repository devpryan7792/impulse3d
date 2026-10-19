import { EPSILON_SQ } from "../../math/MathUtils.js";
import { Vec3 } from "../../math/Vec3.js";
import type { SupportVertex } from "./GJK.js";

/** Result of an EPA query. `normal` points from shape A towards shape B. */
export interface EPAResult {
  normal: Vec3;
  depth: number;
  pointA: Vec3;
  pointB: Vec3;
}

/** A triangular face of the expanding polytope, with an outward unit normal. */
interface EPAFace {
  a: number;
  b: number;
  c: number;
  normal: Vec3;
  /** Plane offset along the outward normal (`>= 0` because the origin is inside). */
  distance: number;
}

/** A directed edge of a visible face, used by the horizon search. */
type Edge = [from: number, to: number];

/** Any vector perpendicular to `direction` (which must be non-zero). */
function perpendicular(direction: Vec3): Vec3 {
  const axis =
    Math.abs(direction.x) < Math.abs(direction.y)
      ? new Vec3(1, 0, 0)
      : new Vec3(0, 1, 0);
  return direction.clone().cross(axis).normalize();
}

/**
 * Creates a face and orients its normal away from `interior`. Degenerate faces
 * are skipped.
 */
function addFace(
  faces: EPAFace[],
  vertices: readonly SupportVertex[],
  interior: Vec3,
  ia: number,
  ib: number,
  ic: number,
): void {
  const pa = vertices[ia].point;
  const normal = vertices[ib].point
    .clone()
    .sub(pa)
    .cross(vertices[ic].point.clone().sub(pa));
  const lengthSq = normal.lengthSq();
  if (lengthSq < EPSILON_SQ) {
    return;
  }
  normal.scale(1 / Math.sqrt(lengthSq));
  if (normal.dot(pa.clone().sub(interior)) < 0) {
    normal.negate();
  }
  faces.push({ a: ia, b: ib, c: ic, normal, distance: normal.dot(pa) });
}

function closestFace(faces: readonly EPAFace[]): EPAFace {
  let best = faces[0];
  for (let i = 1; i < faces.length; i++) {
    if (faces[i].distance < best.distance) {
      best = faces[i];
    }
  }
  return best;
}

/**
 * Converts the closest face into a contact: the barycentric coordinates of the
 * projection of the origin onto the face interpolate the witness points stored
 * with each Minkowski vertex.
 */
function makeResult(face: EPAFace, vertices: readonly SupportVertex[]): EPAResult {
  const vertexA = vertices[face.a];
  const vertexB = vertices[face.b];
  const vertexC = vertices[face.c];

  const e0 = vertexB.point.clone().sub(vertexA.point);
  const e1 = vertexC.point.clone().sub(vertexA.point);
  const projection = face.normal.clone().scale(face.distance).sub(vertexA.point);

  const d00 = e0.dot(e0);
  const d01 = e0.dot(e1);
  const d11 = e1.dot(e1);
  const d20 = projection.dot(e0);
  const d21 = projection.dot(e1);
  const denominator = d00 * d11 - d01 * d01;
  const v = (d11 * d20 - d01 * d21) / denominator;
  const w = (d00 * d21 - d01 * d20) / denominator;
  const u = 1 - v - w;

  const pointA = new Vec3(
    u * vertexA.witnessA.x + v * vertexB.witnessA.x + w * vertexC.witnessA.x,
    u * vertexA.witnessA.y + v * vertexB.witnessA.y + w * vertexC.witnessA.y,
    u * vertexA.witnessA.z + v * vertexB.witnessA.z + w * vertexC.witnessA.z,
  );
  const pointB = new Vec3(
    u * vertexA.witnessB.x + v * vertexB.witnessB.x + w * vertexC.witnessB.x,
    u * vertexA.witnessB.y + v * vertexB.witnessB.y + w * vertexC.witnessB.y,
    u * vertexA.witnessB.z + v * vertexB.witnessB.z + w * vertexC.witnessB.z,
  );

  return {
    normal: face.normal.clone(),
    depth: Math.max(face.distance, 0),
    pointA,
    pointB,
  };
}

/**
 * GJK can terminate with a line or triangle simplex when the origin lies on a
 * symmetry axis or plane. EPA needs a tetrahedron, so the simplex is completed
 * with supports along a perpendicular and the triangle normal; the resulting
 * tetrahedron contains the origin (possibly on its boundary), which is enough
 * for the expansion to push out to the true supporting face.
 */
function completeTetrahedron(
  simplex: readonly SupportVertex[],
  support: (direction: Vec3) => SupportVertex,
): SupportVertex[] | null {
  if (simplex.length < 2) {
    return null;
  }
  const vertices = simplex.slice(0, 4);
  if (vertices.length === 2) {
    const line = vertices[1].point.clone().sub(vertices[0].point);
    vertices.push(support(perpendicular(line)));
  }
  const a = vertices[0].point;
  const ab = vertices[1].point.clone().sub(a);
  const ac = vertices[2].point.clone().sub(a);
  const normal = ab.cross(ac);
  if (normal.lengthSq() < EPSILON_SQ) {
    return null;
  }
  vertices.push(support(normal.normalize()));
  return vertices;
}

/**
 * Expanding Polytope Algorithm (Section 5.2).
 *
 * `simplex` must be the GJK simplex on a hit (2, 3 or 4 vertices), and
 * `support(d)` must return the Minkowski support point for the same shape pair.
 * The polytope is expanded by silhouette/horizon search until the support point
 * is within `tolerance` metres of the closest face.
 *
 * Returns `null` when no non-degenerate tetrahedron can be formed (e.g. exact
 * touching, where no penetration depth exists). `maxIterations` caps the
 * expansion; on exhaustion the current closest face is returned.
 */
export function epa(
  simplex: readonly SupportVertex[],
  support: (direction: Vec3) => SupportVertex,
  tolerance: number = 1e-4,
  maxIterations: number = 32,
): EPAResult | null {
  const completed = completeTetrahedron(simplex, support);
  if (completed === null) {
    return null;
  }

  const vertices: SupportVertex[] = completed.map((vertex) => ({
    point: vertex.point.clone(),
    witnessA: vertex.witnessA.clone(),
    witnessB: vertex.witnessB.clone(),
  }));

  const a = vertices[0].point;
  const ab = vertices[1].point.clone().sub(a);
  const ac = vertices[2].point.clone().sub(a);
  const ad = vertices[3].point.clone().sub(a);
  if (Math.abs(ab.dot(ac.clone().cross(ad))) < 1e-12) {
    return null;
  }

  const interior = a
    .clone()
    .add(vertices[1].point)
    .add(vertices[2].point)
    .add(vertices[3].point)
    .scale(0.25);

  const faces: EPAFace[] = [];
  addFace(faces, vertices, interior, 0, 1, 2);
  addFace(faces, vertices, interior, 0, 2, 3);
  addFace(faces, vertices, interior, 0, 3, 1);
  addFace(faces, vertices, interior, 1, 3, 2);

  for (let iteration = 0; iteration < maxIterations; iteration++) {
    const best = closestFace(faces);
    const supportVertex = support(best.normal);
    if (supportVertex.point.dot(best.normal) - best.distance < tolerance) {
      return makeResult(best, vertices);
    }

    const visible = faces.map((face) => {
      const outward = supportVertex.point.clone().sub(vertices[face.a].point);
      return outward.dot(face.normal) > 1e-15;
    });

    const visibleEdges: Edge[] = [];
    for (let i = 0; i < faces.length; i++) {
      if (!visible[i]) {
        continue;
      }
      const face = faces[i];
      visibleEdges.push([face.a, face.b], [face.b, face.c], [face.c, face.a]);
    }

    // A horizon edge is a visible-face edge whose reverse is not shared with
    // another visible face.
    const horizon: Edge[] = [];
    for (const [from, to] of visibleEdges) {
      let shared = false;
      for (const [otherFrom, otherTo] of visibleEdges) {
        if (otherFrom === to && otherTo === from) {
          shared = true;
          break;
        }
      }
      if (!shared) {
        horizon.push([from, to]);
      }
    }

    const remaining: EPAFace[] = [];
    for (let i = 0; i < faces.length; i++) {
      if (!visible[i]) {
        remaining.push(faces[i]);
      }
    }
    faces.length = 0;
    faces.push(...remaining);

    const newIndex = vertices.length;
    vertices.push({
      point: supportVertex.point.clone(),
      witnessA: supportVertex.witnessA.clone(),
      witnessB: supportVertex.witnessB.clone(),
    });
    for (const [from, to] of horizon) {
      addFace(faces, vertices, interior, from, to, newIndex);
    }
  }

  return makeResult(closestFace(faces), vertices);
}
