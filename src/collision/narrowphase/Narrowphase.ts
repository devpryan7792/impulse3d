import type { Transform } from "../../math/Transform.js";
import { Vec3 } from "../../math/Vec3.js";
import type { Box } from "../shapes/Box.js";
import type { Capsule } from "../shapes/Capsule.js";
import type { ConvexPolytope } from "../shapes/ConvexPolytope.js";
import type { Plane } from "../shapes/Plane.js";
import type { Shape } from "../shapes/Shape.js";
import { ShapeType } from "../shapes/Shape.js";
import type { Sphere } from "../shapes/Sphere.js";
import { boxBoxManifold, boxWorldVertices } from "../manifold/ContactManifold.js";
import {
  boxSphere,
  planeCapsule,
  planePolyhedron,
  planeSphere,
  sphereCapsule,
  sphereSphere,
} from "./Analytic.js";
import type { ContactManifold } from "./Contact.js";
import { flipManifold, hashContactId, makeContactPoint } from "./Contact.js";
import { epa } from "./EPA.js";
import { gjkIntersects, minkowskiSupport } from "./GJK.js";

/**
 * Analytic plane contacts. The plane must be the first argument; the returned
 * normal points from the plane towards `shape`.
 */
export function planeManifold(
  plane: Plane,
  transformPlane: Transform,
  shape: Shape,
  transformShape: Transform,
): ContactManifold | null {
  switch (shape.type) {
    case ShapeType.Sphere:
      return planeSphere(plane, transformPlane, shape as Sphere, transformShape);
    case ShapeType.Capsule:
      return planeCapsule(plane, transformPlane, shape as Capsule, transformShape);
    case ShapeType.Box:
      return planePolyhedron(
        plane,
        transformPlane,
        boxWorldVertices(shape as Box, transformShape),
      );
    case ShapeType.ConvexPolytope:
      return planePolyhedron(
        plane,
        transformPlane,
        (shape as ConvexPolytope).vertices.map((vertex) =>
          transformShape.transformPoint(new Vec3(), vertex),
        ),
      );
    default:
      // Plane-plane: two half-space boundaries have no finite contact region.
      return null;
  }
}

/**
 * Narrowphase entry point. Returns a manifold whose normal points from `shapeA`
 * to `shapeB`, or `null` when the pair does not touch.
 *
 * Dispatch order:
 * 1. analytic fast paths (sphere-sphere, sphere-box, sphere-capsule, planes),
 * 2. box-box clipped manifolds,
 * 3. GJK + EPA for every remaining convex pair.
 */
export function collideShapes(
  shapeA: Shape,
  transformA: Transform,
  shapeB: Shape,
  transformB: Transform,
): ContactManifold | null {
  if (shapeA.type === ShapeType.Plane) {
    return planeManifold(shapeA as Plane, transformA, shapeB, transformB);
  }
  if (shapeB.type === ShapeType.Plane) {
    const manifold = planeManifold(shapeB as Plane, transformB, shapeA, transformA);
    return manifold === null ? null : flipManifold(manifold);
  }

  if (shapeA.type === ShapeType.Sphere) {
    if (shapeB.type === ShapeType.Sphere) {
      return sphereSphere(shapeA as Sphere, transformA, shapeB as Sphere, transformB);
    }
    if (shapeB.type === ShapeType.Box) {
      const manifold = boxSphere(
        shapeB as Box,
        transformB,
        shapeA as Sphere,
        transformA,
      );
      return manifold === null ? null : flipManifold(manifold);
    }
    if (shapeB.type === ShapeType.Capsule) {
      return sphereCapsule(
        shapeA as Sphere,
        transformA,
        shapeB as Capsule,
        transformB,
      );
    }
  } else if (shapeA.type === ShapeType.Box && shapeB.type === ShapeType.Sphere) {
    return boxSphere(shapeA as Box, transformA, shapeB as Sphere, transformB);
  } else if (shapeA.type === ShapeType.Capsule && shapeB.type === ShapeType.Sphere) {
    const manifold = sphereCapsule(
      shapeB as Sphere,
      transformB,
      shapeA as Capsule,
      transformA,
    );
    return manifold === null ? null : flipManifold(manifold);
  } else if (shapeA.type === ShapeType.Box && shapeB.type === ShapeType.Box) {
    return boxBoxManifold(shapeA as Box, transformA, shapeB as Box, transformB);
  }

  const gjk = gjkIntersects(shapeA, transformA, shapeB, transformB);
  if (!gjk.intersects) {
    return null;
  }
  const result = epa(gjk.simplex, (direction) =>
    minkowskiSupport(shapeA, transformA, shapeB, transformB, direction),
  );
  if (result === null) {
    return null;
  }
  const position = result.pointA.clone().add(result.pointB).scale(0.5);
  return {
    normal: result.normal,
    points: [makeContactPoint(position, result.depth, hashContactId(0, 1))],
  };
}
