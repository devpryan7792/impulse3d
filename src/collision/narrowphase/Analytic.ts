import { EPSILON, EPSILON_SQ } from "../../math/MathUtils.js";
import type { Transform } from "../../math/Transform.js";
import { Vec3 } from "../../math/Vec3.js";
import type { Box } from "../shapes/Box.js";
import type { Capsule } from "../shapes/Capsule.js";
import type { Plane } from "../shapes/Plane.js";
import type { Sphere } from "../shapes/Sphere.js";
import type { ContactManifold, ContactPoint } from "./Contact.js";
import { hashContactId, makeContactPoint } from "./Contact.js";
import { reduceManifold } from "../manifold/ContactManifold.js";

/** Fallback normal for coincident centers. */
const FALLBACK_AXIS = new Vec3(1, 0, 0);

/**
 * Analytic sphere-sphere contact. The normal points from A to B; the contact
 * position is the midpoint between the two surface points.
 */
export function sphereSphere(
  sphereA: Sphere,
  transformA: Transform,
  sphereB: Sphere,
  transformB: Transform,
): ContactManifold | null {
  const delta = transformB.position.clone().sub(transformA.position);
  const distance = delta.length();
  const radiusSum = sphereA.radius + sphereB.radius;
  if (distance > radiusSum) {
    return null;
  }
  const normal = distance > EPSILON ? delta.scale(1 / distance) : FALLBACK_AXIS.clone();
  const depth = radiusSum - distance;
  const position = transformA.position
    .clone()
    .addScaled(normal, sphereA.radius - 0.5 * depth);
  return {
    normal,
    points: [makeContactPoint(position, depth, hashContactId(0, 0))],
  };
}

/**
 * Analytic sphere-box contact, normal from the box to the sphere.
 *
 * The sphere center is clamped into the box in local space. Outside the box the
 * closest surface point defines the normal; with the center inside, the nearest
 * face is the escape direction and the depth is radius plus face distance.
 */
export function boxSphere(
  box: Box,
  transformBox: Transform,
  sphere: Sphere,
  transformSphere: Transform,
): ContactManifold | null {
  const localCenter = transformBox.inverseTransformPoint(new Vec3(), transformSphere.position);
  const { x: hx, y: hy, z: hz } = box.halfExtents;
  const closest = new Vec3(
    Math.max(-hx, Math.min(hx, localCenter.x)),
    Math.max(-hy, Math.min(hy, localCenter.y)),
    Math.max(-hz, Math.min(hz, localCenter.z)),
  );
  const delta = localCenter.clone().sub(closest);
  const distanceSq = delta.lengthSq();
  const radius = sphere.radius;

  if (distanceSq > radius * radius) {
    return null;
  }

  if (distanceSq > EPSILON_SQ) {
    const distance = Math.sqrt(distanceSq);
    const localNormal = delta.scale(1 / distance);
    const normal = transformBox.transformDirection(new Vec3(), localNormal);
    const surface = transformBox.transformPoint(new Vec3(), closest);
    const depth = radius - distance;
    const position = surface.addScaled(normal, -0.5 * depth);
    return {
      normal,
      points: [makeContactPoint(position, depth, hashContactId(1, 0))],
    };
  }

  // The sphere center is on or inside the box: escape through the nearest face.
  let axis = 0;
  let sign = 1;
  let faceDistance = hx - localCenter.x;
  const negativeX = localCenter.x + hx;
  if (negativeX < faceDistance) {
    axis = 0;
    sign = -1;
    faceDistance = negativeX;
  }
  const positiveY = hy - localCenter.y;
  if (positiveY < faceDistance) {
    axis = 1;
    sign = 1;
    faceDistance = positiveY;
  }
  const negativeY = localCenter.y + hy;
  if (negativeY < faceDistance) {
    axis = 1;
    sign = -1;
    faceDistance = negativeY;
  }
  const positiveZ = hz - localCenter.z;
  if (positiveZ < faceDistance) {
    axis = 2;
    sign = 1;
    faceDistance = positiveZ;
  }
  const negativeZ = localCenter.z + hz;
  if (negativeZ < faceDistance) {
    axis = 2;
    sign = -1;
    faceDistance = negativeZ;
  }

  const localNormal = new Vec3();
  if (axis === 0) {
    localNormal.x = sign;
  } else if (axis === 1) {
    localNormal.y = sign;
  } else {
    localNormal.z = sign;
  }
  const facePoint = localCenter.clone();
  if (axis === 0) {
    facePoint.x = sign * hx;
  } else if (axis === 1) {
    facePoint.y = sign * hy;
  } else {
    facePoint.z = sign * hz;
  }
  const normal = transformBox.transformDirection(new Vec3(), localNormal);
  const position = transformBox.transformPoint(new Vec3(), facePoint);
  const depth = radius + faceDistance;
  return {
    normal,
    points: [makeContactPoint(position, depth, hashContactId(1, 0))],
  };
}

/**
 * Analytic sphere-capsule contact, normal from the sphere to the capsule. The
 * closest point on the capsule segment is found by projection.
 */
export function sphereCapsule(
  sphere: Sphere,
  transformSphere: Transform,
  capsule: Capsule,
  transformCapsule: Transform,
): ContactManifold | null {
  const epsilon = new Vec3(0, capsule.halfHeight, 0);
  const endpointA = transformCapsule.transformPoint(new Vec3(), epsilon.clone().negate());
  const endpointB = transformCapsule.transformPoint(new Vec3(), epsilon);
  const segment = endpointB.clone().sub(endpointA);
  const center = transformSphere.position;
  const t = Math.max(
    0,
    Math.min(1, center.clone().sub(endpointA).dot(segment) / segment.lengthSq()),
  );
  const closest = endpointA.addScaled(segment, t);
  const delta = closest.clone().sub(center);
  const distance = delta.length();
  const radiusSum = sphere.radius + capsule.radius;
  if (distance > radiusSum) {
    return null;
  }
  const normal = distance > EPSILON ? delta.scale(1 / distance) : FALLBACK_AXIS.clone();
  const depth = radiusSum - distance;
  const position = center
    .clone()
    .addScaled(normal, sphere.radius - 0.5 * depth);
  return {
    normal,
    points: [makeContactPoint(position, depth, hashContactId(2, 0))],
  };
}

/** Analytic plane-sphere contact, normal from the plane to the sphere. */
export function planeSphere(
  plane: Plane,
  transformPlane: Transform,
  sphere: Sphere,
  transformSphere: Transform,
): ContactManifold | null {
  const normal = plane.getWorldNormal(transformPlane);
  const height = normal.dot(transformSphere.position) - plane.getWorldOffset(transformPlane);
  if (height > sphere.radius) {
    return null;
  }
  const depth = sphere.radius - height;
  const position = transformSphere.position
    .clone()
    .addScaled(normal, -0.5 * (height + sphere.radius));
  return {
    normal,
    points: [makeContactPoint(position, depth, hashContactId(3, 0))],
  };
}

/** Analytic plane-capsule contact using the deepest capsule endpoint. */
export function planeCapsule(
  plane: Plane,
  transformPlane: Transform,
  capsule: Capsule,
  transformCapsule: Transform,
): ContactManifold | null {
  const normal = plane.getWorldNormal(transformPlane);
  const deepest = capsule.getSupport(normal.clone().negate(), transformCapsule);
  const height = normal.dot(deepest) - plane.getWorldOffset(transformPlane);
  if (height > 0) {
    return null;
  }
  const depth = -height;
  const position = deepest.addScaled(normal, 0.5 * depth);
  return {
    normal,
    points: [makeContactPoint(position, depth, hashContactId(3, 1))],
  };
}

/**
 * Analytic plane-convex polyhedron contact: every vertex below the plane
 * generates a contact point, and the manifold is reduced to at most four.
 * `vertices` must be world-space points ordered deterministically by the
 * caller so contact ids stay stable across frames.
 */
export function planePolyhedron(
  plane: Plane,
  transformPlane: Transform,
  vertices: readonly Vec3[],
): ContactManifold | null {
  const normal = plane.getWorldNormal(transformPlane);
  const offset = plane.getWorldOffset(transformPlane);
  const points: ContactPoint[] = [];
  for (let i = 0; i < vertices.length; i++) {
    const vertex = vertices[i];
    const height = normal.dot(vertex) - offset;
    if (height > 0) {
      continue;
    }
    const depth = -height;
    const position = vertex.clone().addScaled(normal, 0.5 * depth);
    points.push(makeContactPoint(position, depth, hashContactId(3, i)));
  }
  if (points.length === 0) {
    return null;
  }
  return { normal, points: reduceManifold(points) };
}
