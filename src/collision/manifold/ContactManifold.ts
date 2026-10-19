import { Quat } from "../../math/Quat.js";
import { Mat3 } from "../../math/Mat3.js";
import type { Transform } from "../../math/Transform.js";
import { Vec3 } from "../../math/Vec3.js";
import type { Box } from "../shapes/Box.js";
import { epa } from "../narrowphase/EPA.js";
import { gjkIntersects, minkowskiSupport } from "../narrowphase/GJK.js";
import type { ContactManifold, ContactPoint } from "../narrowphase/Contact.js";
import { hashContactId, makeContactPoint } from "../narrowphase/Contact.js";

/** One of the six faces of an oriented box in world space. */
interface BoxFace {
  center: Vec3;
  normal: Vec3;
  axisIndex: number;
  sign: number;
  /** In-plane axes and their half extents, used to clip the incident polygon. */
  u: Vec3;
  v: Vec3;
  halfU: number;
  halfV: number;
  vertices: Vec3[];
}

const scratchMat3 = new Mat3();

/** World-space corners of a box, ordered by sign bits `(x, y, z)`. */
export function boxWorldVertices(box: Box, transform: Transform): Vec3[] {
  const { x: hx, y: hy, z: hz } = box.halfExtents;
  const vertices: Vec3[] = [];
  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      for (const sz of [-1, 1]) {
        vertices.push(
          transform.transformPoint(new Vec3(), new Vec3(sx * hx, sy * hy, sz * hz)),
        );
      }
    }
  }
  return vertices;
}

/** Builds the face of `box` whose outward normal is most aligned with `target`. */
function selectFace(
  box: Box,
  transform: Transform,
  target: Vec3,
): BoxFace {
  const rotation = Quat.toMat3(scratchMat3, transform.rotation);
  const axes = [
    new Vec3(rotation.m00, rotation.m10, rotation.m20),
    new Vec3(rotation.m01, rotation.m11, rotation.m21),
    new Vec3(rotation.m02, rotation.m12, rotation.m22),
  ];
  const halfExtents = [box.halfExtents.x, box.halfExtents.y, box.halfExtents.z];

  let axisIndex = 0;
  let bestAlignment = Math.abs(target.dot(axes[0]));
  for (let i = 1; i < 3; i++) {
    const alignment = Math.abs(target.dot(axes[i]));
    if (alignment > bestAlignment) {
      axisIndex = i;
      bestAlignment = alignment;
    }
  }

  const sign = target.dot(axes[axisIndex]) >= 0 ? 1 : -1;
  const normal = axes[axisIndex].clone().scale(sign);
  const center = transform.position.clone().addScaled(normal, halfExtents[axisIndex]);
  const u = axes[(axisIndex + 1) % 3];
  const v = axes[(axisIndex + 2) % 3];
  const halfU = halfExtents[(axisIndex + 1) % 3];
  const halfV = halfExtents[(axisIndex + 2) % 3];

  return {
    center,
    normal,
    axisIndex,
    sign,
    u,
    v,
    halfU,
    halfV,
    vertices: [
      center.clone().addScaled(u, halfU).addScaled(v, halfV),
      center.clone().addScaled(u, -halfU).addScaled(v, halfV),
      center.clone().addScaled(u, -halfU).addScaled(v, -halfV),
      center.clone().addScaled(u, halfU).addScaled(v, -halfV),
    ],
  };
}

/**
 * Sutherland-Hodgman clipping of a convex polygon against the half-space
 * `normal · p <= offset` (Section 5.3).
 */
export function clipPolygon(points: readonly Vec3[], normal: Vec3, offset: number): Vec3[] {
  if (points.length === 0) {
    return [];
  }
  const output: Vec3[] = [];
  let previous = points[points.length - 1];
  let previousInside = normal.dot(previous) <= offset;
  for (let i = 0; i < points.length; i++) {
    const current = points[i];
    const currentInside = normal.dot(current) <= offset;
    if (currentInside !== previousInside) {
      const edge = current.clone().sub(previous);
      const t = (offset - normal.dot(previous)) / normal.dot(edge);
      output.push(previous.clone().addScaled(edge, t));
    }
    if (currentInside) {
      output.push(current.clone());
    }
    previous = current;
    previousInside = currentInside;
  }
  return output;
}

/**
 * Manifold reduction to at most four points (Section 5.3): deepest point,
 * point furthest from it, largest triangle, then largest added quadrilateral
 * area.
 */
export function reduceManifold(points: readonly ContactPoint[]): ContactPoint[] {
  if (points.length <= 4) {
    return points.slice();
  }

  let first = 0;
  for (let i = 1; i < points.length; i++) {
    if (points[i].penetration > points[first].penetration) {
      first = i;
    }
  }
  const selectedIndices = [first];
  const selectedPoints = [points[first]];

  let second = first === 0 ? 1 : 0;
  let bestDistance = Vec3.distanceSq(points[second].position, points[first].position);
  for (let i = 0; i < points.length; i++) {
    if (i === first) {
      continue;
    }
    const distance = Vec3.distanceSq(points[i].position, points[first].position);
    if (distance > bestDistance) {
      second = i;
      bestDistance = distance;
    }
  }
  selectedIndices.push(second);
  selectedPoints.push(points[second]);

  const edgeFirstSecond = points[second].position.clone().sub(points[first].position);
  let third = -1;
  let bestArea = -1;
  for (let i = 0; i < points.length; i++) {
    if (i === first || i === second) {
      continue;
    }
    const areaSq = edgeFirstSecond
      .clone()
      .cross(points[i].position.clone().sub(points[first].position))
      .lengthSq();
    if (areaSq > bestArea) {
      third = i;
      bestArea = areaSq;
    }
  }
  selectedIndices.push(third);
  selectedPoints.push(points[third]);

  let fourth = -1;
  let bestQuadArea = -1;
  for (let i = 0; i < points.length; i++) {
    if (i === first || i === second || i === third) {
      continue;
    }
    const fromFirst = points[i].position.clone().sub(points[first].position);
    const area1 = Math.sqrt(
      edgeFirstSecond.clone().cross(fromFirst).lengthSq(),
    );
    const area2 = Math.sqrt(
      points[third]
        .position.clone()
        .sub(points[first].position)
        .cross(fromFirst)
        .lengthSq(),
    );
    if (area1 + area2 > bestQuadArea) {
      fourth = i;
      bestQuadArea = area1 + area2;
    }
  }
  selectedIndices.push(fourth);
  selectedPoints.push(points[fourth]);

  return selectedPoints;
}

/**
 * Contact manifold for box-box pairs.
 *
 * GJK/EPA provides the penetration normal; the incident face is then clipped
 * against the side planes of the reference face (Section 5.3). Edge-edge
 * contacts fall back to the single EPA witness point.
 */
export function boxBoxManifold(
  boxA: Box,
  transformA: Transform,
  boxB: Box,
  transformB: Transform,
): ContactManifold | null {
  const gjk = gjkIntersects(boxA, transformA, boxB, transformB);
  if (!gjk.intersects) {
    return null;
  }
  const result = epa(gjk.simplex, (direction) =>
    minkowskiSupport(boxA, transformA, boxB, transformB, direction),
  );
  if (result === null) {
    return null;
  }

  const normal = result.normal;
  const rotationA = Quat.toMat3(new Mat3(), transformA.rotation);
  const rotationB = Quat.toMat3(new Mat3(), transformB.rotation);
  const alignmentA = Math.max(
    Math.abs(normal.dot(new Vec3(rotationA.m00, rotationA.m10, rotationA.m20))),
    Math.abs(normal.dot(new Vec3(rotationA.m01, rotationA.m11, rotationA.m21))),
    Math.abs(normal.dot(new Vec3(rotationA.m02, rotationA.m12, rotationA.m22))),
  );
  const alignmentB = Math.max(
    Math.abs(normal.dot(new Vec3(rotationB.m00, rotationB.m10, rotationB.m20))),
    Math.abs(normal.dot(new Vec3(rotationB.m01, rotationB.m11, rotationB.m21))),
    Math.abs(normal.dot(new Vec3(rotationB.m02, rotationB.m12, rotationB.m22))),
  );
  const referenceIsA = alignmentA >= alignmentB;

  const referenceFace = referenceIsA
    ? selectFace(boxA, transformA, normal)
    : selectFace(boxB, transformB, normal.clone().negate());
  const incidentFace = referenceIsA
    ? selectFace(boxB, transformB, normal.clone().negate())
    : selectFace(boxA, transformA, normal);

  let polygon = incidentFace.vertices;
  const refU = referenceFace.u;
  const refV = referenceFace.v;
  polygon = clipPolygon(
    polygon,
    refU,
    referenceFace.center.dot(refU) + referenceFace.halfU,
  );
  polygon = clipPolygon(
    polygon,
    refU.clone().negate(),
    referenceFace.halfU - referenceFace.center.dot(refU),
  );
  polygon = clipPolygon(
    polygon,
    refV,
    referenceFace.center.dot(refV) + referenceFace.halfV,
  );
  polygon = clipPolygon(
    polygon,
    refV.clone().negate(),
    referenceFace.halfV - referenceFace.center.dot(refV),
  );

  const refNormal = referenceFace.normal;
  const refPlaneOffset = refNormal.dot(referenceFace.center);
  const points: ContactPoint[] = [];
  for (let i = 0; i < polygon.length; i++) {
    const separation = refNormal.dot(polygon[i]) - refPlaneOffset;
    if (separation > 0) {
      continue;
    }
    const feature = referenceFace.axisIndex * 2 + (referenceFace.sign > 0 ? 0 : 1);
    points.push(
      makeContactPoint(
        polygon[i],
        -separation,
        hashContactId((referenceIsA ? 0x100 : 0x200) + feature, i),
      ),
    );
  }

  if (points.length === 0) {
    const position = result.pointA.clone().add(result.pointB).scale(0.5);
    return {
      normal,
      points: [makeContactPoint(position, result.depth, hashContactId(0x300, 0))],
    };
  }

  return { normal, points: reduceManifold(points) };
}

/** Accumulated impulses cached for warm starting. */
export interface CachedImpulse {
  normalImpulse: number;
  tangentImpulse1: number;
  tangentImpulse2: number;
}

/**
 * Per-frame contact impulse cache used for warm starting (Section 5.3/6.4).
 *
 * Contacts are keyed by body pair and stable feature id. `beginStep` rotates
 * the current frame into the previous frame; the solver reads old impulses with
 * {@link ContactCache.get} and records new ones with
 * {@link ContactCache.store}.
 */
export class ContactCache {
  private previous = new Map<string, Map<number, CachedImpulse>>();
  private current = new Map<string, Map<number, CachedImpulse>>();

  /** Orders a body pair deterministically. */
  static pairKey(bodyA: number, bodyB: number): string {
    return bodyA < bodyB ? `${bodyA}:${bodyB}` : `${bodyB}:${bodyA}`;
  }

  /** Starts a new physics step. */
  beginStep(): void {
    this.previous = this.current;
    this.current = new Map();
  }

  /** Returns the impulse accumulated for this contact in the previous step. */
  get(pairKey: string, contactId: number): CachedImpulse | undefined {
    return this.previous.get(pairKey)?.get(contactId);
  }

  /** Records the impulse accumulated in the current step. */
  store(pairKey: string, contactId: number, impulse: CachedImpulse): void {
    let contacts = this.current.get(pairKey);
    if (contacts === undefined) {
      contacts = new Map();
      this.current.set(pairKey, contacts);
    }
    contacts.set(contactId, impulse);
  }

  /** Drops all cached data. */
  clear(): void {
    this.previous.clear();
    this.current.clear();
  }
}
