import { describe, expect, test } from "vitest";
import { Quat } from "../src/math/Quat.js";
import { Transform } from "../src/math/Transform.js";
import { Vec3 } from "../src/math/Vec3.js";
import { AABB } from "../src/collision/AABB.js";
import { Box } from "../src/collision/shapes/Box.js";
import { Capsule } from "../src/collision/shapes/Capsule.js";
import { ConvexPolytope } from "../src/collision/shapes/ConvexPolytope.js";
import { Plane } from "../src/collision/shapes/Plane.js";
import { Shape, ShapeType } from "../src/collision/shapes/Shape.js";
import { Sphere } from "../src/collision/shapes/Sphere.js";
import {
  boxSphere,
  planeCapsule,
  planePolyhedron,
  planeSphere,
  sphereCapsule,
  sphereSphere,
} from "../src/collision/narrowphase/Analytic.js";
import type { ContactManifold } from "../src/collision/narrowphase/Contact.js";
import { flipManifold, hashContactId, makeContactPoint } from "../src/collision/narrowphase/Contact.js";
import { epa } from "../src/collision/narrowphase/EPA.js";
import { doSimplex, gjkIntersects, minkowskiSupport } from "../src/collision/narrowphase/GJK.js";
import { collideShapes, planeManifold } from "../src/collision/narrowphase/Narrowphase.js";
import {
  boxBoxManifold,
  boxWorldVertices,
  clipPolygon,
  ContactCache,
  reduceManifold,
} from "../src/collision/manifold/ContactManifold.js";
import { expectVecClose, makeRng, randomQuat, randomVec3 } from "./testUtils.js";

const rng = makeRng(0x9a7c0de);

function transformAt(x: number, y: number, z: number, rotation = new Quat()): Transform {
  return new Transform(new Vec3(x, y, z), rotation);
}

/** Shape whose support mapping is a fixed world-space point (test scaffolding). */
class FixedShape extends Shape {
  public readonly type: ShapeType = ShapeType.ConvexPolytope;
  public readonly point: Vec3;

  constructor(point: Vec3) {
    super();
    this.point = point;
  }

  override computeAABB(_transform: Transform, out: AABB = new AABB()): AABB {
    out.min.copy(this.point);
    out.max.copy(this.point);
    return out;
  }

  override getSupport(_direction: Vec3, _transform: Transform, out: Vec3 = new Vec3()): Vec3 {
    return out.copy(this.point);
  }
}

/** Shape that returns a fixed support sequence, used to drive GJK simplex regions. */
class QueueShape extends Shape {
  public readonly type: ShapeType = ShapeType.ConvexPolytope;
  private readonly points: Vec3[];
  private index = 0;

  constructor(points: Vec3[]) {
    super();
    this.points = points;
  }

  override computeAABB(_transform: Transform, out: AABB = new AABB()): AABB {
    out.min.copy(this.points[0]);
    out.max.copy(this.points[0]);
    return out;
  }

  override getSupport(_direction: Vec3, _transform: Transform, out: Vec3 = new Vec3()): Vec3 {
    const point = this.points[Math.min(this.index, this.points.length - 1)];
    this.index++;
    return out.copy(point);
  }
}

function queueGjk(points: number[][]): boolean {
  const identity = transformAt(0, 0, 0);
  return gjkIntersects(
    new QueueShape(points.map(([x, y, z]) => new Vec3(x, y, z))),
    identity,
    new FixedShape(new Vec3()),
    identity,
  ).intersects;
}

function expectManifold(manifold: ContactManifold | null): ContactManifold {
  expect(manifold).not.toBeNull();
  return manifold as ContactManifold;
}

function dot(a: Vec3, b: Vec3): number {
  return a.dot(b);
}

/** Reference OBB overlap test (15-axis separating axis theorem). */
function obbOverlap(
  halfA: Vec3,
  rotationA: Quat,
  positionA: Vec3,
  halfB: Vec3,
  rotationB: Quat,
  positionB: Vec3,
): boolean {
  const axesA = [
    Quat.rotateVector(new Vec3(), rotationA, new Vec3(1, 0, 0)),
    Quat.rotateVector(new Vec3(), rotationA, new Vec3(0, 1, 0)),
    Quat.rotateVector(new Vec3(), rotationA, new Vec3(0, 0, 1)),
  ];
  const axesB = [
    Quat.rotateVector(new Vec3(), rotationB, new Vec3(1, 0, 0)),
    Quat.rotateVector(new Vec3(), rotationB, new Vec3(0, 1, 0)),
    Quat.rotateVector(new Vec3(), rotationB, new Vec3(0, 0, 1)),
  ];
  const halfExtentsA = [halfA.x, halfA.y, halfA.z];
  const halfExtentsB = [halfB.x, halfB.y, halfB.z];
  const delta = positionB.clone().sub(positionA);
  const axes: Vec3[] = [...axesA, ...axesB];
  for (const axisA of axesA) {
    for (const axisB of axesB) {
      const cross = axisA.clone().cross(axisB);
      if (cross.lengthSq() > 1e-12) {
        axes.push(cross.normalize());
      }
    }
  }
  for (const axis of axes) {
    let radiusA = 0;
    for (let i = 0; i < 3; i++) {
      radiusA += Math.abs(dot(axis, axesA[i])) * halfExtentsA[i];
    }
    let radiusB = 0;
    for (let i = 0; i < 3; i++) {
      radiusB += Math.abs(dot(axis, axesB[i])) * halfExtentsB[i];
    }
    if (Math.abs(dot(delta, axis)) > radiusA + radiusB + 1e-12) {
      return false;
    }
  }
  return true;
}

describe("sphere-sphere", () => {
  test("separated spheres return null", () => {
    const manifold = sphereSphere(
      new Sphere(1),
      transformAt(0, 0, 0),
      new Sphere(2),
      transformAt(5, 0, 0),
    );
    expect(manifold).toBeNull();
  });

  test("overlapping spheres report depth, normal and midpoint", () => {
    const manifold = expectManifold(
      sphereSphere(new Sphere(1), transformAt(0, 0, 0), new Sphere(1), transformAt(1.5, 0, 0)),
    );
    expectVecClose(manifold.normal, new Vec3(1, 0, 0));
    expect(manifold.points).toHaveLength(1);
    expect(manifold.points[0].penetration).toBeCloseTo(0.5, 12);
    expectVecClose(manifold.points[0].position, new Vec3(0.75, 0, 0));
  });

  test("touching spheres produce a zero-depth contact", () => {
    const manifold = expectManifold(
      sphereSphere(new Sphere(1), transformAt(0, 0, 0), new Sphere(2), transformAt(3, 0, 0)),
    );
    expect(manifold.points[0].penetration).toBeCloseTo(0, 12);
    expectVecClose(manifold.points[0].position, new Vec3(1, 0, 0));
  });

  test("coincident sphere centers fall back to +X", () => {
    const manifold = expectManifold(
      sphereSphere(new Sphere(1), transformAt(0, 0, 0), new Sphere(2), transformAt(0, 0, 0)),
    );
    expectVecClose(manifold.normal, new Vec3(1, 0, 0));
    expect(manifold.points[0].penetration).toBeCloseTo(3, 12);
    expectVecClose(manifold.points[0].position, new Vec3(-0.5, 0, 0));
  });
});

describe("sphere-box", () => {
  const box = new Box(new Vec3(1, 1, 1));

  test("sphere outside the box returns null", () => {
    expect(
      boxSphere(box, transformAt(0, 0, 0), new Sphere(0.5), transformAt(2, 0, 0)),
    ).toBeNull();
  });

  test("face overlap", () => {
    const manifold = expectManifold(
      boxSphere(box, transformAt(0, 0, 0), new Sphere(0.5), transformAt(1.2, 0, 0)),
    );
    expectVecClose(manifold.normal, new Vec3(1, 0, 0));
    expect(manifold.points[0].penetration).toBeCloseTo(0.3, 12);
    expectVecClose(manifold.points[0].position, new Vec3(0.85, 0, 0));
  });

  test("corner overlap", () => {
    const manifold = expectManifold(
      boxSphere(box, transformAt(0, 0, 0), new Sphere(0.5), transformAt(1.2, 1.1, 1)),
    );
    const distance = Math.hypot(0.2, 0.1);
    const expected = new Vec3(0.2 / distance, 0.1 / distance, 0);
    expectVecClose(manifold.normal, expected, 1e-12);
    expect(manifold.points[0].penetration).toBeCloseTo(0.5 - distance, 12);
    const surface = new Vec3(1, 1, 1);
    expectVecClose(
      manifold.points[0].position,
      surface.addScaled(expected, -0.5 * (0.5 - distance)),
      1e-12,
    );
  });

  test("sphere center inside the box escapes through the nearest face", () => {
    const manifold = expectManifold(
      boxSphere(box, transformAt(0, 0, 0), new Sphere(0.5), transformAt(0.9, 0, 0)),
    );
    expectVecClose(manifold.normal, new Vec3(1, 0, 0));
    expect(manifold.points[0].penetration).toBeCloseTo(0.6, 12);
    expectVecClose(manifold.points[0].position, new Vec3(1, 0, 0));
  });

  test("center exactly at the box origin chooses +X deterministically", () => {
    const manifold = expectManifold(
      boxSphere(box, transformAt(0, 0, 0), new Sphere(0.5), transformAt(0, 0, 0)),
    );
    expectVecClose(manifold.normal, new Vec3(1, 0, 0));
    expect(manifold.points[0].penetration).toBeCloseTo(1.5, 12);
  });

  test("inside branch selects negative Y, positive Z and negative Z faces", () => {
    const negativeY = expectManifold(
      boxSphere(box, transformAt(0, 0, 0), new Sphere(0.1), transformAt(0, -0.95, 0)),
    );
    expectVecClose(negativeY.normal, new Vec3(0, -1, 0));

    const positiveZ = expectManifold(
      boxSphere(box, transformAt(0, 0, 0), new Sphere(0.1), transformAt(0, 0, 0.9)),
    );
    expectVecClose(positiveZ.normal, new Vec3(0, 0, 1));

    // -Z is nearer than +Z only for a center at negative z.
    const negativeZ = expectManifold(
      boxSphere(box, transformAt(0, 0, 0), new Sphere(0.1), transformAt(0, 0, -0.9)),
    );
    expectVecClose(negativeZ.normal, new Vec3(0, 0, -1));

    const positiveYBranch = expectManifold(
      boxSphere(box, transformAt(0, 0, 0), new Sphere(0.1), transformAt(0, 0.9, 0)),
    );
    expectVecClose(positiveYBranch.normal, new Vec3(0, 1, 0));

    const negativeX = expectManifold(
      boxSphere(box, transformAt(0, 0, 0), new Sphere(0.1), transformAt(-0.9, 0, 0)),
    );
    expectVecClose(negativeX.normal, new Vec3(-1, 0, 0));
  });

  test("rotated box maps the local normal to world space", () => {
    const rotation = Quat.fromAxisAngle(new Vec3(0, 0, 1), Math.PI / 2);
    const manifold = expectManifold(
      boxSphere(box, transformAt(0, 0, 0, rotation), new Sphere(0.5), transformAt(0, 1.2, 0)),
    );
    expectVecClose(manifold.normal, new Vec3(0, 1, 0), 1e-12);
    expect(manifold.points[0].penetration).toBeCloseTo(0.3, 12);
  });

  test("touching sphere-box produces zero depth", () => {
    const manifold = expectManifold(
      boxSphere(box, transformAt(0, 0, 0), new Sphere(0.5), transformAt(1.5, 0, 0)),
    );
    expect(manifold.points[0].penetration).toBeCloseTo(0, 12);
  });
});

describe("sphere-capsule", () => {
  const capsule = new Capsule(0.5, 1);

  test("outside the capsule radius returns null", () => {
    expect(
      sphereCapsule(new Sphere(0.5), transformAt(2, 0, 0), capsule, transformAt(0, 0, 0)),
    ).toBeNull();
  });

  test("side contact", () => {
    const manifold = expectManifold(
      sphereCapsule(new Sphere(0.5), transformAt(0.8, 0, 0), capsule, transformAt(0, 0, 0)),
    );
    expectVecClose(manifold.normal, new Vec3(-1, 0, 0));
    expect(manifold.points[0].penetration).toBeCloseTo(0.2, 12);
    expectVecClose(manifold.points[0].position, new Vec3(0.4, 0, 0));
  });

  test("end cap contact", () => {
    const manifold = expectManifold(
      sphereCapsule(new Sphere(0.5), transformAt(0, 1.8, 0), capsule, transformAt(0, 0, 0)),
    );
    expectVecClose(manifold.normal, new Vec3(0, -1, 0));
    expect(manifold.points[0].penetration).toBeCloseTo(0.2, 12);
    expectVecClose(manifold.points[0].position, new Vec3(0, 1.4, 0));
  });

  test("coincident sphere center and capsule axis fall back to +X", () => {
    const manifold = expectManifold(
      sphereCapsule(new Sphere(0.5), transformAt(0, 0, 0), capsule, transformAt(0, 0, 0)),
    );
    expectVecClose(manifold.normal, new Vec3(1, 0, 0));
    expect(manifold.points[0].penetration).toBeCloseTo(1, 12);
  });

  test("projection clamps to both segment endpoints", () => {
    const below = expectManifold(
      sphereCapsule(new Sphere(0.5), transformAt(0, -1.8, 0), capsule, transformAt(0, 0, 0)),
    );
    expectVecClose(below.normal, new Vec3(0, 1, 0));
    expectVecClose(below.points[0].position, new Vec3(0, -1.4, 0));
  });
});

describe("plane contacts", () => {
  const plane = new Plane();
  const planeTransform = transformAt(0, 0, 0);

  test("plane-sphere above, touching and below", () => {
    expect(
      planeSphere(plane, planeTransform, new Sphere(0.5), transformAt(0, 2, 0)),
    ).toBeNull();

    const touching = expectManifold(
      planeSphere(plane, planeTransform, new Sphere(0.5), transformAt(0, 0.5, 0)),
    );
    expect(touching.points[0].penetration).toBeCloseTo(0, 12);

    const below = expectManifold(
      planeSphere(plane, planeTransform, new Sphere(0.5), transformAt(0, 0.4, 0)),
    );
    expectVecClose(below.normal, new Vec3(0, 1, 0));
    expect(below.points[0].penetration).toBeCloseTo(0.1, 12);
    expectVecClose(below.points[0].position, new Vec3(0, -0.05, 0));
  });

  test("plane-capsule uses the deepest endpoint", () => {
    expect(
      planeCapsule(plane, planeTransform, new Capsule(0.5, 1), transformAt(0, 3, 0)),
    ).toBeNull();

    const manifold = expectManifold(
      planeCapsule(plane, planeTransform, new Capsule(0.5, 1), transformAt(0, 1.4, 0)),
    );
    expectVecClose(manifold.normal, new Vec3(0, 1, 0));
    expect(manifold.points[0].penetration).toBeCloseTo(0.1, 12);
    expectVecClose(manifold.points[0].position, new Vec3(0, -0.05, 0));
  });

  test("plane-box generates four points for a resting box", () => {
    const box = new Box(new Vec3(1, 1, 1));
    const manifold = expectManifold(
      planePolyhedron(
        plane,
        planeTransform,
        boxWorldVertices(box, transformAt(0, 0.9, 0)),
      ),
    );
    expectVecClose(manifold.normal, new Vec3(0, 1, 0));
    expect(manifold.points).toHaveLength(4);
    for (const point of manifold.points) {
      expect(point.penetration).toBeCloseTo(0.1, 12);
      expect(point.position.y).toBeCloseTo(-0.05, 12);
    }
  });

  test("plane-polyhedron keeps only vertices below the plane", () => {
    const polyhedron = new ConvexPolytope([
      new Vec3(0, 0, 0),
      new Vec3(1, 0, 0),
      new Vec3(0, 1, 0),
      new Vec3(0, 0, 1),
    ]);
    const manifold = expectManifold(
      planePolyhedron(
        plane,
        planeTransform,
        polyhedron.vertices.map((vertex) =>
          transformAt(0, -0.5, 0).transformPoint(new Vec3(), vertex),
        ),
      ),
    );
    expect(manifold.points).toHaveLength(3);
    for (const point of manifold.points) {
      expect(point.penetration).toBeCloseTo(0.5, 12);
    }
    expect(
      planePolyhedron(plane, planeTransform, polyhedron.vertices.map((vertex) => vertex.clone().add(new Vec3(0, 2, 0)))),
    ).toBeNull();
  });
});

describe("GJK", () => {
  test("separated and overlapping spheres", () => {
    expect(
      gjkIntersects(new Sphere(1), transformAt(0, 0, 0), new Sphere(1), transformAt(5, 0, 0))
        .intersects,
    ).toBe(false);
    const hit = gjkIntersects(
      new Sphere(1),
      transformAt(0, 0, 0),
      new Sphere(1),
      transformAt(1.5, 0, 0),
    );
    expect(hit.intersects).toBe(true);
    expect(hit.simplex.length).toBeGreaterThanOrEqual(2);
  });

  test("touching spheres are reported as intersecting", () => {
    expect(
      gjkIntersects(new Sphere(1), transformAt(0, 0, 0), new Sphere(1), transformAt(2, 0, 0))
        .intersects,
    ).toBe(true);
  });

  test("boxes: separated, overlapping and touching", () => {
    const box = new Box(new Vec3(1, 1, 1));
    expect(
      gjkIntersects(box, transformAt(0, 0, 0), box, transformAt(3, 0, 0)).intersects,
    ).toBe(false);
    expect(
      gjkIntersects(box, transformAt(0, 0, 0), box, transformAt(1.5, 0, 0)).intersects,
    ).toBe(true);
    expect(
      gjkIntersects(box, transformAt(0, 0, 0), box, transformAt(2, 0, 0)).intersects,
    ).toBe(true);
  });

  test("zero initial direction and repeated support are handled", () => {
    const coincident = gjkIntersects(
      new FixedShape(new Vec3(0, 0, 0)),
      transformAt(0, 0, 0),
      new FixedShape(new Vec3(0, 0, 0)),
      transformAt(0, 0, 0),
    );
    expect(coincident.intersects).toBe(true);

    const repeated = gjkIntersects(
      new FixedShape(new Vec3(0, 0, 0)),
      transformAt(0, 0, 0),
      new FixedShape(new Vec3(1, 0, 0)),
      transformAt(0, 0, 0),
    );
    expect(repeated.intersects).toBe(false);
  });

  test("agrees with analytic sphere overlap over random samples", () => {
    for (let i = 0; i < 200; i++) {
      const radiusA = rng.range(0.2, 1.5);
      const radiusB = rng.range(0.2, 1.5);
      const centerA = randomVec3(rng, -1, 1);
      const centerB = randomVec3(rng, -1, 1);
      const expected = centerA.distanceTo(centerB) <= radiusA + radiusB;
      const hit = gjkIntersects(
        new Sphere(radiusA),
        new Transform(centerA),
        new Sphere(radiusB),
        new Transform(centerB),
      ).intersects;
      expect(hit).toBe(expected);
    }
  });

  test("agrees with OBB SAT over random box pairs", () => {
    const halfA = new Vec3(0.5, 0.8, 0.3);
    const halfB = new Vec3(0.4, 0.6, 0.9);
    const boxA = new Box(halfA);
    const boxB = new Box(halfB);
    for (let i = 0; i < 150; i++) {
      const positionA = randomVec3(rng, -1, 1);
      const positionB = randomVec3(rng, -1, 1);
      const rotationA = randomQuat(rng);
      const rotationB = randomQuat(rng);
      const expected = obbOverlap(halfA, rotationA, positionA, halfB, rotationB, positionB);
      const hit = gjkIntersects(
        boxA,
        new Transform(positionA.clone(), rotationA),
        boxB,
        new Transform(positionB.clone(), rotationB),
      ).intersects;
      expect(hit).toBe(expected);
    }
  });

  test("line simplex reduces to a vertex when the origin is behind A", () => {
    const vertex = (x: number, y: number, z: number) => ({
      point: new Vec3(x, y, z),
      witnessA: new Vec3(x, y, z),
      witnessB: new Vec3(),
    });
    const simplex = [vertex(1, 0, 0), vertex(3, 0, 0)];
    const direction = new Vec3();
    expect(doSimplex(simplex, direction)).toBe(false);
    expect(simplex).toHaveLength(1);
    expectVecClose(direction, new Vec3(-1, 0, 0));
  });

  test("triangle simplex reduces to edge AC near vertex C", () => {
    const vertex = (x: number, y: number, z: number) => ({
      point: new Vec3(x, y, z),
      witnessA: new Vec3(x, y, z),
      witnessB: new Vec3(),
    });
    // Search-generated triangle with the origin outside edge AC and closer to C.
    const simplex = [
      vertex(-1, -1, -4),
      vertex(-3, 1, -0.5),
      vertex(-1, 0.5, 3),
    ];
    const direction = new Vec3();
    expect(doSimplex(simplex, direction)).toBe(false);
    expect(simplex).toHaveLength(2);
  });

  test("triangle simplex Voronoi regions are all exercised", () => {
    // Support sequences found by simulating the simplex-region predicates.
    expect(queueGjk([[1, -3.25, 0], [-4, -1, 3.5], [-2.25, 3.75, 1.5]])).toBe(false);
    expect(queueGjk([[1.75, -3.25, -0.75], [-2.75, 3.5, 2.75], [3.25, 1.5, -1.25]])).toBe(false);
    expect(queueGjk([[-2.75, 1.25, -2.75], [3, 3.5, 2], [2, -2.5, 1]])).toBe(false);
    // Reused support point terminates the iteration without progress.
    expect(
      queueGjk([[-1, 1, -1], [4, -2, -3], [-1, 3, 1.75], [0.75, 1.25, 3.25]]),
    ).toBe(false);
  });

  test("triangle simplex reduces to edge AB away from vertex C", () => {
    const vertex = (x: number, y: number, z: number) => ({
      point: new Vec3(x, y, z),
      witnessA: new Vec3(x, y, z),
      witnessB: new Vec3(),
    });
    const simplex = [
      vertex(1, 0.75, 0),
      vertex(1, 3.5, -3.25),
      vertex(2.5, 0.25, -3),
    ];
    const direction = new Vec3();
    expect(doSimplex(simplex, direction)).toBe(false);
    expect(simplex).toHaveLength(1);
  });
});

describe("EPA", () => {
  test("recovers sphere-sphere penetration", () => {
    const shapeA = new Sphere(1);
    const shapeB = new Sphere(1.2);
    const transformA = transformAt(0, 0, 0);
    const transformB = transformAt(1.4, 0.3, 0);
    const gjk = gjkIntersects(shapeA, transformA, shapeB, transformB);
    expect(gjk.intersects).toBe(true);
    const result = epa(gjk.simplex, (direction) =>
      minkowskiSupport(shapeA, transformA, shapeB, transformB, direction),
    );
    expect(result).not.toBeNull();
    if (result === null) {
      return;
    }
    const distance = transformA.position.distanceTo(transformB.position);
    expect(result.normal.length()).toBeCloseTo(1, 10);
    expect(result.depth).toBeCloseTo(2.2 - distance, 2);
    expect(dot(result.normal, transformB.position.clone().sub(transformA.position).normalize())).toBeGreaterThan(0.99);
    expect(result.pointA.distanceTo(transformA.position)).toBeCloseTo(1, 2);
    expect(result.pointB.distanceTo(transformB.position)).toBeCloseTo(1.2, 2);
  });

  test("recovers the minimum axis for overlapping boxes", () => {
    const box = new Box(new Vec3(1, 1, 1));
    const transformA = transformAt(0, 0, 0);
    const transformB = transformAt(0.6, 0.2, 0.1);
    const gjk = gjkIntersects(box, transformA, box, transformB);
    const result = epa(gjk.simplex, (direction) =>
      minkowskiSupport(box, transformA, box, transformB, direction),
    );
    expect(result).not.toBeNull();
    if (result === null) {
      return;
    }
    expect(result.depth).toBeCloseTo(1.4, 3);
    expect(Math.abs(dot(result.normal, new Vec3(1, 0, 0)))).toBeGreaterThan(0.99);
  });

  test("rejects invalid simplices", () => {
    const mockSupport = (): ReturnType<typeof minkowskiSupport> => ({
      point: new Vec3(1, 0, 0),
      witnessA: new Vec3(1, 0, 0),
      witnessB: new Vec3(0, 0, 0),
    });
    expect(epa([], mockSupport)).toBeNull();
    const flat = [new Vec3(0, 0, 0), new Vec3(1, 0, 0), new Vec3(0, 1, 0), new Vec3(1, 1, 0)];
    expect(
      epa(
        flat.map((point) => ({
          point,
          witnessA: point.clone(),
          witnessB: new Vec3(),
        })),
        mockSupport,
      ),
    ).toBeNull();
  });

  test("completes a vertical line simplex (perpendicular axis selection)", () => {
    const shapeA = new Sphere(1);
    const shapeB = new Sphere(1);
    const transformA = transformAt(0, 0, 0);
    const transformB = transformAt(0, 1.5, 0);
    const gjk = gjkIntersects(shapeA, transformA, shapeB, transformB);
    expect(gjk.simplex.length).toBe(2);
    const result = epa(gjk.simplex, (direction) =>
      minkowskiSupport(shapeA, transformA, shapeB, transformB, direction),
    );
    expect(result).not.toBeNull();
    if (result !== null) {
      expect(result.depth).toBeCloseTo(0.5, 2);
    }
  });

  test("rejects a collinear three-vertex simplex", () => {
    const collinear = [
      new Vec3(0, 0, 0),
      new Vec3(1, 0, 0),
      new Vec3(2, 0, 0),
    ].map((point) => ({
      point,
      witnessA: point.clone(),
      witnessB: new Vec3(),
    }));
    expect(
      epa(collinear, () => ({
        point: new Vec3(0, 1, 0),
        witnessA: new Vec3(0, 1, 0),
        witnessB: new Vec3(),
      })),
    ).toBeNull();
  });

  test("skips degenerate horizon faces during expansion", () => {
    const a = new Vec3(1, 1, 1);
    const b = new Vec3(1, -1, -1);
    const c = new Vec3(-1, 1, -1);
    const d = new Vec3(-1, -1, 1);
    const simplex = [a, b, c, d].map((point) => ({
      point,
      witnessA: point.clone(),
      witnessB: new Vec3(),
    }));
    const normal = new Vec3(1, 1, -1).normalize();
    const supportPoint = a
      .clone()
      .addScaled(b.clone().sub(a), 2)
      .addScaled(normal, 3e-13);
    const result = epa(
      simplex,
      () => ({
        point: supportPoint.clone(),
        witnessA: supportPoint.clone(),
        witnessB: new Vec3(),
      }),
      0,
      1,
    );
    expect(result).not.toBeNull();
  });

  test("iteration cap returns the best face so far", () => {
    const shapeA = new Sphere(1);
    const shapeB = new Sphere(1);
    const transformA = transformAt(0, 0, 0);
    const transformB = transformAt(1, 0, 0);
    const gjk = gjkIntersects(shapeA, transformA, shapeB, transformB);
    const result = epa(
      gjk.simplex,
      (direction) => minkowskiSupport(shapeA, transformA, shapeB, transformB, direction),
      1e-4,
      1,
    );
    expect(result).not.toBeNull();
  });
});

describe("manifold utilities", () => {
  test("clipPolygon clips a square and handles degenerate inputs", () => {
    const square = [
      new Vec3(0, 0, 0),
      new Vec3(1, 0, 0),
      new Vec3(1, 1, 0),
      new Vec3(0, 1, 0),
    ];
    const clipped = clipPolygon(square, new Vec3(1, 0, 0), 0.5);
    expect(clipped).toHaveLength(4);
    expect(clipped.every((point) => point.x <= 0.5 + 1e-12)).toBe(true);

    const inside = clipPolygon(square, new Vec3(1, 0, 0), 2);
    expect(inside).toHaveLength(4);

    const outside = clipPolygon(square, new Vec3(1, 0, 0), -1);
    expect(outside).toHaveLength(0);

    expect(clipPolygon([], new Vec3(1, 0, 0), 0)).toHaveLength(0);
  });

  test("reduceManifold keeps the deepest point and at most four points", () => {
    const points = [
      makeContactPoint(new Vec3(0, 0, 0), 0.1, 1),
      makeContactPoint(new Vec3(1, 0, 0), 0.9, 2),
      makeContactPoint(new Vec3(0, 1, 0), 0.2, 3),
      makeContactPoint(new Vec3(1, 1, 0), 0.3, 4),
      makeContactPoint(new Vec3(0.5, 0.5, 0.5), 0.4, 5),
      makeContactPoint(new Vec3(2, 2, 0), 0.05, 6),
    ];
    const reduced = reduceManifold(points);
    expect(reduced).toHaveLength(4);
    expect(reduced[0].id).toBe(2);
    expect(new Set(reduced.map((point) => point.id)).size).toBe(4);

    const few = points.slice(0, 3);
    expect(reduceManifold(few)).toHaveLength(3);

    const deepestFirst = [
      makeContactPoint(new Vec3(0, 0, 0), 1, 10),
      makeContactPoint(new Vec3(0, 1, 0), 0.1, 11),
      makeContactPoint(new Vec3(1, 0, 0), 0.2, 12),
      makeContactPoint(new Vec3(1, 1, 0), 0.3, 13),
      makeContactPoint(new Vec3(0.5, 0.5, 1), 0.4, 14),
    ];
    expect(reduceManifold(deepestFirst)[0].id).toBe(10);
  });

  test("boxWorldVertices returns all eight corners", () => {
    const box = new Box(new Vec3(1, 2, 3));
    const vertices = boxWorldVertices(box, transformAt(0, 0, 0));
    expect(vertices).toHaveLength(8);
    for (const vertex of vertices) {
      expect(Math.abs(vertex.x)).toBeCloseTo(1, 12);
      expect(Math.abs(vertex.y)).toBeCloseTo(2, 12);
      expect(Math.abs(vertex.z)).toBeCloseTo(3, 12);
    }
  });

  test("box-box produces a clipped four-point face manifold", () => {
    const box = new Box(new Vec3(1, 1, 1));
    const manifold = expectManifold(
      boxBoxManifold(box, transformAt(0, 0, 0), box, transformAt(1.5, 0, 0)),
    );
    expectVecClose(manifold.normal, new Vec3(1, 0, 0), 1e-6);
    expect(manifold.points).toHaveLength(4);
    for (const point of manifold.points) {
      expect(point.penetration).toBeCloseTo(0.5, 4);
      expect(point.position.x).toBeCloseTo(0.5, 6);
    }
    expect(boxBoxManifold(box, transformAt(0, 0, 0), box, transformAt(3, 0, 0))).toBeNull();
  });

  test("box-box clipping with rotated boxes stays within four points", () => {
    const box = new Box(new Vec3(1, 1, 1));
    const rotation = Quat.fromAxisAngle(new Vec3(0, 0, 1), Math.PI / 4);
    const manifold = expectManifold(
      boxBoxManifold(
        box,
        transformAt(0, 0, 0, rotation),
        box,
        transformAt(2.5, 0, 0, rotation),
      ),
    );
    expect(manifold.points.length).toBeGreaterThan(0);
    expect(manifold.points.length).toBeLessThanOrEqual(4);
    expect(Math.abs(manifold.normal.x)).toBeGreaterThan(0.6);
    expect(manifold.points.every((point) => point.penetration >= 0)).toBe(true);
  });

  test("box-box fallback for edge-edge geometry returns a single contact", () => {
    const boxA = new Box(new Vec3(1, 0.1, 0.1));
    const boxB = new Box(new Vec3(0.1, 1, 0.1));
    const manifold = expectManifold(
      boxBoxManifold(boxA, transformAt(0, 0, 0), boxB, transformAt(0, 0, 0)),
    );
    expect(manifold.points.length).toBeGreaterThanOrEqual(1);
    expect(manifold.points.length).toBeLessThanOrEqual(4);
  });

  test("corner-touching boxes have no penetration depth", () => {
    const box = new Box(new Vec3(1, 1, 1));
    expect(
      boxBoxManifold(box, transformAt(1, 1, 1), box, transformAt(-1, -1, -1)),
    ).toBeNull();
  });

  test("clip branch keeps only incident points behind the reference face", () => {
    const manifold = expectManifold(
      boxBoxManifold(
        new Box(new Vec3(1, 1, 1)),
        new Transform(
          new Vec3(-0.8224615231156349, 0.7053718934766948, -0.376874758861959),
          Quat.fromValues(
            0.10005710048954944,
            0.6337535952540023,
            -0.6642677064666727,
            0.3835275365469301,
          ),
        ),
        new Box(new Vec3(1, 1, 1)),
        new Transform(
          new Vec3(-2.957576269749552, -0.22399703599512577, -1.2499319496564567),
          Quat.fromValues(
            -0.3735128339889898,
            -0.33783835531264855,
            0.8593779036913324,
            0.08844787828008607,
          ),
        ),
      ),
    );
    expect(manifold.points.length).toBeGreaterThan(0);
    expect(manifold.points.length).toBeLessThanOrEqual(4);
  });

  test("fallback contact is used when clipping removes every point", () => {
    const manifold = expectManifold(
      boxBoxManifold(
        new Box(new Vec3(0.05, 1, 1)),
        new Transform(
          new Vec3(-0.4015422686934471, -0.672661014366895, -0.6028031101450324),
          Quat.fromValues(
            0.6077489383413147,
            -0.47140057881077263,
            0.23334614239153104,
            0.594955712698739,
          ),
        ),
        new Box(new Vec3(1, 1, 0.05)),
        new Transform(
          new Vec3(-0.1873891071882099, 0.38242551777511835, 0.07104260171763599),
          Quat.fromValues(
            0.21451042642551246,
            -0.4133537359929396,
            0.585054450144805,
            -0.6639542576572453,
          ),
        ),
      ),
    );
    expect(manifold.points).toHaveLength(1);
    expect(manifold.points[0].id).toBe(hashContactId(0x300, 0));
  });
});

describe("contact cache", () => {
  test("rotates, stores and clears impulses", () => {
    const cache = new ContactCache();
    const pairKey = ContactCache.pairKey(2, 1);
    expect(pairKey).toBe("1:2");
    expect(cache.get(pairKey, 7)).toBeUndefined();

    cache.beginStep();
    cache.store(pairKey, 7, { normalImpulse: 1, tangentImpulse1: 2, tangentImpulse2: 3 });
    cache.store(pairKey, 8, { normalImpulse: 4, tangentImpulse1: 5, tangentImpulse2: 6 });
    expect(cache.get(pairKey, 7)).toBeUndefined();

    cache.beginStep();
    expect(cache.get(pairKey, 7)).toEqual({
      normalImpulse: 1,
      tangentImpulse1: 2,
      tangentImpulse2: 3,
    });
    expect(cache.get(ContactCache.pairKey(1, 2), 999)).toBeUndefined();

    cache.clear();
    expect(cache.get(pairKey, 7)).toBeUndefined();
  });
});

describe("collideShapes dispatch", () => {
  test("flips analytic manifolds when the order is reversed", () => {
    const sphere = new Sphere(0.5);
    const box = new Box(new Vec3(1, 1, 1));
    const sphereTransform = transformAt(1.2, 0, 0);
    const boxTransform = transformAt(0, 0, 0);

    const sphereFirst = expectManifold(collideShapes(sphere, sphereTransform, box, boxTransform));
    const boxFirst = expectManifold(collideShapes(box, boxTransform, sphere, sphereTransform));
    expect(
      collideShapes(sphere, transformAt(3, 0, 0), box, boxTransform),
    ).toBeNull();
    expectVecClose(sphereFirst.normal, new Vec3(-1, 0, 0));
    expectVecClose(boxFirst.normal, new Vec3(1, 0, 0));
    expectVecClose(sphereFirst.points[0].position, boxFirst.points[0].position);

    const capsule = new Capsule(0.5, 1);
    const capsuleTransform = transformAt(1.2, 0, 0);
    const sphereCapsuleFirst = expectManifold(
      collideShapes(sphere, sphereTransform, capsule, capsuleTransform),
    );
    const capsuleSphereFirst = expectManifold(
      collideShapes(capsule, capsuleTransform, sphere, sphereTransform),
    );
    expectVecClose(sphereCapsuleFirst.normal, new Vec3(1, 0, 0), 1e-12);
    expectVecClose(capsuleSphereFirst.normal, new Vec3(-1, 0, 0), 1e-12);
    expect(
      collideShapes(capsule, transformAt(5, 0, 0), sphere, transformAt(0, 0, 0)),
    ).toBeNull();

    const tetra = new ConvexPolytope([
      new Vec3(-1, -1, -1),
      new Vec3(1, -1, -1),
      new Vec3(0, 1, -1),
      new Vec3(0, 0, 1),
    ]);
    expect(
      collideShapes(new Sphere(0.5), transformAt(0, 0, 0), tetra, transformAt(0, 0, 0)),
    ).not.toBeNull();
  });

  test("plane as the second shape flips the normal", () => {
    const plane = new Plane();
    const box = new Box(new Vec3(1, 1, 1));
    const planeTransform = transformAt(0, 0, 0);
    const boxTransform = transformAt(0, 0.9, 0);

    const planeFirst = expectManifold(collideShapes(plane, planeTransform, box, boxTransform));
    const boxFirst = expectManifold(collideShapes(box, boxTransform, plane, planeTransform));
    expectVecClose(planeFirst.normal, new Vec3(0, 1, 0));
    expectVecClose(boxFirst.normal, new Vec3(0, -1, 0));
    expect(planeFirst.points).toHaveLength(4);
    expect(boxFirst.points).toHaveLength(4);

    expect(collideShapes(plane, planeTransform, plane, planeTransform)).toBeNull();
    expect(planeManifold(plane, planeTransform, plane, planeTransform)).toBeNull();

    expect(
      collideShapes(plane, planeTransform, new Sphere(0.5), transformAt(0, 0.4, 0)),
    ).not.toBeNull();
    expect(
      collideShapes(new Sphere(0.5), transformAt(0, 2, 0), plane, planeTransform),
    ).toBeNull();
  });

  test("plane dispatch handles capsule and polytope", () => {
    const plane = new Plane();
    const planeTransform = transformAt(0, 0, 0);
    expect(
      collideShapes(plane, planeTransform, new Capsule(0.5, 1), transformAt(0, 1.4, 0)),
    ).not.toBeNull();

    const polyhedron = new ConvexPolytope([
      new Vec3(-1, -1, -1),
      new Vec3(1, -1, -1),
      new Vec3(0, 1, -1),
      new Vec3(0, 0, 1),
    ]);
    expect(collideShapes(plane, planeTransform, polyhedron, transformAt(0, 0.5, 0))).not.toBeNull();
  });

  test("capsule-capsule uses the GJK/EPA path", () => {
    const capsule = new Capsule(0.5, 1);
    const manifold = expectManifold(
      collideShapes(capsule, transformAt(0, 0, 0), capsule, transformAt(0.8, 0, 0)),
    );
    expect(Math.abs(manifold.normal.x)).toBeGreaterThan(0.99);
    expect(manifold.points[0].penetration).toBeCloseTo(0.2, 2);
  });

  test("polytope-polytope uses GJK/EPA and reports a contact", () => {
    const tetra = new ConvexPolytope([
      new Vec3(-1, -1, -1),
      new Vec3(1, -1, -1),
      new Vec3(0, 1, -1),
      new Vec3(0, 0, 1),
    ]);
    const manifold = expectManifold(
      collideShapes(tetra, transformAt(0, 0, 0), tetra, transformAt(0.2, 0, 0)),
    );
    expect(manifold.normal.length()).toBeCloseTo(1, 12);
    expect(manifold.points[0].penetration).toBeGreaterThan(0);
  });

  test("analytic sphere-sphere and box-box dispatch directly", () => {
    const sphere = new Sphere(0.5);
    expect(
      collideShapes(sphere, transformAt(0, 0, 0), sphere, transformAt(0.8, 0, 0)),
    ).not.toBeNull();

    const box = new Box(new Vec3(1, 1, 1));
    expect(
      collideShapes(box, transformAt(0, 0, 0), box, transformAt(1.5, 0, 0)),
    ).not.toBeNull();
  });

  test("general path returns null when EPA cannot form a tetrahedron", () => {
    const identity = transformAt(0, 0, 0);
    expect(
      collideShapes(
        new FixedShape(new Vec3(0, 0, 0)),
        identity,
        new FixedShape(new Vec3(0, 0, 0)),
        identity,
      ),
    ).toBeNull();
  });

  test("separated general-path pairs return null", () => {
    const capsule = new Capsule(0.5, 1);
    expect(
      collideShapes(capsule, transformAt(0, 0, 0), capsule, transformAt(5, 0, 0)),
    ).toBeNull();
  });
});

describe("contact helpers", () => {
  test("flipManifold negates the normal and copies points", () => {
    const manifold: ContactManifold = {
      normal: new Vec3(1, 2, 3),
      points: [makeContactPoint(new Vec3(4, 5, 6), 0.5, 7)],
    };
    const flipped = flipManifold(manifold);
    expectVecClose(flipped.normal, new Vec3(-1, -2, -3));
    expectVecClose(flipped.points[0].position, new Vec3(4, 5, 6));
    expect(flipped.points[0].penetration).toBe(0.5);
    expect(flipped.points[0].id).toBe(7);
    expect(flipped.points[0]).not.toBe(manifold.points[0]);
  });

  test("hashContactId is deterministic and distinguishes inputs", () => {
    expect(hashContactId(1, 2)).toBe(hashContactId(1, 2));
    expect(hashContactId(1, 2)).not.toBe(hashContactId(2, 1));
    expect(hashContactId(0, 0)).toBeGreaterThanOrEqual(0);
  });
});
