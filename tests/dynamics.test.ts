import { describe, expect, test } from "vitest";
import { ContactCache } from "../src/collision/manifold/ContactManifold.js";
import { Box } from "../src/collision/shapes/Box.js";
import { Capsule } from "../src/collision/shapes/Capsule.js";
import { ConvexPolytope } from "../src/collision/shapes/ConvexPolytope.js";
import { Plane } from "../src/collision/shapes/Plane.js";
import { Shape, ShapeType } from "../src/collision/shapes/Shape.js";
import { Sphere } from "../src/collision/shapes/Sphere.js";
import { ContactSolver } from "../src/dynamics/ContactSolver.js";
import type { ContactPair } from "../src/dynamics/ContactSolver.js";
import { ConstraintRow } from "../src/dynamics/ConstraintRow.js";
import { BallSocketJoint, DistanceJoint, HingeJoint } from "../src/dynamics/Joint.js";
import { BodyType, RigidBody, computeLocalInertia } from "../src/dynamics/RigidBody.js";
import { World } from "../src/dynamics/World.js";
import type { WorldOptions } from "../src/dynamics/World.js";
import { Mat3 } from "../src/math/Mat3.js";
import { Quat } from "../src/math/Quat.js";
import { Vec3 } from "../src/math/Vec3.js";

const DT = 1 / 60;

/** Steps a world `steps` times at the fixed test rate. */
function stepN(world: World, steps: number, dt = DT): void {
  for (let i = 0; i < steps; i++) {
    world.step(dt);
  }
}

/** Adds an infinite static ground plane and returns it. */
function addGround(world: World, friction = 0.5, restitution = 0): RigidBody {
  return world.addBody(
    new RigidBody({ shape: new Plane(), type: BodyType.Static, friction, restitution }),
  );
}

/** Builds a `RigidBody` message matcher without re-typing the prefix. */
function RigidBodyError(fragment: string): RegExp {
  return new RegExp(`RigidBody: .*${fragment}`);
}

/** World-space +Y axis of a body (its "up" direction). */
function worldUpAxis(body: RigidBody): Vec3 {
  return body.transform.transformDirection(new Vec3(), new Vec3(0, 1, 0));
}

/** Regular tetrahedron used as a minimal convex polytope fixture. */
function tetrahedron(): ConvexPolytope {
  return new ConvexPolytope([
    new Vec3(0, 0, 0),
    new Vec3(1, 0, 0),
    new Vec3(0, 1, 0),
    new Vec3(0, 0, 1),
  ]);
}

describe("RigidBody", () => {
  test("derives mass, inverse mass and inertia from the shape", () => {
    const sphere = new RigidBody({ shape: new Sphere(0.5), mass: 2 });
    expect(sphere.inverseMass).toBeCloseTo(0.5, 12);
    expect(sphere.localInertia.m00).toBeCloseTo((2 / 5) * 2 * 0.25, 12);
    expect(sphere.localInertia.m11).toBeCloseTo(sphere.localInertia.m00, 12);
    expect(sphere.localInertia.m22).toBeCloseTo(sphere.localInertia.m00, 12);

    const box = new RigidBody({ shape: new Box(new Vec3(1, 1, 1)), mass: 2 });
    expect(box.localInertia.m00).toBeCloseTo((1 / 12) * 2 * (4 + 4), 12);

    const capsule = new RigidBody({ shape: new Capsule(0.25, 0.5), mass: 1.5 });
    expect(capsule.localInertia.m11).toBeGreaterThan(0);

    const polytope = new RigidBody({ shape: tetrahedron(), mass: 3 });
    expect(polytope.localInertia.m00).toBeGreaterThan(0);

    expect(computeLocalInertia(new Sphere(1), 1).m00).toBeCloseTo(0.4, 12);
    expect(() => computeLocalInertia(new Plane(), 1)).toThrow(RangeError);
  });

  test("rejects zero-volume polytopes and singular inertia tensors", () => {
    const flat = new ConvexPolytope([
      new Vec3(0, 0, 0),
      new Vec3(1, 0, 0),
      new Vec3(0, 1, 0),
      new Vec3(1, 1, 0),
    ]);
    expect(() => computeLocalInertia(flat, 1)).toThrow(RangeError);

    // A stub shape whose inertia tensor cannot be inverted must be rejected
    // rather than silently producing NaN accelerations.
    const singular = {
      type: ShapeType.Box,
      computeLocalInertia: (): Mat3 => Mat3.zero(new Mat3()),
    } as unknown as Shape;
    expect(() => new RigidBody({ shape: singular, mass: 1 })).toThrow(RangeError);
  });

  test("static and kinematic bodies have infinite mass", () => {
    const plane = new RigidBody({ shape: new Plane(), type: BodyType.Static });
    expect(plane.type).toBe(BodyType.Static);
    expect(plane.inverseMass).toBe(0);
    expect(plane.isDynamic).toBe(false);
    expect(plane.isKinematic).toBe(false);
    expect(plane.isAwake).toBe(false);
    expect(plane.isSleeping).toBe(false);
    expect(plane.kineticEnergy()).toBe(0);
    expect(plane.worldInertia.m00).toBe(0);

    const kinematic = new RigidBody({
      shape: new Sphere(0.5),
      type: BodyType.Kinematic,
      position: new Vec3(0, 0, 0),
      linearVelocity: new Vec3(2, 0, 0),
      angularVelocity: new Vec3(0, 0, 1),
    });
    expect(kinematic.isKinematic).toBe(true);
    expect(kinematic.isAwake).toBe(true);
    expect(kinematic.isSleeping).toBe(false);
    expect(kinematic.inverseMass).toBe(0);
  });

  test("validates constructor options", () => {
    expect(() => new RigidBody(undefined as never)).toThrow(RangeError);
    expect(() => new RigidBody(7 as never)).toThrow(RangeError);
    expect(() => new RigidBody({} as never)).toThrow(RigidBodyError("shape"));
    expect(() => new RigidBody({ shape: new Sphere(0.5), mass: -1 })).toThrow(
      RigidBodyError("mass"),
    );
    expect(() => new RigidBody({ shape: new Sphere(0.5), mass: Number.NaN })).toThrow(
      RigidBodyError("mass"),
    );
    expect(
      new RigidBody({ shape: new Sphere(0.5), mass: 0 }).type,
    ).toBe(BodyType.Static);
    expect(
      () => new RigidBody({ shape: new Sphere(0.5), mass: 0, type: BodyType.Dynamic }),
    ).toThrow(RangeError);
    expect(() => new RigidBody({ shape: new Sphere(0.5), type: (7 as number) as BodyType })).toThrow(
      RigidBodyError("unknown body type"),
    );
    expect(() => new RigidBody({ shape: new Plane(), mass: 1 })).toThrow(
      RigidBodyError("plane"),
    );

    for (const value of [Number.NaN, -1]) {
      expect(() => new RigidBody({ shape: new Sphere(0.5), friction: value })).toThrow(
        RigidBodyError("friction"),
      );
      expect(
        () => new RigidBody({ shape: new Sphere(0.5), restitution: value }),
      ).toThrow(RigidBodyError("restitution"));
      expect(
        () => new RigidBody({ shape: new Sphere(0.5), linearDamping: value }),
      ).toThrow(RigidBodyError("linearDamping"));
      expect(
        () => new RigidBody({ shape: new Sphere(0.5), angularDamping: value }),
      ).toThrow(RigidBodyError("angularDamping"));
    }
  });

  test("normalizes the initial rotation and defaults optional fields", () => {
    const body = new RigidBody({
      shape: new Sphere(0.5),
      rotation: Quat.fromValues(2, 0, 0, 0),
      userData: { tag: "probe" },
    });
    expect(body.rotation.length()).toBeCloseTo(1, 12);
    expect(body.friction).toBe(0.5);
    expect(body.restitution).toBe(0);
    expect(body.linearDamping).toBe(0);
    expect(body.angularDamping).toBe(0);
    expect(body.userData).toEqual({ tag: "probe" });
    expect(body.sleepTime).toBe(0);

    const bare = new RigidBody({ shape: new Sphere(0.5) });
    expect(bare.position.lengthSq()).toBe(0);
    expect(bare.linearVelocity.lengthSq()).toBe(0);
    expect(bare.userData).toBeNull();
  });

  test("ignores force and impulse APIs on non-dynamic bodies", () => {
    const body = new RigidBody({ shape: new Sphere(0.5), type: BodyType.Kinematic });
    body.applyForce(new Vec3(1, 0, 0));
    body.applyForce(new Vec3(1, 0, 0), new Vec3(0, 1, 0));
    body.applyTorque(new Vec3(0, 1, 0));
    body.applyImpulse(new Vec3(1, 0, 0));
    body.applyImpulse(new Vec3(1, 0, 0), new Vec3(0, 1, 0));
    body.applyAngularImpulse(new Vec3(0, 1, 0));
    body.wake();
    body.sleep();
    expect(body.force.lengthSq()).toBe(0);
    expect(body.torque.lengthSq()).toBe(0);
    expect(body.linearVelocity.lengthSq()).toBe(0);
    expect(body.isAwake).toBe(true);
  });

  test("applies forces, torques and impulses", () => {
    const body = new RigidBody({ shape: new Sphere(0.5), mass: 2 });
    body.applyForce(new Vec3(0, 4, 0));
    expect(body.force.y).toBe(4);
    body.applyForce(new Vec3(0, 0, 2), new Vec3(1, 0, 0));
    expect(body.torque.y).toBe(-2);
    body.applyTorque(new Vec3(0, 0, 1));
    expect(body.torque.z).toBe(1);
    body.integrateForces(0.5, new Vec3(0, -10, 0));
    // v = gravity·dt + F/m·dt = -5 + 4/2·0.5 = -4
    expect(body.linearVelocity.y).toBeCloseTo(-4, 12);
    body.clearForces();
    expect(body.force.lengthSq()).toBe(0);
    expect(body.torque.lengthSq()).toBe(0);

    // Off-center impulse: linear plus angular response.
    const pushed = new RigidBody({ shape: new Sphere(0.5), mass: 1 });
    pushed.applyImpulse(new Vec3(0, 0, 4), new Vec3(1, 0, 0));
    expect(pushed.linearVelocity.z).toBeCloseTo(4, 12);
    expect(pushed.angularVelocity.y).toBeCloseTo(-40, 9);

    const shoved = new RigidBody({ shape: new Sphere(0.5), mass: 1 });
    shoved.applyImpulse(new Vec3(0, 0, 4));
    expect(shoved.linearVelocity.z).toBeCloseTo(4, 12);
    expect(shoved.angularVelocity.lengthSq()).toBe(0);

    const spun = new RigidBody({ shape: new Sphere(0.5), mass: 1 });
    spun.applyAngularImpulse(new Vec3(0, 0.4, 0));
    expect(spun.angularVelocity.y).toBeCloseTo(4, 12);
  });

  test("applies damping to velocities during force integration", () => {
    const body = new RigidBody({
      shape: new Sphere(0.5),
      mass: 1,
      linearDamping: 1,
      angularDamping: 2,
      linearVelocity: new Vec3(1, 0, 0),
      angularVelocity: new Vec3(0, 1, 0),
    });
    body.integrateForces(0.5, new Vec3());
    expect(body.linearVelocity.x).toBeCloseTo(1 / 1.5, 12);
    expect(body.angularVelocity.y).toBeCloseTo(0.5, 12);
  });

  test("integrates positions and pseudo velocities", () => {
    const body = new RigidBody({
      shape: new Sphere(0.5),
      position: new Vec3(0, 0, 0),
    });
    body.pseudoLinearVelocity.set(2, 0, 0);
    body.pseudoAngularVelocity.set(0, 0, 1);
    body.integrateVelocity(0.5);
    expect(body.position.x).toBeCloseTo(1, 12);
    expect(body.pseudoLinearVelocity.lengthSq()).toBe(0);
    expect(body.pseudoAngularVelocity.lengthSq()).toBe(0);

    const kinematic = new RigidBody({
      shape: new Sphere(0.5),
      type: BodyType.Kinematic,
      linearVelocity: new Vec3(1, 0, 0),
    });
    kinematic.integrateForces(0.5, new Vec3(0, -9.81, 0));
    expect(kinematic.linearVelocity.y).toBe(0);
    kinematic.integrateVelocity(0.5);
    expect(kinematic.position.x).toBeCloseTo(0.5, 12);

    const sleeper = new RigidBody({ shape: new Sphere(0.5), position: new Vec3(0, 5, 0) });
    sleeper.linearVelocity.set(0, 1, 0);
    sleeper.sleep();
    sleeper.integrateForces(0.5, new Vec3(0, -9.81, 0));
    sleeper.integrateVelocity(0.5);
    expect(sleeper.position.y).toBeCloseTo(5, 12);
    expect(sleeper.linearVelocity.lengthSq()).toBe(0);
  });

  test("reports velocity at a world point and kinetic energy", () => {
    const body = new RigidBody({ shape: new Sphere(0.5), mass: 2 });
    body.linearVelocity.set(1, 0, 0);
    body.angularVelocity.set(0, 0, 2);
    const velocity = body.getVelocityAtPoint(new Vec3(0, 1, 0));
    expect(velocity.x).toBeCloseTo(-1, 12);
    expect(velocity.y).toBeCloseTo(0, 12);

    expect(body.kineticEnergy()).toBeCloseTo(0.5 * 2 * 1 + 0.5 * 0.2 * 4, 9);
    body.angularVelocity.setZero();
    body.linearVelocity.set(3, 4, 0);
    expect(body.kineticEnergy()).toBeCloseTo(25, 9);
  });

  test("sleeps, wakes and computes AABBs", () => {
    const body = new RigidBody({
      shape: new Sphere(0.5),
      mass: 1,
      position: new Vec3(1, 2, 3),
      linearVelocity: new Vec3(1, 0, 0),
      angularVelocity: new Vec3(0, 1, 0),
    });
    const aabb = body.computeAABB();
    expect(aabb.min.x).toBeCloseTo(0.5, 12);
    expect(aabb.max.y).toBeCloseTo(2.5, 12);

    body.pseudoLinearVelocity.set(1, 0, 0);
    body.pseudoAngularVelocity.set(0, 1, 0);
    body.sleep();
    expect(body.isSleeping).toBe(true);
    expect(body.linearVelocity.lengthSq()).toBe(0);
    expect(body.angularVelocity.lengthSq()).toBe(0);
    expect(body.pseudoLinearVelocity.lengthSq()).toBe(0);
    expect(body.pseudoAngularVelocity.lengthSq()).toBe(0);

    // Waking an already-awake body is a no-op and keeps the sleep timer.
    body.sleepTime = 0.4;
    body.wake();
    expect(body.isAwake).toBe(true);
    expect(body.sleepTime).toBe(0);
    body.sleepTime = 0.4;
    body.wake();
    expect(body.sleepTime).toBe(0.4);
  });
});

describe("ConstraintRow", () => {
  test("computes effective mass and clamps accumulated impulses", () => {
    const bodyA = new RigidBody({ shape: new Sphere(0.5), mass: 2 });
    const bodyB = new RigidBody({ shape: new Sphere(0.5), mass: 1 });
    const row = new ConstraintRow(
      bodyA,
      bodyB,
      new Vec3(1, 0, 0),
      new Vec3(),
      new Vec3(-1, 0, 0),
      new Vec3(),
    );
    bodyA.linearVelocity.set(1, 0, 0);
    bodyB.linearVelocity.set(-1, 0, 0);
    row.prepare();
    expect(row.effectiveMass).toBeCloseTo(1 / 1.5, 12);
    expect(row.relativeVelocity()).toBeCloseTo(2, 12);

    // Signorini-style lower clamp: the negative delta is discarded.
    row.lowerBound = 0;
    expect(row.solve()).toBe(0);
    expect(row.impulse).toBe(0);
    expect(row.solve()).toBe(0);

    // Unclamped solve converges the relative velocity to the bias.
    row.lowerBound = -10;
    row.upperBound = 10;
    expect(row.solve()).toBeCloseTo(-4 / 3, 12);
    expect(row.impulse).toBeCloseTo(-4 / 3, 12);
    expect(row.relativeVelocity()).toBeCloseTo(0, 12);
    expect(row.solve()).toBeCloseTo(0, 12);

    const beforeA = bodyA.linearVelocity.x;
    row.warmStart();
    expect(bodyA.linearVelocity.x).toBeLessThan(beforeA);
    const warmed = bodyA.linearVelocity.x;
    row.impulse = 0;
    row.warmStart();
    expect(bodyA.linearVelocity.x).toBe(warmed);

    const pseudo = new ConstraintRow(
      bodyA,
      bodyB,
      new Vec3(1, 0, 0),
      new Vec3(),
      new Vec3(-1, 0, 0),
      new Vec3(),
    );
    pseudo.prepare();
    expect(pseudo.relativeVelocityPseudo()).toBe(0);
    pseudo.applyPseudoImpulse(2);
    expect(pseudo.relativeVelocityPseudo()).toBeCloseTo(3, 12);
  });

  test("reports zero effective mass for a fully static row", () => {
    const bodyA = new RigidBody({ shape: new Plane(), type: BodyType.Static });
    const bodyB = new RigidBody({ shape: new Plane(), type: BodyType.Static });
    const row = new ConstraintRow(
      bodyA,
      bodyB,
      new Vec3(1, 0, 0),
      new Vec3(1, 0, 0),
      new Vec3(1, 0, 0),
      new Vec3(1, 0, 0),
    );
    row.prepare();
    expect(row.effectiveMass).toBe(0);
    expect(row.solve()).toBe(0);
  });
});

describe("ContactSolver", () => {
  test("builds, warms, solves and stores a contact constraint", () => {
    const bodyA = new RigidBody({
      shape: new Sphere(0.5),
      mass: 1,
      position: new Vec3(0, 1.1, 0),
      linearVelocity: new Vec3(0, -1, 0),
    });
    const bodyB = new RigidBody({ shape: new Plane(), type: BodyType.Static });
    const pair: ContactPair = {
      bodyA,
      bodyB,
      manifold: {
        normal: new Vec3(0, -1, 0),
        points: [{ id: 5, position: new Vec3(0, 0.6, 0), penetration: 0.1 }],
      },
    };
    const solver = new ContactSolver();
    const cache = new ContactCache();
    solver.prepare([pair], DT, cache);
    expect(solver.constraints).toHaveLength(1);
    const constraint = solver.constraints[0];
    expect(constraint.pairKey).toBe(ContactCache.pairKey(bodyA.id, bodyB.id));
    expect(constraint.tangent1.dot(constraint.normal)).toBeCloseTo(0, 12);
    expect(constraint.tangent2.dot(constraint.normal)).toBeCloseTo(0, 12);

    solver.warmStart();
    solver.solve(4);
    solver.solvePositions(2);
    solver.store(cache);
    // `store` writes the current frame; `get` reads the previous frame.
    cache.beginStep();
    const cached = cache.get(constraint.pairKey, 5);
    expect(cached).toBeDefined();
    expect(cached!.normalImpulse).toBeGreaterThan(0);
    expect(bodyA.linearVelocity.y).toBeCloseTo(0, 6);

    // A second substep reads the cached impulse (warm start path).
    solver.prepare([pair], DT, cache);
    solver.warmStart();
    expect(solver.constraints[0].points[0].normalRow.impulse).toBeGreaterThan(0);

    solver.prepare([], DT, cache);
    expect(solver.constraints).toHaveLength(0);
  });

  test("picks a well-conditioned tangent basis for x-aligned normals", () => {
    const bodyA = new RigidBody({ shape: new Box(new Vec3(0.5, 0.5, 0.5)), mass: 1 });
    const bodyB = new RigidBody({ shape: new Box(new Vec3(0.5, 0.5, 0.5)), mass: 1 });
    const pair: ContactPair = {
      bodyA,
      bodyB,
      manifold: {
        normal: new Vec3(1, 0, 0),
        points: [{ id: 1, position: new Vec3(0, 0, 0), penetration: 0.01 }],
      },
    };
    const solver = new ContactSolver({ baumgarte: 0.1, penetrationSlop: 0.01, restitutionThreshold: 2 });
    solver.prepare([pair], DT, new ContactCache());
    const constraint = solver.constraints[0];
    expect(constraint.tangent1.dot(constraint.normal)).toBeCloseTo(0, 12);
    expect(constraint.tangent1.length()).toBeCloseTo(1, 12);
    expect(constraint.tangent2.dot(constraint.normal)).toBeCloseTo(0, 12);
    expect(constraint.tangent2.length()).toBeCloseTo(1, 12);
  });
});

describe("World", () => {
  test("validates constructor options", () => {
    expect(new World().substeps).toBe(8);
    expect(new World().solverIterations).toBe(10);
    const gravity = new Vec3(0, -1, 0);
    const custom = new World({
      gravity,
      solverIterations: 4,
      substeps: 2,
      positionIterations: 0,
      baumgarte: 0,
      penetrationSlop: 0,
      restitutionThreshold: 0,
      sleepEnergyThreshold: 0,
      sleepTime: 0,
      broadphaseMargin: 0,
      broadphaseVelocityScale: 0,
    });
    gravity.set(0, -99, 0);
    expect(custom.gravity.y).toBe(-1);
    expect(custom.substeps).toBe(2);
    expect(custom.positionIterations).toBe(0);
    expect(() => new World({ penetrationSlop: -1 })).toThrow(RangeError);

    const invalid: Array<Record<string, unknown>> = [
      { gravity: new Vec3(Number.NaN, 0, 0) },
      { solverIterations: Number.NaN },
      { solverIterations: 0 },
      { substeps: 1.5 },
      { substeps: 0 },
      { positionIterations: Number.NaN },
      { positionIterations: -1 },
      { baumgarte: Number.NaN },
      { baumgarte: -0.1 },
      { penetrationSlop: Number.NaN },
      { penetrationSlop: -0.1 },
      { restitutionThreshold: Number.NaN },
      { restitutionThreshold: -1 },
      { sleepEnergyThreshold: Number.NaN },
      { sleepEnergyThreshold: -1 },
      { sleepTime: Number.NaN },
      { sleepTime: -1 },
      { broadphaseMargin: Number.NaN },
      { broadphaseMargin: -1 },
      { broadphaseVelocityScale: Number.NaN },
      { broadphaseVelocityScale: -1 },
    ];
    for (const options of invalid) {
      expect(() => new World(options as WorldOptions)).toThrow(RangeError);
    }
  });

  test("validates step timestep", () => {
    const world = new World();
    expect(() => world.step(Number.NaN)).toThrow(RangeError);
    expect(() => world.step(0)).toThrow(RangeError);
    expect(() => world.step(-DT)).toThrow(RangeError);
  });

  test("adds, removes and validates bodies and joints", () => {
    const world = new World();
    const a = world.addBody(new RigidBody({ shape: new Sphere(0.5) }));
    expect(() => world.addBody(a)).toThrow(RangeError);
    const b = world.addBody(new RigidBody({ shape: new Sphere(0.5) }));
    const c = world.addBody(new RigidBody({ shape: new Sphere(0.5) }));

    const jointAB = world.addJoint(new DistanceJoint(a, b));
    const jointBC = world.addJoint(new DistanceJoint(b, c));
    world.addJoint(new DistanceJoint(c, a));
    expect(world.getStats().jointCount).toBe(3);
    expect(() => world.addJoint(jointAB)).toThrow(RangeError);

    const foreignA = new RigidBody({ shape: new Sphere(0.5) });
    expect(() => world.addJoint(new DistanceJoint(foreignA, b))).toThrow(RangeError);
    const foreignB = new RigidBody({ shape: new Sphere(0.5) });
    expect(() => world.addJoint(new DistanceJoint(b, foreignB))).toThrow(RangeError);

    const stranger = new RigidBody({ shape: new Sphere(0.5) });
    expect(world.removeBody(stranger)).toBe(false);
    expect(world.removeJoint(new DistanceJoint(a, c))).toBe(false);

    // Removing `a` must drop both of its joints and keep the unrelated one.
    expect(world.removeBody(a)).toBe(true);
    expect(world.joints).toEqual([jointBC]);
    expect(world.removeJoint(jointBC)).toBe(true);
    expect(world.getStats().jointCount).toBe(0);
    expect(world.getStats().bodyCount).toBe(2);
  });

  test("clears all state", () => {
    const world = new World();
    addGround(world);
    const box = world.addBody(
      new RigidBody({ shape: new Box(new Vec3(0.5, 0.5, 0.5)), position: new Vec3(0, 0.5, 0) }),
    );
    world.addJoint(new DistanceJoint(box, world.addBody(new RigidBody({ shape: new Sphere(0.5), position: new Vec3(3, 0.5, 0) }))));
    stepN(world, 5);
    world.clear();
    const stats = world.getStats();
    expect(stats.bodyCount).toBe(0);
    expect(stats.jointCount).toBe(0);
    expect(stats.stepCount).toBe(0);
    expect(stats.broadphasePairs).toBe(0);
    expect(stats.manifoldPairs).toBe(0);
    expect(stats.contactPoints).toBe(0);
    expect(stats.awakeBodies).toBe(0);
    expect(stats.sleepingBodies).toBe(0);

    // A cleared world accepts the same body instance again.
    expect(world.addBody(box)).toBe(box);
    world.step(DT);
    expect(world.getStats().bodyCount).toBe(1);
  });

  test("integrates gravity and settles a sphere on the ground", () => {
    const world = new World();
    addGround(world);
    const ball = world.addBody(
      new RigidBody({ shape: new Sphere(0.5), mass: 1, position: new Vec3(0, 2, 0) }),
    );
    stepN(world, 600);
    expect(ball.position.y).toBeGreaterThan(0.48);
    expect(ball.position.y).toBeLessThan(0.5);
    expect(ball.isSleeping).toBe(true);
    expect(world.getStats().sleepingBodies).toBe(1);
    expect(world.getStats().awakeBodies).toBe(0);
  });

  test("conserves linear momentum in a head-on elastic collision", () => {
    const world = new World({ gravity: new Vec3() });
    const left = world.addBody(
      new RigidBody({
        shape: new Sphere(0.5),
        mass: 1,
        restitution: 1,
        friction: 0,
        position: new Vec3(-2, 0, 0),
        linearVelocity: new Vec3(4, 0, 0),
      }),
    );
    const right = world.addBody(
      new RigidBody({
        shape: new Sphere(0.5),
        mass: 2,
        restitution: 1,
        friction: 0,
        position: new Vec3(2, 0, 0),
        linearVelocity: new Vec3(-2, 0, 0),
      }),
    );
    const momentum = new Vec3()
      .addScaled(left.linearVelocity, left.mass)
      .addScaled(right.linearVelocity, right.mass);
    expect(momentum.length()).toBeCloseTo(0, 12);
    stepN(world, 120);
    const afterMomentum = new Vec3()
      .addScaled(left.linearVelocity, left.mass)
      .addScaled(right.linearVelocity, right.mass);
    expect(afterMomentum.length()).toBeLessThan(1e-9);
    expect(left.linearVelocity.x).toBeCloseTo(-4, 6);
    expect(right.linearVelocity.x).toBeCloseTo(2, 6);
  });

  test("applies restitution above the closing-speed threshold", () => {
    const apexAfterImpact = (restitution: number): number => {
      const world = new World();
      addGround(world, 0.5, restitution);
      const ball = world.addBody(
        new RigidBody({
          shape: new Sphere(0.5),
          mass: 1,
          restitution,
          position: new Vec3(0, 2, 0),
          linearVelocity: new Vec3(0, -4, 0),
        }),
      );
      let apex = 0;
      for (let i = 0; i < 180; i++) {
        world.step(DT);
        if (i > 30) {
          apex = Math.max(apex, ball.position.y);
        }
      }
      return apex;
    };
    expect(apexAfterImpact(1)).toBeGreaterThan(2.2);
    expect(apexAfterImpact(0)).toBeLessThan(1);
  });

  test("holds static friction and lets kinetic friction slide", () => {
    const displacement = (friction: number): number => {
      const world = new World();
      const angle = (20 * Math.PI) / 180;
      const rotation = Quat.fromAxisAngle(new Vec3(0, 0, 1), angle);
      world.addBody(
        new RigidBody({ shape: new Plane(), type: BodyType.Static, rotation, friction }),
      );
      const box = world.addBody(
        new RigidBody({
          shape: new Box(new Vec3(0.4, 0.4, 0.4)),
          mass: 1,
          friction,
          rotation,
          position: new Vec3(0, 0.42, 0),
        }),
      );
      const start = box.position.clone();
      stepN(world, 240);
      return box.position.distanceTo(start);
    };
    expect(displacement(0.8)).toBeLessThan(0.01);
    expect(displacement(0.05)).toBeGreaterThan(5);
  });

  test("keeps a 20-box column upright for 3000 ticks and sleeps it", () => {
    const world = new World();
    addGround(world);
    const boxes: RigidBody[] = [];
    for (let i = 0; i < 20; i++) {
      boxes.push(
        world.addBody(
          new RigidBody({
            shape: new Box(new Vec3(0.5, 0.5, 0.5)),
            mass: 1,
            position: new Vec3(0, 0.5 + i, 0),
          }),
        ),
      );
    }
    let sleptAt = -1;
    for (let i = 0; i < 3000; i++) {
      world.step(DT);
      if (sleptAt < 0 && world.getStats().sleepingBodies === boxes.length) {
        sleptAt = i;
      }
    }
    expect(sleptAt).toBeGreaterThanOrEqual(0);
    for (const box of boxes) {
      expect(Math.abs(box.position.x)).toBeLessThan(0.1);
      expect(Math.abs(box.position.z)).toBeLessThan(0.1);
      expect(worldUpAxis(box).y).toBeGreaterThan(0.99);
    }
    expect(boxes[19].position.y).toBeGreaterThan(19.4);
    expect(boxes[19].position.y).toBeLessThan(19.6);
    expect(world.getStats().sleepingBodies).toBe(20);
  });

  test("sleeps a stationary pile of 100 spheres within one second", () => {
    const world = new World();
    addGround(world, 0.4);
    const spheres: RigidBody[] = [];
    for (let ix = 0; ix < 5; ix++) {
      for (let iz = 0; iz < 5; iz++) {
        for (let iy = 0; iy < 4; iy++) {
          spheres.push(
            world.addBody(
              new RigidBody({
                shape: new Sphere(0.5),
                mass: 1,
                friction: 0.4,
                position: new Vec3(ix - 2, 0.5 + iy * 1.01, iz - 2),
              }),
            ),
          );
        }
      }
    }
    stepN(world, 60);
    expect(world.getStats().sleepingBodies).toBe(100);
    for (const sphere of spheres) {
      expect(sphere.isSleeping).toBe(true);
      expect(sphere.linearVelocity.lengthSq()).toBe(0);
    }
  });

  test("wakes a sleeping joint triangle as one unit", () => {
    const world = new World();
    addGround(world);
    const boxes: RigidBody[] = [];
    for (let i = 0; i < 3; i++) {
      boxes.push(
        world.addBody(
          new RigidBody({
            shape: new Box(new Vec3(0.5, 0.5, 0.5)),
            mass: 1,
            position: new Vec3(i - 1, 0.5, 0),
          }),
        ),
      );
    }
    // A joint triangle exercises the wake traversal's cycle guard: waking one
    // member must visit the other two exactly once.
    const [left, middle, right] = boxes;
    world.addJoint(
      new DistanceJoint(left, middle, { restLength: left.position.distanceTo(middle.position) }),
    );
    world.addJoint(
      new DistanceJoint(middle, right, { restLength: middle.position.distanceTo(right.position) }),
    );
    world.addJoint(
      new DistanceJoint(left, right, { restLength: left.position.distanceTo(right.position) }),
    );
    stepN(world, 600);
    expect(world.getStats().sleepingBodies).toBe(3);

    world.addBody(
      new RigidBody({
        shape: new Sphere(0.5),
        mass: 1,
        position: new Vec3(-1, 4, 0),
        linearVelocity: new Vec3(0, -8, 0),
      }),
    );
    stepN(world, 25);
    for (const box of boxes) {
      expect(box.isAwake).toBe(true);
    }
  });

  test("wakes dormant bodies via joints and applied forces", () => {
    const world = new World();
    const anchor = world.addBody(
      new RigidBody({ shape: new Sphere(0.1), type: BodyType.Static, position: new Vec3(0, 3, 0) }),
    );
    const bob = world.addBody(
      new RigidBody({
        shape: new Sphere(0.2),
        mass: 1,
        position: new Vec3(1, 3, 0),
        linearDamping: 4,
        angularDamping: 4,
      }),
    );
    world.addJoint(new BallSocketJoint(anchor, bob, { localAnchorB: new Vec3(-1, 0, 0) }));
    stepN(world, 600);
    expect(bob.isSleeping).toBe(true);

    // A kinematic neighbour keeps the joint active and wakes the sleeper.
    bob.sleep();
    const carrier = world.addBody(
      new RigidBody({
        shape: new Sphere(0.1),
        type: BodyType.Kinematic,
        position: new Vec3(3, 3, 0),
        linearVelocity: new Vec3(0, 0.5, 0),
      }),
    );
    world.addJoint(new DistanceJoint(bob, carrier, { restLength: 2 }));
    world.step(DT);
    expect(bob.isAwake).toBe(true);

    // The sleeping side may be either end of the joint.
    const bob2 = world.addBody(
      new RigidBody({
        shape: new Sphere(0.2),
        mass: 1,
        position: new Vec3(-1, 3, 0),
        linearDamping: 4,
        angularDamping: 4,
      }),
    );
    bob2.sleep();
    world.addJoint(new DistanceJoint(carrier, bob2, { restLength: 2 }));
    world.step(DT);
    expect(bob2.isAwake).toBe(true);
    // A lone sleeper without contacts stays asleep until directly disturbed.
    const sleeper = world.addBody(
      new RigidBody({ shape: new Sphere(0.5), mass: 1, position: new Vec3(10, 1, 10) }),
    );
    sleeper.sleep();
    world.step(DT);
    expect(sleeper.isSleeping).toBe(true);
    sleeper.applyForce(new Vec3(0, 1, 0));
    expect(sleeper.isAwake).toBe(true);
  });

  test("skips contacts between joint-connected bodies unless opted in", () => {
    const build = (collideConnected: boolean): World => {
      const world = new World();
      const frame = world.addBody(
        new RigidBody({
          shape: new Box(new Vec3(0.5, 0.5, 0.5)),
          type: BodyType.Static,
          position: new Vec3(0, 2, 0),
        }),
      );
      const door = world.addBody(
        new RigidBody({
          shape: new Box(new Vec3(0.5, 0.5, 0.5)),
          mass: 1,
          position: new Vec3(0, 2, 0),
        }),
      );
      const joint = new BallSocketJoint(frame, door);
      joint.collideConnected = collideConnected;
      world.addJoint(joint);
      world.step(DT);
      return world;
    };
    expect(build(false).contactSolver.constraints).toHaveLength(0);
    expect(build(false).getStats().manifoldPairs).toBe(0);
    expect(build(true).contactSolver.constraints.length).toBeGreaterThan(0);
  });

  test("reports broadphase and manifold statistics", () => {
    const world = new World();
    addGround(world);
    // A static box resting on the infinite static plane is a static-static
    // broadphase pair and must be ignored.
    world.addBody(
      new RigidBody({
        shape: new Box(new Vec3(0.5, 0.5, 0.5)),
        type: BodyType.Static,
        position: new Vec3(4, 0.5, 0),
      }),
    );
    world.addBody(
      new RigidBody({ shape: new Box(new Vec3(0.5, 0.5, 0.5)), position: new Vec3(0, 0.5, 0) }),
    );
    stepN(world, 10);
    const stats = world.getStats();
    expect(stats.stepCount).toBe(10);
    expect(stats.bodyCount).toBe(3);
    expect(stats.awakeBodies).toBe(1);
    expect(stats.broadphasePairs).toBeGreaterThanOrEqual(1);
    expect(stats.manifoldPairs).toBeGreaterThanOrEqual(1);
    expect(stats.contactPoints).toBeGreaterThanOrEqual(1);
  });
});

describe("Joints", () => {
  test("validates connected bodies", () => {
    const body = new RigidBody({ shape: new Sphere(0.5) });
    expect(() => new BallSocketJoint(body, body)).toThrow(RangeError);
    const staticA = new RigidBody({ shape: new Plane(), type: BodyType.Static });
    const staticB = new RigidBody({ shape: new Plane(), type: BodyType.Static });
    expect(() => new BallSocketJoint(staticA, staticB)).toThrow(RangeError);
    expect(() => new DistanceJoint(staticA, staticB)).toThrow(RangeError);
  });

  test("ball socket keeps anchors together under load", () => {
    const world = new World();
    const anchor = world.addBody(
      new RigidBody({
        shape: new Box(new Vec3(0.1, 0.1, 0.1)),
        type: BodyType.Static,
        position: new Vec3(0, 3, 0),
      }),
    );
    const bob = world.addBody(
      new RigidBody({
        shape: new Box(new Vec3(0.2, 0.2, 0.2)),
        mass: 1,
        position: new Vec3(1, 3, 0),
      }),
    );
    world.addJoint(new BallSocketJoint(anchor, bob, { localAnchorB: new Vec3(-1, 0, 0) }));
    stepN(world, 300);
    const anchorWorld = new Vec3(0, 3, 0);
    const bobWorld = bob.transform.transformPoint(new Vec3(), new Vec3(-1, 0, 0));
    expect(anchorWorld.distanceTo(bobWorld)).toBeLessThan(0.001);
    expect(bob.position.y).toBeLessThan(2.9);
  });

  test("distance joint uses the initial spacing and holds it", () => {
    const world = new World();
    const left = world.addBody(
      new RigidBody({ shape: new Box(new Vec3(0.2, 0.2, 0.2)), mass: 2, position: new Vec3(-1, 3, 0) }),
    );
    const right = world.addBody(
      new RigidBody({ shape: new Box(new Vec3(0.2, 0.2, 0.2)), mass: 1, position: new Vec3(1, 3, 0) }),
    );
    const joint = new DistanceJoint(left, right);
    expect(joint.restLength).toBeCloseTo(2, 12);
    world.addJoint(joint);
    stepN(world, 300);
    expect(left.position.distanceTo(right.position)).toBeCloseTo(2, 3);
    expect(Math.abs(left.position.distanceTo(right.position) - 2)).toBeLessThan(0.001);
  });

  test("distance joint handles coincident anchors and validates rest length", () => {
    const a = new RigidBody({ shape: new Sphere(0.2), mass: 1, position: new Vec3(0, 0, 0) });
    const b = new RigidBody({ shape: new Sphere(0.2), mass: 1, position: new Vec3(0, 0, 0) });
    const joint = new DistanceJoint(a, b, { restLength: 0 });
    joint.prepare(DT, 0.2);
    expect(joint.rows[0].bias).toBeCloseTo(0, 12);
    expect(joint.rows[0].effectiveMass).toBeGreaterThan(0);

    const separated = new DistanceJoint(a, b, { restLength: 0 });
    b.position.set(0.5, 0, 0);
    separated.prepare(DT, 0.2);
    expect(separated.rows[0].bias).toBeLessThan(0);

    for (const restLength of [Number.NaN, -1]) {
      expect(() => new DistanceJoint(a, b, { restLength })).toThrow(RangeError);
    }

    // Explicit anchors are cloned, not aliased.
    const anchorA = new Vec3(0.1, 0, 0);
    const anchorB = new Vec3(-0.1, 0, 0);
    const anchored = new DistanceJoint(a, b, { localAnchorA: anchorA, localAnchorB: anchorB });
    anchorA.set(9, 9, 9);
    expect(anchored.localAnchorA.x).toBe(0.1);
    expect(anchored.localAnchorB.x).toBe(-0.1);
  });

  test("hinge door swings on its axis without drifting", () => {
    const world = new World();
    const frame = world.addBody(
      new RigidBody({
        shape: new Box(new Vec3(0.1, 0.5, 0.1)),
        type: BodyType.Static,
        position: new Vec3(0, 1, 0),
      }),
    );
    const door = world.addBody(
      new RigidBody({
        shape: new Box(new Vec3(0.5, 0.5, 0.05)),
        mass: 2,
        position: new Vec3(0.5, 1, 0),
      }),
    );
    world.addJoint(
      new HingeJoint(frame, door, {
        localAnchorA: new Vec3(0, 0, 0),
        localAnchorB: new Vec3(-0.5, 0, 0),
        localAxisA: new Vec3(0, 1, 0),
        localAxisB: new Vec3(0, 1, 0),
      }),
    );
    stepN(world, 300);
    const frameAnchor = frame.transform.transformPoint(new Vec3(), new Vec3());
    const doorAnchor = door.transform.transformPoint(new Vec3(), new Vec3(-0.5, 0, 0));
    expect(frameAnchor.distanceTo(doorAnchor)).toBeLessThan(0.001);
    const axisA = frame.transform.transformDirection(new Vec3(), new Vec3(0, 1, 0)).normalize();
    const axisB = door.transform.transformDirection(new Vec3(), new Vec3(0, 1, 0)).normalize();
    expect(axisA.cross(axisB).length()).toBeLessThan(0.001);
    expect(door.position.y).toBeCloseTo(1, 2);
  });

  test("hinge axes align with any reference frame choice", () => {
    const axes = [
      new Vec3(0, 0.5, 0.9),
      new Vec3(0.6, 0.1, 0.9),
      new Vec3(0.01, 0.5, 0.05),
      new Vec3(1, 0, 0),
    ];
    for (const axis of axes) {
      const world = new World();
      const frame = world.addBody(
        new RigidBody({ shape: new Box(new Vec3(0.1, 0.1, 0.1)), type: BodyType.Static }),
      );
      const door = world.addBody(
        new RigidBody({
          shape: new Box(new Vec3(0.2, 0.2, 0.2)),
          mass: 1,
          position: new Vec3(1, 0, 0),
        }),
      );
      const joint = new HingeJoint(frame, door, {
        localAxisA: axis,
        localAxisB: axis,
        localAnchorB: new Vec3(-1, 0, 0),
      });
      world.addJoint(joint);
      stepN(world, 5);
      expect(joint.rows).toHaveLength(5);
    }
    const a = new RigidBody({ shape: new Sphere(0.2), mass: 1 });
    const b = new RigidBody({ shape: new Sphere(0.2), mass: 1, position: new Vec3(1, 0, 0) });
    expect(() => new HingeJoint(a, b, { localAxisA: new Vec3() })).toThrow(RangeError);
    expect(() => new HingeJoint(a, b, { localAxisB: new Vec3() })).toThrow(RangeError);
  });

  test("disabled joints do not constrain and inactive joints report so", () => {
    const world = new World();
    const anchor = world.addBody(
      new RigidBody({ shape: new Sphere(0.1), type: BodyType.Static, position: new Vec3(0, 3, 0) }),
    );
    const bob = world.addBody(
      new RigidBody({ shape: new Sphere(0.2), mass: 1, position: new Vec3(1, 3, 0) }),
    );
    const joint = new BallSocketJoint(anchor, bob);
    joint.enabled = false;
    world.addJoint(joint);
    stepN(world, 120);
    expect(bob.position.y).toBeLessThan(1);

    const sleeper = new RigidBody({ shape: new Sphere(0.2), mass: 1, position: new Vec3(0, 3, 0) });
    const other = new RigidBody({ shape: new Sphere(0.2), mass: 1, position: new Vec3(1, 3, 0) });
    const socket = new BallSocketJoint(sleeper, other);
    other.sleep();
    expect(socket.isActive()).toBe(true);
    sleeper.sleep();
    expect(socket.isActive()).toBe(false);

    const kinematic = new RigidBody({ shape: new Sphere(0.2), type: BodyType.Kinematic });
    const sleepingDynamic = new RigidBody({ shape: new Sphere(0.2), mass: 1, position: new Vec3(1, 0, 0) });
    sleepingDynamic.sleep();
    const carried = new BallSocketJoint(kinematic, sleepingDynamic);
    expect(carried.isActive()).toBe(true);
  });

  test("prevents runaway orbital energy with symplectic Euler", () => {
    const world = new World({ gravity: new Vec3(), substeps: 1, solverIterations: 1 });
    const satellite = world.addBody(
      new RigidBody({
        shape: new Sphere(0.1),
        mass: 1,
        position: new Vec3(1, 0, 0),
        linearVelocity: new Vec3(0, 0, 1),
      }),
    );
    const energy = (): number =>
      0.5 * satellite.linearVelocity.lengthSq() - 1 / satellite.position.length();
    const initial = energy();
    let minEnergy = initial;
    let maxEnergy = initial;
    const dt = 1 / 240;
    for (let i = 0; i < 2400; i++) {
      const r = satellite.position.length();
      const force = new Vec3()
        .copy(satellite.position)
        .scale(-1 / (r * r * r));
      satellite.applyForce(force);
      world.step(dt);
      const e = energy();
      minEnergy = Math.min(minEnergy, e);
      maxEnergy = Math.max(maxEnergy, e);
    }
    expect(maxEnergy - initial).toBeLessThan(Math.abs(initial) * 0.05);
    expect(initial - minEnergy).toBeLessThan(Math.abs(initial) * 0.05);
    expect(energy()).toBeLessThan(initial + Math.abs(initial) * 0.01);
  });
});
