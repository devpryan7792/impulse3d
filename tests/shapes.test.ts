import { describe, expect, test } from "vitest";
import { Mat3 } from "../src/math/Mat3.js";
import { Mat4 } from "../src/math/Mat4.js";
import { Quat } from "../src/math/Quat.js";
import { Transform } from "../src/math/Transform.js";
import { Vec3 } from "../src/math/Vec3.js";
import { AABB } from "../src/collision/AABB.js";
import { Shape, ShapeType } from "../src/collision/shapes/Shape.js";
import { Sphere } from "../src/collision/shapes/Sphere.js";
import { Box } from "../src/collision/shapes/Box.js";
import { Capsule } from "../src/collision/shapes/Capsule.js";
import { Plane } from "../src/collision/shapes/Plane.js";
import { ConvexPolytope } from "../src/collision/shapes/ConvexPolytope.js";
import {
  expectVecClose,
  makeRng,
  randomQuat,
  randomUnitVec3,
  randomVec3,
} from "./testUtils.js";

const rng = makeRng(0xc0111de);

/** Worst-case dot-product excess of `samples` over the support point. */
function supportExcess(shape: Shape, transform: Transform, direction: Vec3, samples: Vec3[]): number {
  const support = shape.getSupport(direction, transform);
  const bound = support.dot(direction);
  let excess = 0;
  for (const sample of samples) {
    excess = Math.max(excess, sample.dot(direction) - bound);
  }
  return excess;
}

function randomSphereInteriorPoint(rngLocal: typeof rng, center: Vec3, radius: number): Vec3 {
  const direction = randomUnitVec3(rngLocal);
  return center.clone().addScaled(direction, radius * rngLocal.next());
}

describe("Transform", () => {
  test("identity defaults, set, copy, clone", () => {
    const identity = new Transform();
    expect(identity.position.equals(Vec3.zero())).toBe(true);
    expect(identity.rotation.equals(new Quat())).toBe(true);
    const transform = new Transform(new Vec3(1, 2, 3), Quat.fromAxisAngle(new Vec3(0, 0, 1), 0.5));
    const clone = transform.clone();
    expectVecClose(clone.position, transform.position, 0);
    expect(clone.rotation.equals(transform.rotation)).toBe(true);
    clone.setIdentity();
    expect(clone.position.equals(Vec3.zero())).toBe(true);
    expect(clone.rotation.equals(new Quat())).toBe(true);
    clone.copy(transform);
    expectVecClose(clone.position, new Vec3(1, 2, 3), 0);
    clone.set(new Vec3(-1, -2, -3), new Quat());
    expectVecClose(clone.position, new Vec3(-1, -2, -3), 0);
    expect(clone.toString()).toContain("Transform(Vec3(-1, -2, -3)");
  });

  test("point and direction transforms are exact inverses", () => {
    const rotation = Quat.fromAxisAngle(new Vec3(1, 2, -1), 1.1);
    const transform = new Transform(new Vec3(4, -1, 2), rotation);
    const quarterTurn = Quat.fromAxisAngle(new Vec3(0, 0, 1), Math.PI / 2);
    const rotated = new Transform(new Vec3(1, 0, 0), quarterTurn);
    expectVecClose(rotated.transformPoint(new Vec3(), new Vec3(1, 0, 0)), new Vec3(1, 1, 0), 1e-12);
    expectVecClose(rotated.transformDirection(new Vec3(), new Vec3(0, 0, 1)), new Vec3(0, 0, 1), 1e-12);

    for (let i = 0; i < 100; i++) {
      const probe = randomVec3(rng, -10, 10);
      const worldPoint = transform.transformPoint(new Vec3(), probe);
      expectVecClose(transform.inverseTransformPoint(new Vec3(), worldPoint), probe, 1e-12);
      const direction = randomVec3(rng, -10, 10);
      const worldDirection = transform.transformDirection(new Vec3(), direction);
      expectVecClose(transform.inverseTransformDirection(new Vec3(), worldDirection), direction, 1e-12);
      expect(worldDirection.length()).toBeCloseTo(direction.length(), 12);
    }
    const scratch = new Vec3(1, 1, 1);
    const inPlace = scratch.clone();
    transform.transformPoint(inPlace, inPlace);
    expectVecClose(inPlace, transform.transformPoint(new Vec3(), scratch), 1e-12);
  });

  test("toMat4 matches point and direction transforms", () => {
    const transform = new Transform(randomVec3(rng, -3, 3), randomQuat(rng));
    const matrix = transform.toMat4();
    const out = new Mat4();
    expect(transform.toMat4(out)).toBe(out);
    const probe = randomVec3(rng, -3, 3);
    expectVecClose(
      Mat4.transformPoint(new Vec3(), matrix, probe),
      transform.transformPoint(new Vec3(), probe),
      1e-12,
    );
    expectVecClose(
      Mat4.transformDirection(new Vec3(), matrix, probe),
      transform.transformDirection(new Vec3(), probe),
      1e-12,
    );
  });
});

describe("AABB", () => {
  test("construction, corners and derived quantities", () => {
    const degenerate = new AABB();
    expectVecClose(degenerate.min, Vec3.zero(), 0);
    expectVecClose(degenerate.max, Vec3.zero(), 0);
    const interval = new AABB(new Vec3(-1, -2, -3), new Vec3(1, 2, 3));
    expectVecClose(AABB.from(interval).min, interval.min, 0);
    expectVecClose(interval.clone().max, interval.max, 0);
    expect(interval.getCenter().equals(Vec3.zero())).toBe(true);
    const halfExtents = interval.getHalfExtents();
    expectVecClose(halfExtents, new Vec3(1, 2, 3), 0);
    const outCenter = new Vec3();
    expect(interval.getCenter(outCenter)).toBe(outCenter);
    const outHalf = new Vec3();
    expect(interval.getHalfExtents(outHalf)).toBe(outHalf);
    const shifted = new AABB();
    expect(shifted.setFromCenterAndHalfExtents(new Vec3(1, 2, 3), new Vec3(0.5, 0.5, 0.5)).min.equals(new Vec3(0.5, 1.5, 2.5))).toBe(true);
    expect(shifted.copy(interval).equals(interval)).toBe(true);
    const assigned = new AABB();
    expect(assigned.set(new Vec3(0, 0, 0), new Vec3(1, 1, 1)).max.equals(new Vec3(1, 1, 1))).toBe(true);
    expect(interval.toString()).toContain("AABB(min=Vec3(-1, -2, -3)");
    expect(interval.isFinite()).toBe(true);
    expect(interval.isEmpty()).toBe(false);
  });

  test("empty and non-finite classifications", () => {
    const empty = new AABB(new Vec3(1, 1, 1), new Vec3(-1, -1, -1));
    expect(empty.isEmpty()).toBe(true);
    expect(empty.surfaceArea()).toBe(0);
    expect(empty.overlaps(new AABB(new Vec3(-1, -1, -1), new Vec3(1, 1, 1)))).toBe(false);
    const infinite = new AABB(new Vec3(-Infinity, -Infinity, -Infinity), new Vec3(Infinity, Infinity, Infinity));
    expect(infinite.isFinite()).toBe(false);
    expect(infinite.hasNaN()).toBe(false);
    expect(infinite.isEmpty()).toBe(false);
    const nan = new AABB(new Vec3(NaN, 0, 0), new Vec3(1, 1, 1));
    expect(nan.hasNaN()).toBe(true);
    expect(new AABB(new Vec3(0, 0, 0), new Vec3(1, NaN, 1)).hasNaN()).toBe(true);
    expect(new AABB(new Vec3(0, 0, 0), new Vec3(1, 1, 1)).hasNaN()).toBe(false);
  });

  test("surface area matches the SAH metric", () => {
    expect(new AABB(new Vec3(1, 1, 1), new Vec3(3, 3, 3)).surfaceArea()).toBeCloseTo(24, 12);
    expect(new AABB(new Vec3(0, 0, 0), new Vec3(1, 1, 0)).surfaceArea()).toBeCloseTo(2, 12);
    expect(new AABB(new Vec3(0, 0, 0), new Vec3(0, 0, 0)).surfaceArea()).toBe(0);
  });

  test("overlap, containment and point tests are inclusive", () => {
    const a = new AABB(new Vec3(0, 0, 0), new Vec3(1, 1, 1));
    const b = new AABB(new Vec3(1, 1, 1), new Vec3(2, 2, 2));
    const c = new AABB(new Vec3(1.001, 1, 1), new Vec3(2, 2, 2));
    expect(a.overlaps(b)).toBe(true);
    expect(a.overlaps(c)).toBe(false);
    expect(a.overlaps(a.clone())).toBe(true);
    expect(a.contains(new AABB(new Vec3(0.25, 0.25, 0.25), new Vec3(0.75, 0.75, 0.75)))).toBe(true);
    expect(a.contains(new AABB(new Vec3(-0.25, 0.25, 0.25), new Vec3(0.75, 0.75, 0.75)))).toBe(false);
    expect(a.contains(new AABB(new Vec3(0.5, 0.5, 0.5), new Vec3(1.5, 1.5, 1.5)))).toBe(false);
    expect(a.contains(new AABB(new Vec3(1, 1, 1), new Vec3(1.5, 1.5, 1.5)))).toBe(false);
    expect(a.contains(a)).toBe(true);
    expect(a.containsPoint(new Vec3(0.5, 0.5, 0.5))).toBe(true);
    expect(a.containsPoint(new Vec3(1, 1, 1))).toBe(true);
    expect(a.containsPoint(new Vec3(1.5, 0.5, 0.5))).toBe(false);
  });

  test("expand, expandBy and merge (instance and static)", () => {
    const a = new AABB(new Vec3(-1, -1, -1), new Vec3(1, 1, 1));
    const expanded = a.clone().expand(0.5);
    expectVecClose(expanded.min, new Vec3(-1.5, -1.5, -1.5), 0);
    expectVecClose(expanded.max, new Vec3(1.5, 1.5, 1.5), 0);
    const byVector = a.clone().expandBy(new Vec3(1, 2, 3));
    expectVecClose(byVector.min, new Vec3(-2, -3, -4), 0);
    expectVecClose(byVector.max, new Vec3(2, 3, 4), 0);

    const b = new AABB(new Vec3(2, 0, 0), new Vec3(4, 1, 1));
    const merged = a.clone().merge(b);
    expectVecClose(merged.min, new Vec3(-1, -1, -1), 0);
    expectVecClose(merged.max, new Vec3(4, 1, 1), 0);
    const out = new AABB();
    expect(AABB.merge(out, a, b)).toBe(out);
    expect(out.equals(merged)).toBe(true);
    const empty = new AABB(new Vec3(1, 1, 1), new Vec3(-1, -1, -1));
    expect(AABB.merge(new AABB(), empty, b).equals(b)).toBe(true);
    expect(AABB.merge(new AABB(), a, empty).equals(a)).toBe(true);
    expect(a.equals(merged, 1)).toBe(false);
  });

  test("velocity fattening follows the Section 4.1 envelope", () => {
    const tight = new AABB(new Vec3(-1, -1, -1), new Vec3(1, 1, 1));
    const fat = AABB.fatten(new AABB(), tight, new Vec3(2, -4, 0), 0.5, 2, 0.1);
    expectVecClose(fat.min, new Vec3(-1.1, -5.1, -1.1), 1e-12);
    expectVecClose(fat.max, new Vec3(3.1, 1.1, 1.1), 1e-12);
    const zeroVelocity = AABB.fatten(new AABB(), tight, Vec3.zero(), 1 / 60, 1, 0.05);
    expectVecClose(zeroVelocity.min, new Vec3(-1.05, -1.05, -1.05), 1e-12);
    expectVecClose(zeroVelocity.max, new Vec3(1.05, 1.05, 1.05), 1e-12);
    const alias = tight.clone();
    AABB.fatten(alias, alias, new Vec3(1, 0, 0), 1, 1, 0);
    expectVecClose(alias.min, new Vec3(-1, -1, -1), 1e-12);
    expectVecClose(alias.max, new Vec3(2, 1, 1), 1e-12);
  });
});

describe("Sphere", () => {
  test("validation and geometry metadata", () => {
    expect(() => new Sphere(0)).toThrow(RangeError);
    expect(() => new Sphere(-1)).toThrow(RangeError);
    const sphere = new Sphere(2);
    expect(sphere.radius).toBe(2);
    expect(sphere.type).toBe(ShapeType.Sphere);
    expect(sphere instanceof Shape).toBe(true);
    expect(sphere.volume()).toBeCloseTo((4 / 3) * Math.PI * 8, 12);
  });

  test("computeAABB is the center sphere box", () => {
    const sphere = new Sphere(1.5);
    const transform = new Transform(new Vec3(1, -2, 3), randomQuat(rng));
    const aabb = sphere.computeAABB(transform);
    expectVecClose(aabb.min, new Vec3(-0.5, -3.5, 1.5), 1e-12);
    expectVecClose(aabb.max, new Vec3(2.5, -0.5, 4.5), 1e-12);
    const out = new AABB();
    expect(sphere.computeAABB(transform, out)).toBe(out);
    expect(out.equals(aabb)).toBe(true);
  });

  test("support mapping is the analytical boundary point", () => {
    const sphere = new Sphere(2);
    const center = new Vec3(3, -1, 4);
    const transform = new Transform(center, randomQuat(rng));
    const samples = [center.clone()];
    for (let i = 0; i < 200; i++) {
      samples.push(randomSphereInteriorPoint(rng, center, 2));
    }
    for (let i = 0; i < 100; i++) {
      const direction = randomVec3(rng, -5, 5);
      const support = sphere.getSupport(direction, transform);
      expect(Vec3.distance(support, center)).toBeCloseTo(2, 10);
      expectVecClose(support, center.clone().addScaled(direction.clone().normalize(), 2), 1e-10);
      expect(supportExcess(sphere, transform, direction, samples)).toBeLessThan(1e-9);
    }
    const out = new Vec3();
    expect(sphere.getSupport(new Vec3(0, 1, 0), transform, out)).toBe(out);
    expectVecClose(out, new Vec3(3, 1, 4), 1e-12);
    expectVecClose(sphere.getSupport(new Vec3(0, -1, 0), transform), new Vec3(3, -3, 4), 1e-12);
    expectVecClose(sphere.getSupport(new Vec3(0, 0, 0), transform), new Vec3(5, -1, 4), 1e-12);
  });

  test("local inertia follows I = 2/5 m r^2", () => {
    const sphere = new Sphere(1.5);
    const inertia = sphere.computeLocalInertia(3);
    const expected = 0.4 * 3 * 1.5 * 1.5;
    expect(inertia.m00).toBeCloseTo(expected, 12);
    expect(inertia.m11).toBeCloseTo(expected, 12);
    expect(inertia.m22).toBeCloseTo(expected, 12);
    expect(inertia.m01).toBe(0);
    const out = new Mat3();
    expect(sphere.computeLocalInertia(1, out)).toBe(out);
    expect(() => sphere.computeLocalInertia(0)).toThrow(RangeError);
  });
});

describe("Box", () => {
  const halfExtents = new Vec3(1, 2, 3);

  test("validation and metadata", () => {
    expect(() => new Box(new Vec3(0, 1, 1))).toThrow(RangeError);
    expect(() => new Box(new Vec3(1, -1, 1))).toThrow(RangeError);
    expect(() => new Box(new Vec3(1, 1, 0))).toThrow(RangeError);
    const box = new Box(halfExtents);
    expect(box.type).toBe(ShapeType.Box);
    expect(box.volume()).toBeCloseTo(48, 12);
    const mutated = new Vec3(1, 1, 1);
    const fromMutated = new Box(mutated);
    mutated.set(9, 9, 9);
    expectVecClose(fromMutated.halfExtents, new Vec3(1, 1, 1), 0);
  });

  test("computeAABB matches brute-force corner extremes", () => {
    const box = new Box(halfExtents);
    const identity = box.computeAABB(new Transform());
    expectVecClose(identity.min, new Vec3(-1, -2, -3), 1e-12);
    expectVecClose(identity.max, new Vec3(1, 2, 3), 1e-12);

    const quarterTurn = Quat.fromAxisAngle(new Vec3(0, 0, 1), Math.PI / 2);
    const rotated = box.computeAABB(new Transform(new Vec3(0, 0, 0), quarterTurn));
    expectVecClose(rotated.min, new Vec3(-2, -1, -3), 1e-12);
    expectVecClose(rotated.max, new Vec3(2, 1, 3), 1e-12);

    for (let i = 0; i < 50; i++) {
      const transform = new Transform(randomVec3(rng, -5, 5), randomQuat(rng));
      const aabb = box.computeAABB(transform);
      const corners: Vec3[] = [];
      for (const sx of [-1, 1]) {
        for (const sy of [-1, 1]) {
          for (const sz of [-1, 1]) {
            corners.push(
              transform.transformPoint(
                new Vec3(),
                new Vec3(
                  sx * box.halfExtents.x,
                  sy * box.halfExtents.y,
                  sz * box.halfExtents.z,
                ),
              ),
            );
          }
        }
      }
      for (const corner of corners) {
        // The corner path (quaternion sandwich) and the AABB path (rotation
        // matrix) round differently, so allow a few ulps of slack.
        expect(corner.x).toBeGreaterThanOrEqual(aabb.min.x - 1e-9);
        expect(corner.x).toBeLessThanOrEqual(aabb.max.x + 1e-9);
        expect(corner.y).toBeGreaterThanOrEqual(aabb.min.y - 1e-9);
        expect(corner.y).toBeLessThanOrEqual(aabb.max.y + 1e-9);
        expect(corner.z).toBeGreaterThanOrEqual(aabb.min.z - 1e-9);
        expect(corner.z).toBeLessThanOrEqual(aabb.max.z + 1e-9);
      }
      const bruteMin = corners.reduce((acc, corner) => Vec3.min(new Vec3(), acc, corner), corners[0]);
      const bruteMax = corners.reduce((acc, corner) => Vec3.max(new Vec3(), acc, corner), corners[0]);
      expectVecClose(aabb.min, bruteMin, 1e-12);
      expectVecClose(aabb.max, bruteMax, 1e-12);
    }
  });

  test("support mapping selects the extreme corner", () => {
    const box = new Box(halfExtents);
    const transform = new Transform(new Vec3(1, 2, 3), new Quat());
    expectVecClose(box.getSupport(new Vec3(1, 0, 0), transform), new Vec3(2, 2, 3), 1e-12);
    expectVecClose(box.getSupport(new Vec3(0, -1, 0), transform), new Vec3(1, 0, 3), 1e-12);
    expectVecClose(box.getSupport(new Vec3(-1, -1, -1), transform), new Vec3(0, 0, 0), 1e-12);
    expectVecClose(box.getSupport(new Vec3(1, 0, 0), transform, new Vec3()), new Vec3(2, 2, 3), 1e-12);
    expectVecClose(box.getSupport(new Vec3(0.5, 0, 0), transform), new Vec3(2, 2, 3), 1e-12);

    for (let i = 0; i < 50; i++) {
      const probe = new Transform(randomVec3(rng, -4, 4), randomQuat(rng));
      const direction = randomVec3(rng, -5, 5);
      const samples: Vec3[] = [];
      for (let j = 0; j < 100; j++) {
        const local = new Vec3(
          rng.range(-1, 1) * halfExtents.x,
          rng.range(-1, 1) * halfExtents.y,
          rng.range(-1, 1) * halfExtents.z,
        );
        samples.push(probe.transformPoint(new Vec3(), local));
      }
      expect(supportExcess(box, probe, direction, samples)).toBeLessThan(1e-9);
    }
  });

  test("exactly-zero local direction yields a face support point", () => {
    const box = new Box(halfExtents);
    const transform = new Transform(new Vec3(1, 2, 3), new Quat());
    const support = box.getSupport(new Vec3(1, 0, 0), transform);
    expectVecClose(support, new Vec3(2, 2, 3), 1e-12);
    const rotated = new Transform(new Vec3(0, 0, 0), Quat.fromAxisAngle(new Vec3(0, 0, 1), Math.PI / 2));
    const localDirection = new Vec3(1, 0, 0);
    const worldDirection = rotated.transformDirection(new Vec3(), localDirection);
    const rotatedSupport = box.getSupport(worldDirection, rotated);
    const expected = rotated.transformPoint(new Vec3(), new Vec3(halfExtents.x, 0, 0));
    expectVecClose(rotatedSupport, expected, 1e-12);
  });

  test("local inertia follows the Section 2.3 box formula", () => {
    const mass = 7;
    const inertia = new Box(new Vec3(0.5, 1, 2)).computeLocalInertia(mass);
    const factor = mass / 12;
    expect(inertia.m00).toBeCloseTo(factor * ((2 * 1) ** 2 + (2 * 2) ** 2), 12);
    expect(inertia.m11).toBeCloseTo(factor * ((2 * 0.5) ** 2 + (2 * 2) ** 2), 12);
    expect(inertia.m22).toBeCloseTo(factor * ((2 * 0.5) ** 2 + (2 * 1) ** 2), 12);
    const cube = new Box(new Vec3(1, 1, 1)).computeLocalInertia(6);
    expect(cube.m00).toBeCloseTo(6 * (2 / 3), 12);
    const out = new Mat3();
    expect(new Box(new Vec3(1, 1, 1)).computeLocalInertia(1, out)).toBe(out);
    expect(() => new Box(new Vec3(1, 1, 1)).computeLocalInertia(-1)).toThrow(RangeError);
  });
});

describe("Capsule", () => {
  test("validation and metadata", () => {
    expect(() => new Capsule(0, 1)).toThrow(RangeError);
    expect(() => new Capsule(1, 0)).toThrow(RangeError);
    expect(() => new Capsule(-1, 1)).toThrow(RangeError);
    const capsule = new Capsule(0.5, 1);
    expect(capsule.radius).toBe(0.5);
    expect(capsule.halfHeight).toBe(1);
    expect(capsule.type).toBe(ShapeType.Capsule);
    expect(capsule.volume()).toBeCloseTo(Math.PI * 0.25 * 2 + (4 / 3) * Math.PI * 0.125, 12);
  });

  test("computeAABB matches the extruded segment bounds", () => {
    const capsule = new Capsule(0.5, 1);
    const identity = capsule.computeAABB(new Transform());
    expectVecClose(identity.min, new Vec3(-0.5, -1.5, -0.5), 1e-12);
    expectVecClose(identity.max, new Vec3(0.5, 1.5, 0.5), 1e-12);
    const quarterTurn = Quat.fromAxisAngle(new Vec3(0, 0, 1), Math.PI / 2);
    const rotated = capsule.computeAABB(new Transform(Vec3.zero(), quarterTurn));
    expectVecClose(rotated.min, new Vec3(-1.5, -0.5, -0.5), 1e-12);
    expectVecClose(rotated.max, new Vec3(1.5, 0.5, 0.5), 1e-12);
    for (let i = 0; i < 50; i++) {
      const transform = new Transform(randomVec3(rng, -4, 4), randomQuat(rng));
      const aabb = capsule.computeAABB(transform);
      const endpointA = transform.transformPoint(new Vec3(), new Vec3(0, -1, 0));
      const endpointB = transform.transformPoint(new Vec3(), new Vec3(0, 1, 0));
      for (const endpoint of [endpointA, endpointB]) {
        expect(aabb.containsPoint(endpoint)).toBe(true);
        const endpointBox = new AABB().setFromCenterAndHalfExtents(endpoint, new Vec3(0.49, 0.49, 0.49));
        expect(aabb.contains(endpointBox)).toBe(true);
        expect(aabb.overlaps(endpointBox)).toBe(true);
      }
    }
  });

  test("support mapping is the endpoint offset by the radius", () => {
    const capsule = new Capsule(0.5, 1);
    const transform = new Transform(Vec3.zero(), new Quat());
    expectVecClose(capsule.getSupport(new Vec3(0, 1, 0), transform), new Vec3(0, 1.5, 0), 1e-12);
    expectVecClose(capsule.getSupport(new Vec3(0, -1, 0), transform), new Vec3(0, -1.5, 0), 1e-12);
    expectVecClose(capsule.getSupport(new Vec3(1, 0, 0), transform), new Vec3(0.5, 1, 0), 1e-12);
    expectVecClose(capsule.getSupport(Vec3.zero(), transform), new Vec3(0.5, 1, 0), 1e-12);
    const out = new Vec3();
    expect(capsule.getSupport(new Vec3(1, 0, 0), transform, out)).toBe(out);

    for (let i = 0; i < 50; i++) {
      const probe = new Transform(randomVec3(rng, -4, 4), randomQuat(rng));
      const direction = randomVec3(rng, -5, 5);
      const samples: Vec3[] = [];
      const endpointA = probe.transformPoint(new Vec3(), new Vec3(0, -1, 0));
      samples.push(endpointA, randomSphereInteriorPoint(rng, endpointA, 0.5));
      for (let j = 0; j < 60; j++) {
        const t = rng.range(-1, 1);
        const axisPoint = probe.transformPoint(new Vec3(), new Vec3(0, t, 0));
        samples.push(randomSphereInteriorPoint(rng, axisPoint, 0.5));
      }
      expect(supportExcess(capsule, probe, direction, samples)).toBeLessThan(1e-9);
    }
  });

  test("local inertia reduces to a sphere when the cylinder height vanishes", () => {
    const radius = 0.75;
    const mass = 2.5;
    const capsule = new Capsule(radius, 1e-9);
    const inertia = capsule.computeLocalInertia(mass);
    const sphereInertia = 0.4 * mass * radius * radius;
    expect(inertia.m00).toBeCloseTo(sphereInertia, 6);
    expect(inertia.m11).toBeCloseTo(sphereInertia, 6);
    expect(inertia.m22).toBeCloseTo(sphereInertia, 6);
  });

  test("local inertia matches Monte Carlo integration of the solid", () => {
    const radius = 0.6;
    const halfHeight = 0.8;
    const mass = 1.7;
    const capsule = new Capsule(radius, halfHeight);
    const inertia = capsule.computeLocalInertia(mass);

    const boundX = radius;
    const boundY = halfHeight + radius;
    const boundZ = radius;
    const volumeBox = 8 * boundX * boundY * boundZ;
    const sampleCount = 2_000_000;
    let insideCount = 0;
    let sumYyZz = 0;
    let sumXxZz = 0;
    for (let i = 0; i < sampleCount; i++) {
      const x = rng.range(-boundX, boundX);
      const y = rng.range(-boundY, boundY);
      const z = rng.range(-boundZ, boundZ);
      const clampedY = y < -halfHeight ? -halfHeight : y > halfHeight ? halfHeight : y;
      const dy = y - clampedY;
      if (x * x + dy * dy + z * z <= radius * radius) {
        insideCount++;
        sumYyZz += y * y + z * z;
        sumXxZz += x * x + z * z;
      }
    }
    const capsuleVolume = Math.PI * radius * radius * 2 * halfHeight + (4 / 3) * Math.PI * radius ** 3;
    expect((insideCount / sampleCount) * volumeBox).toBeCloseTo(capsuleVolume, 1);
    const transverse = (mass * sumYyZz) / insideCount;
    const axial = (mass * sumXxZz) / insideCount;
    expect(inertia.m00).toBeCloseTo(transverse, 2);
    expect(inertia.m22).toBeCloseTo(transverse, 2);
    expect(inertia.m11).toBeCloseTo(axial, 2);
    expect(inertia.m00).toBeGreaterThan(inertia.m11);
    const out = new Mat3();
    expect(capsule.computeLocalInertia(1, out)).toBe(out);
    expect(() => capsule.computeLocalInertia(0)).toThrow(RangeError);
  });
});

describe("Plane", () => {
  test("construction validates and normalizes the normal", () => {
    const ground = new Plane();
    expectVecClose(ground.localNormal, new Vec3(0, 1, 0), 0);
    expect(ground.offset).toBe(0);
    expect(ground.type).toBe(ShapeType.Plane);
    const steep = new Plane(new Vec3(0, 2, 0), 3);
    expectVecClose(steep.localNormal, new Vec3(0, 1, 0), 1e-15);
    expect(steep.offset).toBe(3);
    expect(() => new Plane(new Vec3(0, 0, 0))).toThrow(RangeError);
    expect(() => new Plane(new Vec3(0, 1, 0), NaN)).toThrow(RangeError);
    expect(ground.toString()).toContain("Plane(normal=Vec3(0, 1, 0)");
  });

  test("computeAABB is infinite and world plane geometry is transformed", () => {
    const plane = new Plane(new Vec3(0, 1, 0), 2);
    const aabb = plane.computeAABB(new Transform(new Vec3(1, 1, 1), randomQuat(rng)));
    expect(aabb.min.x).toBe(-Infinity);
    expect(aabb.max.z).toBe(Infinity);
    const transform = new Transform(new Vec3(0, 3, 0), new Quat());
    expectVecClose(plane.getWorldNormal(transform), new Vec3(0, 1, 0), 1e-15);
    expect(plane.getWorldOffset(transform)).toBeCloseTo(5, 12);
    const out = new Vec3();
    expect(plane.getWorldNormal(transform, out)).toBe(out);

    const quarterTurn = Quat.fromAxisAngle(new Vec3(0, 0, 1), Math.PI / 2);
    const rotated = new Transform(new Vec3(0, 0, 0), quarterTurn);
    expectVecClose(plane.getWorldNormal(rotated), new Vec3(-1, 0, 0), 1e-12);
    expect(plane.getWorldOffset(rotated)).toBeCloseTo(2, 12);
  });

  test("getSupport rejects non-zero directions and projects the origin", () => {
    const plane = new Plane(new Vec3(0, 1, 0), 2);
    const transform = new Transform(new Vec3(0, 1, 0), new Quat());
    expect(() => plane.getSupport(new Vec3(0, 1, 0), transform)).toThrow(RangeError);
    expect(() => plane.getSupport(new Vec3(1, 0, 0), transform)).toThrow(RangeError);
    const support = plane.getSupport(Vec3.zero(), transform);
    expectVecClose(support, new Vec3(0, 3, 0), 1e-12);
    const out = new Vec3();
    expect(plane.getSupport(Vec3.zero(), transform, out)).toBe(out);
  });
});

describe("ConvexPolytope", () => {
  const tetrahedron = (): Vec3[] => [
    new Vec3(1, 1, 1),
    new Vec3(-1, -1, 1),
    new Vec3(-1, 1, -1),
    new Vec3(1, -1, -1),
  ];

  test("validation rejects degenerate vertex sets", () => {
    expect(() => new ConvexPolytope([new Vec3(), new Vec3(1, 0, 0), new Vec3(0, 1, 0)])).toThrow(RangeError);
    expect(() =>
      new ConvexPolytope([new Vec3(NaN, 0, 0), new Vec3(1, 1, 0), new Vec3(0, 1, 1), new Vec3(1, 0, 1)]),
    ).toThrow(RangeError);
    expect(() =>
      new ConvexPolytope([new Vec3(0, 0, 0), new Vec3(1, 0, 0), new Vec3(0, 1, 0), new Vec3(0, 0, Infinity)]),
    ).toThrow(RangeError);
  });

  test("vertices are copied and typed", () => {
    const input = tetrahedron();
    const polytope = new ConvexPolytope(input);
    expect(polytope.type).toBe(ShapeType.ConvexPolytope);
    expect(polytope.vertices.length).toBe(4);
    input[0].set(100, 100, 100);
    expectVecClose(polytope.vertices[0], new Vec3(1, 1, 1), 0);
  });

  test("computeAABB matches brute-force transformed vertices", () => {
    const polytope = new ConvexPolytope(tetrahedron());
    const identity = polytope.computeAABB(new Transform());
    expectVecClose(identity.min, new Vec3(-1, -1, -1), 1e-12);
    expectVecClose(identity.max, new Vec3(1, 1, 1), 1e-12);
    for (let i = 0; i < 50; i++) {
      const transform = new Transform(randomVec3(rng, -5, 5), randomQuat(rng));
      const aabb = polytope.computeAABB(transform);
      const transformed = polytope.vertices.map((vertex) => transform.transformPoint(new Vec3(), vertex));
      const bruteMin = transformed.reduce((acc, p) => Vec3.min(new Vec3(), acc, p), transformed[0]);
      const bruteMax = transformed.reduce((acc, p) => Vec3.max(new Vec3(), acc, p), transformed[0]);
      expectVecClose(aabb.min, bruteMin, 1e-12);
      expectVecClose(aabb.max, bruteMax, 1e-12);
    }
  });

  test("support mapping is the extreme vertex in world space", () => {
    const polytope = new ConvexPolytope(tetrahedron());
    const transform = new Transform(new Vec3(2, -1, 0.5), Quat.fromAxisAngle(new Vec3(1, 1, 0), 0.8));
    for (let i = 0; i < 100; i++) {
      const direction = randomVec3(rng, -5, 5);
      const support = polytope.getSupport(direction, transform);
      const bound = support.dot(direction);
      let bestVertex = -Infinity;
      for (const vertex of polytope.vertices) {
        bestVertex = Math.max(bestVertex, transform.transformPoint(new Vec3(), vertex).dot(direction));
      }
      expect(bound).toBeCloseTo(bestVertex, 10);
      const samples: Vec3[] = [];
      for (let j = 0; j < 40; j++) {
        const weights = [rng.next(), rng.next(), rng.next(), rng.next()];
        const total = weights.reduce((sum, w) => sum + w, 0);
        const combination = new Vec3();
        for (let k = 0; k < 4; k++) {
          combination.addScaled(polytope.vertices[k], weights[k] / total);
        }
        samples.push(transform.transformPoint(new Vec3(), combination));
      }
      expect(supportExcess(polytope, transform, direction, samples)).toBeLessThan(1e-9);
    }
    const out = new Vec3();
    expect(polytope.getSupport(new Vec3(0, 0, 1), transform, out)).toBe(out);
  });
});

describe("ShapeType", () => {
  test("discriminants are distinct and stable", () => {
    const types = [
      new Sphere(1).type,
      new Box(new Vec3(1, 1, 1)).type,
      new Capsule(1, 1).type,
      new Plane().type,
      new ConvexPolytope(tetrahedronVertices()).type,
    ];
    expect(new Set(types).size).toBe(5);
    expect(ShapeType.Sphere).toBe(0);
    expect(ShapeType.ConvexPolytope).toBe(4);
  });
});

function tetrahedronVertices(): Vec3[] {
  return [new Vec3(1, 1, 1), new Vec3(-1, -1, 1), new Vec3(-1, 1, -1), new Vec3(1, -1, -1)];
}
