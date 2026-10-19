import { describe, expect, test } from "vitest";
import { Mat3 } from "../src/math/Mat3.js";
import { Mat4 } from "../src/math/Mat4.js";
import { Quat } from "../src/math/Quat.js";
import { Transform } from "../src/math/Transform.js";
import { Vec3 } from "../src/math/Vec3.js";
import { approximatelyEqual, clamp, EPSILON, EPSILON_SQ } from "../src/math/MathUtils.js";
import {
  expectMat3Close,
  expectMat4Close,
  expectSameRotation,
  expectVecClose,
  makeRng,
  quatRotationAngle,
  randomQuat,
  randomUnitVec3,
  randomVec3,
} from "./testUtils.js";

const rng = makeRng(0x5eed1);

function randomWellConditionedMat3(): Mat3 {
  const s = 0.5;
  return Mat3.fromValues(
    s * rng.range(-1, 1) + 1.5,
    s * rng.range(-1, 1),
    s * rng.range(-1, 1),
    s * rng.range(-1, 1),
    s * rng.range(-1, 1) + 1.5,
    s * rng.range(-1, 1),
    s * rng.range(-1, 1),
    s * rng.range(-1, 1),
    s * rng.range(-1, 1) + 1.5,
  );
}

describe("MathUtils", () => {
  test("clamp bounds and interior", () => {
    expect(clamp(-2, 0, 1)).toBe(0);
    expect(clamp(2, 0, 1)).toBe(1);
    expect(clamp(0.25, 0, 1)).toBe(0.25);
  });

  test("approximatelyEqual respects tolerance", () => {
    expect(approximatelyEqual(1, 1 + 1e-13)).toBe(true);
    expect(approximatelyEqual(1, 1.001)).toBe(false);
    expect(approximatelyEqual(1, 1.5, 0.6)).toBe(true);
  });

  test("tolerances are positive and ordered", () => {
    expect(EPSILON).toBeGreaterThan(0);
    expect(EPSILON_SQ).toBeGreaterThan(0);
    expect(EPSILON_SQ).toBeLessThan(EPSILON);
  });
});

describe("Vec3", () => {
  test("construction, defaults and scalar field layout", () => {
    const zero = new Vec3();
    expect(zero.x).toBe(0);
    expect(zero.y).toBe(0);
    expect(zero.z).toBe(0);
    const partial = new Vec3(1);
    expect(partial.x).toBe(1);
    expect(partial.y).toBe(0);
    expect(partial.z).toBe(0);
    expect(Vec3.zero().equals(new Vec3(0, 0, 0))).toBe(true);
    expect(Vec3.fromValues(1, 2, 3).equals(new Vec3(1, 2, 3))).toBe(true);
    expect(zero.toString()).toBe("Vec3(0, 0, 0)");
  });

  test("set / setZero / copy / clone are independent", () => {
    const a = new Vec3(1, 2, 3);
    const b = a.clone();
    expect(b.equals(a)).toBe(true);
    b.set(4, 5, 6);
    expect(a.equals(new Vec3(1, 2, 3))).toBe(true);
    expect(new Vec3().copy(a).equals(a)).toBe(true);
    a.setZero();
    expect(a.equals(Vec3.zero())).toBe(true);
  });

  test("array interop round trip and bounds checking", () => {
    const buffer = new Float64Array(6);
    new Vec3(1, 2, 3).writeTo(buffer, 1);
    expectVecClose(Vec3.fromArray(buffer, 1), new Vec3(1, 2, 3), 0);
    expectVecClose(Vec3.fromArray([7, 8, 9]), new Vec3(7, 8, 9), 0);
    const out = new Vec3();
    expect(Vec3.fromArray([0, 0, 0, 4, 5, 6], 3, out)).toBe(out);
    expectVecClose(out, new Vec3(4, 5, 6), 0);
    expect(() => Vec3.fromArray([1, 2], 0)).toThrow(RangeError);
    expect(() => new Vec3().writeTo(new Float64Array(2), 0)).toThrow(RangeError);
  });

  test("instance arithmetic matches static out-parameter forms", () => {
    const a = new Vec3(1, -2, 3);
    const b = new Vec3(-4, 5, -6);
    expectVecClose(a.clone().add(b), Vec3.add(new Vec3(), a, b), 0);
    expectVecClose(a.clone().sub(b), Vec3.sub(new Vec3(), a, b), 0);
    expectVecClose(a.clone().scale(2.5), Vec3.scale(new Vec3(), a, 2.5), 0);
    expectVecClose(a.clone().addScaled(b, -1.5), Vec3.addScaled(new Vec3(), a, b, -1.5), 0);
    expectVecClose(a.clone().negate(), Vec3.scale(new Vec3(), a, -1), 0);
    expectVecClose(Vec3.negate(new Vec3(), a), new Vec3(-1, 2, -3), 0);
    expectVecClose(a.clone().cross(b), Vec3.cross(new Vec3(), a, b), 0);
  });

  test("static and instance operations are alias-safe", () => {
    const a = new Vec3(1, 2, 3);
    const doubled = a.clone();
    doubled.add(doubled);
    expectVecClose(doubled, new Vec3(2, 4, 6), 0);
    const viaStatic = new Vec3(1, 2, 3);
    Vec3.add(viaStatic, viaStatic, viaStatic);
    expectVecClose(viaStatic, new Vec3(2, 4, 6), 0);
    const self = new Vec3(1, 2, 3);
    self.cross(self);
    expectVecClose(self, Vec3.zero(), 0);
    const scaled = new Vec3(1, 2, 3);
    Vec3.scale(scaled, scaled, 2);
    expectVecClose(scaled, new Vec3(2, 4, 6), 0);
    const lerped = new Vec3(0, 0, 0);
    Vec3.lerp(lerped, lerped, new Vec3(2, 2, 2), 0.5);
    expectVecClose(lerped, new Vec3(1, 1, 1), 0);
  });

  test("dot, length and distance invariants (randomized)", () => {
    for (let i = 0; i < 200; i++) {
      const a = randomVec3(rng, -50, 50);
      const b = randomVec3(rng, -50, 50);
      expect(a.dot(b)).toBeCloseTo(Vec3.dot(b, a), 9);
      expect(a.lengthSq()).toBeCloseTo(a.dot(a), 6);
      expect(a.length()).toBeCloseTo(Math.sqrt(a.lengthSq()), 9);
      expect(a.distanceTo(b)).toBeCloseTo(b.distanceTo(a), 9);
      expect(Vec3.distanceSq(a, b)).toBeCloseTo(a.distanceTo(b) ** 2, 6);
      expect(a.equals(a, 0)).toBe(true);
      expect(a.isFinite()).toBe(true);
    }
    expect(new Vec3(Infinity, 0, 0).isFinite()).toBe(false);
    expect(Vec3.isFinite(new Vec3(0, NaN, 0))).toBe(false);
    expect(Vec3.isFinite(new Vec3(1, 2, 3))).toBe(true);
    expect(Vec3.equals(new Vec3(1, 1, 1), new Vec3(1.0001, 1, 1), 1e-3)).toBe(true);
    expect(new Vec3(1, 1, 1).equals(new Vec3(2, 1, 1), 0.5)).toBe(false);
  });

  test("cross product is right-handed and orthogonal", () => {
    expectVecClose(
      Vec3.cross(new Vec3(), new Vec3(1, 0, 0), new Vec3(0, 1, 0)),
      new Vec3(0, 0, 1),
      0,
    );
    expectVecClose(
      Vec3.cross(new Vec3(), new Vec3(0, 1, 0), new Vec3(0, 0, 1)),
      new Vec3(1, 0, 0),
      0,
    );
    expectVecClose(
      Vec3.cross(new Vec3(), new Vec3(0, 0, 1), new Vec3(1, 0, 0)),
      new Vec3(0, 1, 0),
      0,
    );
    for (let i = 0; i < 200; i++) {
      const a = randomVec3(rng);
      const b = randomVec3(rng);
      const c = Vec3.cross(new Vec3(), a, b);
      expect(Math.abs(c.dot(a))).toBeLessThan(1e-12);
      expect(Math.abs(c.dot(b))).toBeLessThan(1e-12);
      expectVecClose(Vec3.cross(new Vec3(), b, a), c.clone().negate(), 1e-12);
      const sine = c.length() / (a.length() * b.length());
      const angle = Math.acos(clamp(a.dot(b) / (a.length() * b.length()), -1, 1));
      expect(sine).toBeCloseTo(Math.sin(angle), 9);
    }
  });

  test("triple product matches chained crosses and the Lagrange identity", () => {
    for (let i = 0; i < 100; i++) {
      const a = randomVec3(rng);
      const b = randomVec3(rng);
      const c = randomVec3(rng);
      const expected = Vec3.cross(new Vec3(), a, Vec3.cross(new Vec3(), b, c));
      expectVecClose(Vec3.tripleProduct(new Vec3(), a, b, c), expected, 1e-12);
      const lagrange = b.clone().scale(a.dot(c)).addScaled(c, -a.dot(b));
      expectVecClose(Vec3.tripleProduct(new Vec3(), a, b, c), lagrange, 1e-12);
      expect(a.dot(Vec3.cross(new Vec3(), b, c))).toBeCloseTo(
        b.dot(Vec3.cross(new Vec3(), c, a)),
        9,
      );
    }
  });

  test("lerp, distance and reflect", () => {
    const a = new Vec3(-1, 2, 4);
    const b = new Vec3(3, -2, 0);
    expectVecClose(Vec3.lerp(new Vec3(), a, b, 0), a, 0);
    expectVecClose(Vec3.lerp(new Vec3(), a, b, 1), b, 0);
    expectVecClose(Vec3.lerp(new Vec3(), a, b, 0.5), new Vec3(1, 0, 2), 0);
    expectVecClose(Vec3.lerp(new Vec3(), a, b, 2), new Vec3(7, -6, -4), 0);
    expect(Vec3.distance(a, b)).toBeCloseTo(Math.sqrt(48), 9);

    const normal = new Vec3(0, 1, 0);
    const incoming = new Vec3(1, -1, 0);
    const reflected = Vec3.reflect(new Vec3(), incoming, normal);
    expectVecClose(reflected, new Vec3(1, 1, 0), 1e-15);
    expect(reflected.length()).toBeCloseTo(incoming.length(), 12);
    expectVecClose(Vec3.reflect(new Vec3(), reflected, normal), incoming, 1e-15);
    const perpendicular = new Vec3(1, 0, 0);
    expectVecClose(Vec3.reflect(new Vec3(), perpendicular, normal), perpendicular, 1e-15);
    const opposite = new Vec3(0, -3, 0);
    expectVecClose(Vec3.reflect(new Vec3(), opposite, normal), opposite.clone().negate(), 1e-15);
    const unitNormal = randomUnitVec3(rng);
    const arbitrary = randomVec3(rng);
    const mirrored = Vec3.reflect(new Vec3(), arbitrary, unitNormal);
    expect(mirrored.length()).toBeCloseTo(arbitrary.length(), 12);
    expectVecClose(Vec3.reflect(new Vec3(), mirrored, unitNormal), arbitrary, 1e-12);
  });

  test("normalize handles general, zero and fallback directions", () => {
    const v = new Vec3(3, 0, 4);
    expectVecClose(v.clone().normalize(), new Vec3(0.6, 0, 0.8), 1e-15);
    expectVecClose(Vec3.normalize(new Vec3(), v), new Vec3(0.6, 0, 0.8), 1e-15);
    expectVecClose(Vec3.normalize(new Vec3(), new Vec3()), Vec3.zero(), 0);
    const tiny = new Vec3(1e-14, 0, 0);
    expectVecClose(Vec3.normalize(new Vec3(), tiny), Vec3.zero(), 0);
    expectVecClose(Vec3.normalize(new Vec3(), tiny, new Vec3(0, 1, 0)), new Vec3(0, 1, 0), 0);
    expectVecClose(Vec3.normalize(new Vec3(), v, new Vec3(0, 1, 0)), new Vec3(0.6, 0, 0.8), 1e-15);
  });

  test("component-wise min and max", () => {
    const a = new Vec3(1, -5, 3);
    const b = new Vec3(2, 4, -1);
    expectVecClose(Vec3.min(new Vec3(), a, b), new Vec3(1, -5, -1), 0);
    expectVecClose(Vec3.max(new Vec3(), a, b), new Vec3(2, 4, 3), 0);
    expectVecClose(Vec3.min(new Vec3(), a, a.clone()), a, 0);
  });
});

describe("Mat3", () => {
  const randomRotation = (): Mat3 => randomQuat(rng).toMat3();

  test("construction, identity, copy and diagnostics", () => {
    const identity = new Mat3();
    expect(identity.m00).toBe(1);
    expect(identity.m11).toBe(1);
    expect(identity.m22).toBe(1);
    expect(identity.m01).toBe(0);
    expect(identity.isFinite()).toBe(true);
    const m = Mat3.fromValues(1, 2, 3, 4, 5, 6, 7, 8, 9);
    expect(m.m22).toBe(9);
    const copy = m.clone();
    expect(copy.equals(m)).toBe(true);
    expect(m.equals(m, 1e-12)).toBe(true);
    expect(m.toString()).toContain("Mat3[1, 2, 3");
    expect(Mat3.fromDiagonal(2, 3, 4).m11).toBe(3);
    expect(new Mat3().set(1, 1, 1, 1, 1, 1, 1, 1, 1).equals(Mat3.fromValues(1, 1, 1, 1, 1, 1, 1, 1, 1))).toBe(true);
    expect(Mat3.fromValues(1, 2, 3, 4, 5, 6, 7, 8, 9).identity().equals(new Mat3())).toBe(true);
    expect(Mat3.zero(new Mat3()).equals(Mat3.fromValues(0, 0, 0, 0, 0, 0, 0, 0, 0))).toBe(true);
    expect(Mat3.zero(new Mat3()).equals(new Mat3())).toBe(false);
    expect(new Mat3().set(Infinity, 0, 0, 0, 0, 0, 0, 0, 0).isFinite()).toBe(false);
  });

  test("array interop round trip and bounds checking", () => {
    const buffer = new Float32Array(11);
    const m = Mat3.fromValues(1, 2, 3, 4, 5, 6, 7, 8, 9);
    m.writeTo(buffer, 2);
    expectMat3Close(Mat3.fromArray(buffer, 2), m, 1e-6);
    expectMat3Close(Mat3.fromArray([1, 2, 3, 4, 5, 6, 7, 8, 9]), m, 0);
    const out = new Mat3();
    expect(Mat3.fromArray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 1, out)).toBe(out);
    expect(() => Mat3.fromArray([1, 2], 0)).toThrow(RangeError);
    expect(() => m.writeTo(new Float64Array(3), 0)).toThrow(RangeError);
  });

  test("multiply matches manual computation and is associative", () => {
    const a = Mat3.fromValues(1, 2, 3, 4, 5, 6, 7, 8, 9);
    const b = Mat3.fromValues(9, 8, 7, 6, 5, 4, 3, 2, 1);
    const product = Mat3.multiply(new Mat3(), a, b);
    expect(product.m00).toBe(1 * 9 + 2 * 6 + 3 * 3);
    expect(product.m01).toBe(1 * 8 + 2 * 5 + 3 * 2);
    expect(product.m22).toBe(7 * 7 + 8 * 4 + 9 * 1);
    const c = Mat3.fromValues(1, 0, 2, -1, 3, 0, 0, 1, 1);
    const left = Mat3.multiply(new Mat3(), Mat3.multiply(new Mat3(), a, b), c);
    const right = Mat3.multiply(new Mat3(), a, Mat3.multiply(new Mat3(), b, c));
    expectMat3Close(left, right, 1e-12);
    expectMat3Close(a.clone().multiply(b), product, 0);
    const alias = a.clone();
    Mat3.multiply(alias, alias, b);
    expectMat3Close(alias, product, 0);
    expectMat3Close(new Mat3().multiply(Mat3.fromDiagonal(2, 3, 4)), Mat3.fromDiagonal(2, 3, 4), 0);
  });

  test("transpose is an involution and inverts products", () => {
    for (let i = 0; i < 50; i++) {
      const a = randomWellConditionedMat3();
      const b = randomWellConditionedMat3();
      const transpose = Mat3.transpose(new Mat3(), a);
      expect(transpose.m01).toBe(a.m10);
      expect(transpose.m20).toBe(a.m02);
      expectMat3Close(Mat3.transpose(new Mat3(), transpose), a, 1e-12);
      const productTranspose = Mat3.transpose(new Mat3(), Mat3.multiply(new Mat3(), a, b));
      const reversed = Mat3.multiply(
        new Mat3(),
        Mat3.transpose(new Mat3(), b),
        Mat3.transpose(new Mat3(), a),
      );
      expectMat3Close(productTranspose, reversed, 1e-9);
      expectMat3Close(a.clone().transpose(), transpose, 0);
    }
  });

  test("determinant: known values and multiplicativity", () => {
    expect(new Mat3().determinant()).toBe(1);
    expect(Mat3.fromDiagonal(2, 3, 4).determinant()).toBe(24);
    expect(Mat3.fromValues(1, 2, 3, 4, 5, 6, 7, 8, 9).determinant()).toBeCloseTo(0, 12);
    for (let i = 0; i < 50; i++) {
      const a = randomWellConditionedMat3();
      const b = randomWellConditionedMat3();
      expect(Mat3.determinant(Mat3.multiply(new Mat3(), a, b))).toBeCloseTo(
        a.determinant() * b.determinant(),
        6,
      );
      expect(Mat3.determinant(randomRotation())).toBeCloseTo(1, 9);
    }
  });

  test("invert satisfies M * M^-1 = I and is an involution", () => {
    for (let i = 0; i < 200; i++) {
      const m = randomWellConditionedMat3();
      const inverse = new Mat3();
      expect(Mat3.invert(inverse, m)).toBe(true);
      expectMat3Close(Mat3.multiply(new Mat3(), m, inverse), new Mat3(), 1e-9);
      expectMat3Close(Mat3.multiply(new Mat3(), inverse, m), new Mat3(), 1e-9);
      const roundTrip = new Mat3();
      expect(Mat3.invert(roundTrip, inverse)).toBe(true);
      expectMat3Close(roundTrip, m, 1e-6);
    }
    const rotation = randomRotation();
    const rotationInverse = new Mat3();
    expect(Mat3.invert(rotationInverse, rotation)).toBe(true);
    expectMat3Close(rotationInverse, Mat3.transpose(new Mat3(), rotation), 1e-12);
    const inPlace = randomWellConditionedMat3();
    const copy = inPlace.clone();
    expect(inPlace.invert()).toBe(true);
    expectMat3Close(Mat3.multiply(new Mat3(), copy, inPlace), new Mat3(), 1e-9);
  });

  test("singular matrices report failure and zero the output", () => {
    const singular = Mat3.fromValues(1, 2, 3, 2, 4, 6, 7, 8, 9);
    const out = Mat3.fromValues(1, 1, 1, 1, 1, 1, 1, 1, 1);
    expect(Mat3.invert(out, singular)).toBe(false);
    expect(out.equals(Mat3.zero(new Mat3()))).toBe(true);
    expect(Mat3.zero(new Mat3()).invert()).toBe(false);
  });

  test("vector transforms (forward, transposed, alias)", () => {
    const m = Mat3.fromValues(1, 2, 3, 4, 5, 6, 7, 8, 9);
    const v = new Vec3(1, -2, 0.5);
    const product = Mat3.transformVector(new Vec3(), m, v);
    expect(product.x).toBeCloseTo(1 * 1 + 2 * -2 + 3 * 0.5, 12);
    expect(product.y).toBeCloseTo(4 * 1 + 5 * -2 + 6 * 0.5, 12);
    expect(product.z).toBeCloseTo(7 * 1 + 8 * -2 + 9 * 0.5, 12);
    const transpose = Mat3.transpose(new Mat3(), m);
    expectVecClose(
      Mat3.transformVectorTranspose(new Vec3(), m, v),
      Mat3.transformVector(new Vec3(), transpose, v),
      1e-12,
    );
    for (let i = 0; i < 20; i++) {
      const rotation = randomRotation();
      const direction = randomVec3(rng);
      const forward = Mat3.transformVector(new Vec3(), rotation, direction);
      expectVecClose(Mat3.transformVectorTranspose(new Vec3(), rotation, forward), direction, 1e-12);
      expect(forward.length()).toBeCloseTo(direction.length(), 12);
    }
    const aliasVector = v.clone();
    Mat3.transformVector(aliasVector, m, aliasVector);
    expectVecClose(aliasVector, product, 1e-12);
    const aliasTranspose = v.clone();
    Mat3.transformVectorTranspose(aliasTranspose, m, aliasTranspose);
    expectVecClose(aliasTranspose, Mat3.transformVector(new Vec3(), transpose, v), 1e-12);
  });

  test("skew-symmetric operator reproduces the cross product", () => {
    for (let i = 0; i < 100; i++) {
      const r = randomVec3(rng, -10, 10);
      const v = randomVec3(rng, -10, 10);
      const skew = Mat3.skew(new Mat3(), r);
      expect(skew.m00).toBe(0);
      expect(skew.m01).toBe(-r.z);
      expect(skew.m12).toBe(-r.x);
      expect(skew.m20).toBe(-r.y);
      expectVecClose(Mat3.transformVector(new Vec3(), skew, v), Vec3.cross(new Vec3(), r, v), 1e-12);
      expectMat3Close(Mat3.transpose(new Mat3(), skew).multiplyScalar(-1), skew, 1e-15);
    }
  });

  test("inertia tensor transformation R I R^T", () => {
    for (let i = 0; i < 100; i++) {
      const rotation = randomRotation();
      const bodyInertia = Mat3.fromDiagonal(rng.range(0.5, 3), rng.range(0.5, 3), rng.range(0.5, 3));
      const world = Mat3.transformInertia(new Mat3(), rotation, bodyInertia);
      expect(world.m01).toBeCloseTo(world.m10, 12);
      expect(world.m02).toBeCloseTo(world.m20, 12);
      expect(world.m12).toBeCloseTo(world.m21, 12);
      expect(world.m00 + world.m11 + world.m22).toBeCloseTo(
        bodyInertia.m00 + bodyInertia.m11 + bodyInertia.m22,
        9,
      );
      const bodyVelocity = randomVec3(rng, -5, 5);
      const bodyMomentum = Mat3.transformVector(new Vec3(), bodyInertia, bodyVelocity);
      const worldVelocity = Mat3.transformVector(new Vec3(), rotation, bodyVelocity);
      expectVecClose(
        Mat3.transformVector(new Vec3(), world, worldVelocity),
        Mat3.transformVector(new Vec3(), rotation, bodyMomentum),
        1e-9,
      );
    }
  });

  test("multiplyScalar scales in place or into a separate output", () => {
    const m = Mat3.fromValues(1, 2, 3, 4, 5, 6, 7, 8, 9);
    const scaled = Mat3.multiplyScalar(new Mat3(), m, 2);
    expect(scaled.m22).toBe(18);
    expect(m.m22).toBe(9);
    expectMat3Close(
      m.clone().multiplyScalar(0.5),
      Mat3.fromValues(0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5),
      0,
    );
  });
});

describe("Mat4", () => {
  test("construction, identity and diagnostics", () => {
    const identity = new Mat4();
    expect(identity.m00).toBe(1);
    expect(identity.m11).toBe(1);
    expect(identity.m22).toBe(1);
    expect(identity.m33).toBe(1);
    expect(identity.m03).toBe(0);
    expect(identity.isFinite()).toBe(true);
    const m = Mat4.fromValues(
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16,
    );
    expect(m.m33).toBe(16);
    expect(m.clone().equals(m)).toBe(true);
    expect(m.equals(Mat4.zero(new Mat4()))).toBe(false);
    expect(m.toString()).toContain("Mat4[1, 2, 3, 4");
    expect(Mat4.zero(new Mat4()).equals(new Mat4().set(
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
    ))).toBe(true);
    expect(new Mat4().set(
      NaN, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
    ).isFinite()).toBe(false);
    expect(new Mat4().identity().equals(new Mat4())).toBe(true);
    expect(new Mat4().copy(new Mat4()).equals(new Mat4())).toBe(true);
  });

  test("array interop round trip and bounds checking", () => {
    const buffer = new Float64Array(20);
    const m = Mat4.fromValues(
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16,
    );
    m.writeTo(buffer, 4);
    expectMat4Close(Mat4.fromArray(buffer, 4), m, 0);
    const out = new Mat4();
    const values = [0, ...buffer.slice(4, 20)];
    expect(Mat4.fromArray(values, 1, out)).toBe(out);
    expectMat4Close(out, m, 0);
    expect(() => Mat4.fromArray([1, 2], 0)).toThrow(RangeError);
    expect(() => m.writeTo(new Float64Array(4), 0)).toThrow(RangeError);
  });

  test("multiply, transpose and determinant identities", () => {
    const a = Mat4.fromValues(
      2, 0, 0, 1,
      0, 3, 0, 2,
      0, 0, 4, 3,
      0, 0, 0, 1,
    );
    const b = Mat4.fromValues(
      1, 0, 0, -1,
      0, 1, 0, -2,
      0, 0, 1, -3,
      0, 0, 0, 1,
    );
    const product = Mat4.multiply(new Mat4(), a, b);
    expect(product.m03).toBe(2 * -1 + 1);
    expect(product.m13).toBe(3 * -2 + 2);
    expect(product.m23).toBe(4 * -3 + 3);
    expectMat4Close(a.clone().multiply(b), product, 0);
    const alias = a.clone();
    Mat4.multiply(alias, alias, b);
    expectMat4Close(alias, product, 0);
    const transpose = Mat4.transpose(new Mat4(), a);
    expect(transpose.m30).toBe(a.m03);
    expect(transpose.m03).toBe(a.m30);
    expectMat4Close(Mat4.transpose(new Mat4(), transpose), a, 0);
    expect(a.clone().transpose().equals(transpose)).toBe(true);
    expect(a.determinant()).toBeCloseTo(2 * 3 * 4, 9);
    expect(Mat4.determinant(product)).toBeCloseTo(a.determinant() * b.determinant(), 9);
  });

  test("invert satisfies M * M^-1 = I for affine and general matrices", () => {
    for (let i = 0; i < 100; i++) {
      const rotation = randomQuat(rng);
      const translation = randomVec3(rng, -10, 10);
      const affine = Mat4.fromRotationTranslation(new Mat4(), rotation, translation);
      const inverse = new Mat4();
      expect(Mat4.invert(inverse, affine)).toBe(true);
      expectMat4Close(Mat4.multiply(new Mat4(), affine, inverse), new Mat4(), 1e-9);
      const roundTrip = new Mat4();
      expect(Mat4.invert(roundTrip, inverse)).toBe(true);
      expectMat4Close(roundTrip, affine, 1e-7);
      const general = Mat4.fromValues(
        rng.range(-1, 1) + 2, rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1),
        rng.range(-1, 1), rng.range(-1, 1) + 2, rng.range(-1, 1), rng.range(-1, 1),
        rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1) + 2, rng.range(-1, 1),
        rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1) + 2,
      );
      const generalInverse = new Mat4();
      expect(Mat4.invert(generalInverse, general)).toBe(true);
      expectMat4Close(Mat4.multiply(new Mat4(), general, generalInverse), new Mat4(), 1e-9);
    }
    const inPlace = Mat4.fromValues(
      2, 0, 0, 1,
      0, 2, 0, 0,
      0, 0, 2, 0,
      0, 0, 0, 1,
    );
    expect(inPlace.invert()).toBe(true);
    expect(inPlace.m00).toBeCloseTo(0.5, 12);
    expect(inPlace.m03).toBeCloseTo(-0.5, 12);
  });

  test("singular matrices report failure and zero the output", () => {
    const singular = Mat4.fromValues(
      1, 2, 3, 4,
      2, 4, 6, 8,
      1, 0, 0, 0,
      0, 1, 0, 0,
    );
    const out = new Mat4();
    expect(Mat4.invert(out, singular)).toBe(false);
    expect(out.equals(Mat4.zero(new Mat4()))).toBe(true);
  });

  test("transformPoint, transformDirection and perspective divide", () => {
    const scale = Mat4.fromValues(
      2, 0, 0, 1,
      0, 3, 0, 2,
      0, 0, 4, 3,
      0, 0, 0, 1,
    );
    const point = new Vec3(1, 1, 1);
    expectVecClose(Mat4.transformPoint(new Vec3(), scale, point), new Vec3(3, 5, 7), 1e-12);
    expectVecClose(Mat4.transformDirection(new Vec3(), scale, point), new Vec3(2, 3, 4), 1e-12);
    const alias = point.clone();
    Mat4.transformPoint(alias, scale, alias);
    expectVecClose(alias, new Vec3(3, 5, 7), 1e-12);
    const aliasDirection = point.clone();
    Mat4.transformDirection(aliasDirection, scale, aliasDirection);
    expectVecClose(aliasDirection, new Vec3(2, 3, 4), 1e-12);
    const projective = Mat4.fromValues(
      2, 0, 0, 0,
      0, 2, 0, 0,
      0, 0, 2, 0,
      0, 0, 0, 2,
    );
    expectVecClose(Mat4.transformPoint(new Vec3(), projective, point), point, 1e-12);

    const rotation = randomQuat(rng);
    const translation = randomVec3(rng, -5, 5);
    const transform = new Transform(translation, rotation);
    const matrix = Mat4.fromRotationTranslation(new Mat4(), rotation, translation);
    const probe = randomVec3(rng, -5, 5);
    expectVecClose(
      Mat4.transformPoint(new Vec3(), matrix, probe),
      transform.transformPoint(new Vec3(), probe),
      1e-12,
    );
    expectMat4Close(transform.toMat4(), matrix, 1e-12);
    expectVecClose(
      Mat4.transformDirection(new Vec3(), matrix, probe),
      transform.transformDirection(new Vec3(), probe),
      1e-12,
    );
  });

  test("fromMat3 embeds the linear block with an identity border", () => {
    const rotation = randomQuat(rng).toMat3();
    const matrix = Mat4.fromMat3(new Mat4(), rotation);
    expect(matrix.m00).toBe(rotation.m00);
    expect(matrix.m22).toBe(rotation.m22);
    expect(matrix.m03).toBe(0);
    expect(matrix.m33).toBe(1);
    expect(matrix.m30).toBe(0);
  });
});

describe("Quat", () => {
  test("construction, conjugate, dot and normalization", () => {
    const identity = new Quat();
    expect(identity.w).toBe(1);
    expect(identity.x).toBe(0);
    const q = Quat.fromValues(0.5, 0.5, 0.5, 0.5);
    expect(q.dot(q)).toBeCloseTo(1, 12);
    expect(q.lengthSq()).toBeCloseTo(1, 12);
    expect(q.length()).toBeCloseTo(1, 12);
    const copy = q.clone();
    expect(copy.equals(q)).toBe(true);
    expect(q.equals(copy.set(1, 0, 0, 0))).toBe(false);
    expect(q.equals(q, 1e-12)).toBe(true);
    expect(q.toString()).toContain("Quat(0.5, 0.5");
    expect(new Quat().identity().equals(new Quat())).toBe(true);
    expect(Quat.identity().equals(new Quat())).toBe(true);
    expect(Quat.fromValues(2, 0, 0, 0).clone().normalize().equals(new Quat())).toBe(true);
    const v = new Vec3(1, 2, 3);
    const conjugate = Quat.conjugate(new Quat(), q);
    expect(q.clone().conjugate().equals(conjugate)).toBe(true);
    expectVecClose(q.rotateVector(conjugate.rotateVector(v, new Vec3()), new Vec3()), v, 1e-12);
    expect(() => new Quat().set(0, 0, 0, 0).normalize()).toThrow(RangeError);
    expect(() => Quat.normalize(new Quat(), Quat.fromValues(0, 0, 0, 0))).toThrow(RangeError);
    expect(Quat.dot(q, q)).toBeCloseTo(1, 12);
    expect(Quat.lengthSq(q)).toBeCloseTo(1, 12);
  });

  test("array interop round trip and bounds checking", () => {
    const buffer = new Float64Array(8);
    const q = Quat.fromValues(0.5, -0.5, 0.5, -0.5);
    q.writeTo(buffer, 2);
    expect(Quat.fromArray(buffer, 2).equals(q)).toBe(true);
    expect(Quat.fromArray([1, 0, 0, 0]).equals(new Quat())).toBe(true);
    const out = new Quat();
    expect(Quat.fromArray([0, 1, 0, 0, 0], 1, out)).toBe(out);
    expect(out.equals(new Quat())).toBe(true);
    expect(() => Quat.fromArray([1, 2], 0)).toThrow(RangeError);
    expect(() => new Quat().writeTo(new Float64Array(3), 0)).toThrow(RangeError);
  });

  test("fromAxisAngle produces the documented rotation", () => {
    const q = Quat.fromAxisAngle(new Vec3(0, 0, 1), Math.PI / 2);
    expect(q.w).toBeCloseTo(Math.SQRT1_2, 12);
    expect(q.z).toBeCloseTo(Math.SQRT1_2, 12);
    expectVecClose(q.rotateVector(new Vec3(1, 0, 0), new Vec3()), new Vec3(0, 1, 0), 1e-12);
    expectSameRotation(Quat.fromAxisAngle(new Vec3(0, 0, 5), Math.PI / 2), q, 1e-12);
    const out = new Quat();
    expect(Quat.fromAxisAngle(new Vec3(1, 0, 0), 0.5, out)).toBe(out);
    expect(() => Quat.fromAxisAngle(new Vec3(0, 0, 0), 1)).toThrow(RangeError);
  });

  test("multiplication matches rotation matrix composition (randomized)", () => {
    for (let i = 0; i < 200; i++) {
      const a = randomQuat(rng);
      const b = randomQuat(rng);
      const combined = Quat.multiply(new Quat(), a, b);
      expectMat3Close(combined.toMat3(), Mat3.multiply(new Mat3(), a.toMat3(), b.toMat3()), 1e-12);
      expectSameRotation(a.clone().multiply(b), combined, 1e-12);
      const v = randomVec3(rng, -5, 5);
      const sequential = a.rotateVector(b.rotateVector(v, new Vec3()), new Vec3());
      expectVecClose(combined.rotateVector(v, new Vec3()), sequential, 1e-12);
      expectVecClose(
        Mat3.transformVector(new Vec3(), combined.toMat3(), v),
        combined.rotateVector(v, new Vec3()),
        1e-12,
      );
    }
  });

  test("rotations preserve length and inverse rotations undo them", () => {
    for (let i = 0; i < 100; i++) {
      const q = randomQuat(rng);
      const inverse = Quat.inverse(new Quat(), q);
      expectSameRotation(inverse, Quat.conjugate(new Quat(), q), 1e-12);
      const v = randomVec3(rng, -10, 10);
      const rotated = q.rotateVector(v, new Vec3());
      expect(rotated.length()).toBeCloseTo(v.length(), 12);
      expectVecClose(inverse.rotateVector(rotated, new Vec3()), v, 1e-12);
      expectSameRotation(Quat.multiply(new Quat(), q, inverse), new Quat(), 1e-12);
    }
    expect(() => Quat.inverse(new Quat(), Quat.fromValues(0, 0, 0, 0))).toThrow(RangeError);
  });

  test("toMat3 is orthonormal with unit determinant", () => {
    for (let i = 0; i < 100; i++) {
      const q = randomQuat(rng);
      const matrix = q.toMat3();
      expectMat3Close(
        Mat3.multiply(new Mat3(), Mat3.transpose(new Mat3(), matrix), matrix),
        new Mat3(),
        1e-12,
      );
      expect(matrix.determinant()).toBeCloseTo(1, 12);
      expectMat3Close(Quat.toMat3(new Mat3(), q), matrix, 0);
      expectMat3Close(q.toMat3(new Mat3()), matrix, 0);
    }
    const ninetyDegrees = Quat.fromAxisAngle(new Vec3(0, 0, 1), Math.PI / 2);
    const matrix = ninetyDegrees.toMat3();
    expect(matrix.m00).toBeCloseTo(0, 12);
    expect(matrix.m01).toBeCloseTo(-1, 12);
    expect(matrix.m10).toBeCloseTo(1, 12);
    expect(matrix.m22).toBeCloseTo(1, 12);
  });

  test("slerp interpolates along the shortest arc and clamps t", () => {
    const start = Quat.fromAxisAngle(new Vec3(0, 0, 1), 0);
    const end = Quat.fromAxisAngle(new Vec3(0, 0, 1), Math.PI / 2);
    expectSameRotation(Quat.slerp(new Quat(), start, end, 0), start, 1e-12);
    expectSameRotation(Quat.slerp(new Quat(), start, end, 1), end, 1e-12);
    expectSameRotation(Quat.slerp(new Quat(), start, end, -1), start, 1e-12);
    expectSameRotation(Quat.slerp(new Quat(), start, end, 2), end, 1e-12);
    const midpoint = Quat.slerp(new Quat(), start, end, 0.5);
    const expectedMidpoint = Quat.fromAxisAngle(new Vec3(0, 0, 1), Math.PI / 4);
    expect(quatRotationAngle(midpoint, expectedMidpoint)).toBeLessThan(1e-12);

    const tenDegrees = Quat.fromAxisAngle(new Vec3(0, 0, 1), (10 * Math.PI) / 180);
    const threeFiftyDegrees = Quat.fromAxisAngle(new Vec3(0, 0, 1), (350 * Math.PI) / 180);
    const shortArcMidpoint = Quat.slerp(new Quat(), tenDegrees, threeFiftyDegrees, 0.5);
    expect(quatRotationAngle(shortArcMidpoint, new Quat())).toBeLessThan(1e-12);
    const quarter = Quat.slerp(new Quat(), tenDegrees, threeFiftyDegrees, 0.25);
    const expectedQuarter = Quat.fromAxisAngle(new Vec3(0, 0, 1), (5 * Math.PI) / 180);
    expect(quatRotationAngle(quarter, expectedQuarter)).toBeLessThan(1e-12);

    const identical = randomQuat(rng);
    expectSameRotation(Quat.slerp(new Quat(), identical, identical.clone(), 0.5), identical, 1e-12);
  });

  test("angular velocity integration matches the closed form for constant omega", () => {
    const omega = new Vec3(0, 0, 2);
    const q = new Quat();
    const dt = 1 / 2000;
    const totalTime = 1.5;
    for (let i = 0; i < 3000; i++) {
      q.integrateAngularVelocity(omega, dt);
      expect(q.length()).toBeCloseTo(1, 14);
    }
    expect(quatRotationAngle(q, Quat.fromAxisAngle(new Vec3(0, 0, 1), 2 * totalTime))).toBeLessThan(1e-4);

    const xRotation = new Quat();
    for (let i = 0; i < 4000; i++) {
      Quat.integrateAngularVelocity(xRotation, xRotation, new Vec3(Math.PI / 2, 0, 0), 1 / 4000);
    }
    expectVecClose(xRotation.rotateVector(new Vec3(0, 1, 0), new Vec3()), new Vec3(0, 0, 1), 1e-4);
    expectSameRotation(
      Quat.integrateAngularVelocity(new Quat(), new Quat(), omega, 0),
      new Quat(),
      0,
    );
  });

  test("angular integration error is second order in dt", () => {
    const measureError = (dt: number, totalTime: number): number => {
      const q = new Quat();
      const omega = new Vec3(0, 0, 2);
      const steps = Math.round(totalTime / dt);
      for (let i = 0; i < steps; i++) {
        Quat.integrateAngularVelocity(q, q, omega, dt);
      }
      return quatRotationAngle(q, Quat.fromAxisAngle(new Vec3(0, 0, 1), 2 * totalTime));
    };
    const coarse = measureError(0.1, 1);
    const fine = measureError(0.05, 1);
    expect(coarse).toBeGreaterThan(1e-4);
    expect(coarse / fine).toBeGreaterThan(3.5);
    expect(coarse / fine).toBeLessThan(4.5);
  });

  test("angular integration tracks an RK4 reference for time-varying omega", () => {
    const omegaAt = (t: number): Vec3 => new Vec3(1.5 * Math.sin(2 * t), 0.9 * Math.cos(3 * t), 0.6);
    const derivative = (out: Quat, q: Quat, omega: Vec3): Quat => {
      out.set(
        -0.5 * (omega.x * q.x + omega.y * q.y + omega.z * q.z),
        0.5 * (q.w * omega.x + omega.y * q.z - omega.z * q.y),
        0.5 * (q.w * omega.y - omega.x * q.z + omega.z * q.x),
        0.5 * (q.w * omega.z + omega.x * q.y - omega.y * q.x),
      );
      return out;
    };
    const addScaled = (out: Quat, base: Quat, delta: Quat, scale: number): Quat => {
      out.set(
        base.w + delta.w * scale,
        base.x + delta.x * scale,
        base.y + delta.y * scale,
        base.z + delta.z * scale,
      );
      return out;
    };
    const referenceAt = (dt: number, totalTime: number): Quat => {
      const steps = Math.round(totalTime / dt);
      const reference = new Quat();
      const stage = new Quat();
      for (let i = 0; i < steps; i++) {
        const t = i * dt;
        const k1 = derivative(new Quat(), reference, omegaAt(t));
        const k2 = derivative(new Quat(), addScaled(stage, reference, k1, dt / 2), omegaAt(t + dt / 2));
        const k3 = derivative(new Quat(), addScaled(stage, reference, k2, dt / 2), omegaAt(t + dt / 2));
        const k4 = derivative(new Quat(), addScaled(stage, reference, k3, dt), omegaAt(t + dt));
        const sixth = dt / 6;
        Quat.normalize(
          reference,
          stage.set(
            reference.w + sixth * (k1.w + 2 * k2.w + 2 * k3.w + k4.w),
            reference.x + sixth * (k1.x + 2 * k2.x + 2 * k3.x + k4.x),
            reference.y + sixth * (k1.y + 2 * k2.y + 2 * k3.y + k4.y),
            reference.z + sixth * (k1.z + 2 * k2.z + 2 * k3.z + k4.z),
          ),
        );
      }
      return reference;
    };
    const integratedAt = (dt: number, totalTime: number): Quat => {
      const steps = Math.round(totalTime / dt);
      const integrated = new Quat();
      for (let i = 0; i < steps; i++) {
        Quat.integrateAngularVelocity(integrated, integrated, omegaAt(i * dt), dt);
      }
      return integrated;
    };
    const errorAt = (dt: number): number =>
      quatRotationAngle(integratedAt(dt, 2), referenceAt(dt, 2));

    const coarse = errorAt(1e-3);
    const fine = errorAt(1e-4);
    const finest = errorAt(1e-5);
    expect(coarse).toBeGreaterThan(1e-4);
    expect(fine).toBeLessThan(3e-4);
    expect(coarse / fine).toBeGreaterThan(4);
    expect(coarse / fine).toBeLessThan(20);
    expect(finest).toBeLessThan(3e-5);
  });
});
