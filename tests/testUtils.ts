import { Mat3 } from "../src/math/Mat3.js";
import { Mat4 } from "../src/math/Mat4.js";
import { Quat } from "../src/math/Quat.js";
import { Vec3 } from "../src/math/Vec3.js";

/**
 * Deterministic 32-bit PRNG (mulberry32). Every randomized test draws from a
 * fixed seed so failures are reproducible.
 */
export interface SeededRng {
  next(): number;
  range(min: number, max: number): number;
}

export function makeRng(seed: number): SeededRng {
  let state = seed >>> 0;
  const next = (): number => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    next,
    range: (min: number, max: number): number => min + (max - min) * next(),
  };
}

/** Uniformly random vector with components in `[min, max]`. */
export function randomVec3(rng: SeededRng, min = -1, max = 1): Vec3 {
  return new Vec3(rng.range(min, max), rng.range(min, max), rng.range(min, max));
}

/** Random unit vector (rejection sampling on the cube). */
export function randomUnitVec3(rng: SeededRng): Vec3 {
  for (;;) {
    const v = randomVec3(rng);
    const lengthSq = v.lengthSq();
    if (lengthSq > 1e-4 && lengthSq <= 1) {
      return v.normalize();
    }
  }
}

/** Random unit quaternion (normalized uniform cube sample). */
export function randomQuat(rng: SeededRng): Quat {
  for (;;) {
    const q = Quat.fromValues(
      rng.range(-1, 1),
      rng.range(-1, 1),
      rng.range(-1, 1),
      rng.range(-1, 1),
    );
    if (q.lengthSq() > 0.05) {
      return q.normalize();
    }
  }
}

/** Rotation angle (radians) between two unit quaternions representing rotations. */
export function quatRotationAngle(a: Quat, b: Quat): number {
  const relative = Quat.multiply(new Quat(), Quat.conjugate(new Quat(), a), b);
  const vectorLength = Math.hypot(relative.x, relative.y, relative.z);
  return 2 * Math.atan2(vectorLength, Math.abs(relative.w));
}

/** Asserts component-wise closeness with a readable failure message. */
export function expectVecClose(
  actual: Vec3,
  expected: Vec3,
  tolerance = 1e-12,
  label = "Vec3",
): void {
  const error = Math.max(
    Math.abs(actual.x - expected.x),
    Math.abs(actual.y - expected.y),
    Math.abs(actual.z - expected.z),
  );
  if (error > tolerance) {
    throw new Error(
      `${label} mismatch: got ${actual.toString()}, expected ${expected.toString()} ` +
        `(max component error ${error} > ${tolerance})`,
    );
  }
}

/** Asserts that two quaternions describe the same rotation within `angleTolerance` radians. */
export function expectSameRotation(a: Quat, b: Quat, angleTolerance = 1e-9, label = "Quat"): void {
  const angle = quatRotationAngle(a, b);
  if (angle > angleTolerance) {
    throw new Error(
      `${label} rotations differ by ${angle} rad (> ${angleTolerance}): ` +
        `${a.toString()} vs ${b.toString()}`,
    );
  }
}

/** Asserts matrix closeness with a readable failure message. */
export function expectMat3Close(a: Mat3, b: Mat3, tolerance = 1e-12, label = "Mat3"): void {
  const error = Math.max(
    Math.abs(a.m00 - b.m00),
    Math.abs(a.m01 - b.m01),
    Math.abs(a.m02 - b.m02),
    Math.abs(a.m10 - b.m10),
    Math.abs(a.m11 - b.m11),
    Math.abs(a.m12 - b.m12),
    Math.abs(a.m20 - b.m20),
    Math.abs(a.m21 - b.m21),
    Math.abs(a.m22 - b.m22),
  );
  if (error > tolerance) {
    throw new Error(`${label} mismatch: max component error ${error} > ${tolerance}`);
  }
}

/** Asserts matrix closeness with a readable failure message. */
export function expectMat4Close(a: Mat4, b: Mat4, tolerance = 1e-12, label = "Mat4"): void {
  const error = Math.max(
    Math.abs(a.m00 - b.m00),
    Math.abs(a.m01 - b.m01),
    Math.abs(a.m02 - b.m02),
    Math.abs(a.m03 - b.m03),
    Math.abs(a.m10 - b.m10),
    Math.abs(a.m11 - b.m11),
    Math.abs(a.m12 - b.m12),
    Math.abs(a.m13 - b.m13),
    Math.abs(a.m20 - b.m20),
    Math.abs(a.m21 - b.m21),
    Math.abs(a.m22 - b.m22),
    Math.abs(a.m23 - b.m23),
    Math.abs(a.m30 - b.m30),
    Math.abs(a.m31 - b.m31),
    Math.abs(a.m32 - b.m32),
    Math.abs(a.m33 - b.m33),
  );
  if (error > tolerance) {
    throw new Error(`${label} mismatch: max component error ${error} > ${tolerance}`);
  }
}
