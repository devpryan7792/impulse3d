/**
 * Shared numerical constants and scalar helpers used across the math core.
 *
 * All tolerances are expressed for double-precision (`number`) arithmetic.
 * The engine intentionally avoids `Number.EPSILON`-scale thresholds in
 * geometric predicates: they are dominated by the accumulated error of
 * matrix/quaternion composition chains.
 */

/** Absolute tolerance for geometric degeneracy tests. */
export const EPSILON = 1e-12;

/** Squared equivalent of {@link EPSILON}, for lengthSq() comparisons. */
export const EPSILON_SQ = 1e-24;

/** Clamps `value` into the inclusive range `[min, max]`. */
export function clamp(value: number, min: number, max: number): number {
  if (value < min) {
    return min;
  }
  if (value > max) {
    return max;
  }
  return value;
}

/** Returns true when `|a - b| <= tolerance`. */
export function approximatelyEqual(a: number, b: number, tolerance: number = EPSILON): boolean {
  return Math.abs(a - b) <= tolerance;
}
