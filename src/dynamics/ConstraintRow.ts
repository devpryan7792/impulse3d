import { Mat3 } from "../math/Mat3.js";
import { clamp } from "../math/MathUtils.js";
import { Vec3 } from "../math/Vec3.js";
import type { RigidBody } from "./RigidBody.js";

const scratchAngular = new Vec3();

/**
 * One scalar velocity constraint row shared by contacts and joints.
 *
 * The row stores the four Jacobian blocks of a two-body constraint:
 *
 * ```
 * J = [ linearA  angularA  linearB  angularB ]
 * ```
 *
 * so the relative velocity is
 * `linearA·vA + angularA·ωA + linearB·vB + angularB·ωB`.
 * A positive impulse `λ` applies `linearA λ` / `angularA λ` to body A and
 * `linearB λ` / `angularB λ` to body B. The effective mass is
 * `1 / (J M⁻¹ Jᵀ)` and the accumulated impulse is clamped to
 * `[lowerBound, upperBound]`.
 */
export class ConstraintRow {
  /** Accumulated impulse from previous iterations/steps. */
  public impulse = 0;
  /** Inverse of `J M⁻¹ Jᵀ` (zero for a fully static row). */
  public effectiveMass = 0;
  /** Target relative velocity: `Cdot = bias`. */
  public bias = 0;
  /** Lower clamp for the accumulated impulse. */
  public lowerBound = -Infinity;
  /** Upper clamp for the accumulated impulse. */
  public upperBound = Infinity;

  constructor(
    public readonly bodyA: RigidBody,
    public readonly bodyB: RigidBody,
    public readonly linearA: Vec3,
    public readonly angularA: Vec3,
    public readonly linearB: Vec3,
    public readonly angularB: Vec3,
  ) {}

  /** Recomputes the effective mass from the current inertia tensors. */
  public prepare(): void {
    let denominator =
      this.bodyA.inverseMass * this.linearA.lengthSq() +
      this.bodyB.inverseMass * this.linearB.lengthSq();

    Mat3.transformVector(scratchAngular, this.bodyA.worldInverseInertia, this.angularA);
    denominator += this.angularA.dot(scratchAngular);
    Mat3.transformVector(scratchAngular, this.bodyB.worldInverseInertia, this.angularB);
    denominator += this.angularB.dot(scratchAngular);

    this.effectiveMass = denominator > 0 ? 1 / denominator : 0;
  }

  /** Current relative velocity along the constraint direction. */
  public relativeVelocity(): number {
    return (
      this.linearA.dot(this.bodyA.linearVelocity) +
      this.angularA.dot(this.bodyA.angularVelocity) +
      this.linearB.dot(this.bodyB.linearVelocity) +
      this.angularB.dot(this.bodyB.angularVelocity)
    );
  }

  /** Applies `amount` directly to both bodies (used by warm starting). */
  public applyImpulse(amount: number): void {
    this.bodyA.linearVelocity.addScaled(this.linearA, this.bodyA.inverseMass * amount);
    Mat3.transformVector(scratchAngular, this.bodyA.worldInverseInertia, this.angularA);
    this.bodyA.angularVelocity.addScaled(scratchAngular, amount);

    this.bodyB.linearVelocity.addScaled(this.linearB, this.bodyB.inverseMass * amount);
    Mat3.transformVector(scratchAngular, this.bodyB.worldInverseInertia, this.angularB);
    this.bodyB.angularVelocity.addScaled(scratchAngular, amount);
  }

  /** Relative velocity along the row using split-impulse pseudo-velocities. */
  public relativeVelocityPseudo(): number {
    return (
      this.linearA.dot(this.bodyA.pseudoLinearVelocity) +
      this.angularA.dot(this.bodyA.pseudoAngularVelocity) +
      this.linearB.dot(this.bodyB.pseudoLinearVelocity) +
      this.angularB.dot(this.bodyB.pseudoAngularVelocity)
    );
  }

  /** Applies a pseudo-impulse to the split-impulse velocity state only. */
  public applyPseudoImpulse(amount: number): void {
    this.bodyA.pseudoLinearVelocity.addScaled(
      this.linearA,
      this.bodyA.inverseMass * amount,
    );
    Mat3.transformVector(scratchAngular, this.bodyA.worldInverseInertia, this.angularA);
    this.bodyA.pseudoAngularVelocity.addScaled(scratchAngular, amount);

    this.bodyB.pseudoLinearVelocity.addScaled(
      this.linearB,
      this.bodyB.inverseMass * amount,
    );
    Mat3.transformVector(scratchAngular, this.bodyB.worldInverseInertia, this.angularB);
    this.bodyB.pseudoAngularVelocity.addScaled(scratchAngular, amount);
  }

  /** Applies the accumulated impulse (warm start). */
  public warmStart(): void {
    if (this.impulse !== 0) {
      this.applyImpulse(this.impulse);
    }
  }

  /**
   * Solves one sequential-impulse update and returns the impulse actually
   * applied (after clamping).
   */
  public solve(): number {
    const relativeVelocity = this.relativeVelocity();
    const delta = -this.effectiveMass * (relativeVelocity - this.bias);
    const previous = this.impulse;
    const next = clamp(previous + delta, this.lowerBound, this.upperBound);
    this.impulse = next;
    const applied = next - previous;
    if (applied !== 0) {
      this.applyImpulse(applied);
    }
    return applied;
  }
}
