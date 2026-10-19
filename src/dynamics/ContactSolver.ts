import { Vec3 } from "../math/Vec3.js";
import { ContactCache } from "../collision/manifold/ContactManifold.js";
import type { CachedImpulse } from "../collision/manifold/ContactManifold.js";
import type { ContactManifold, ContactPoint } from "../collision/narrowphase/Contact.js";
import { ConstraintRow } from "./ConstraintRow.js";
import type { RigidBody } from "./RigidBody.js";

/** A body pair and its narrowphase manifold, before solver setup. */
export interface ContactPair {
  bodyA: RigidBody;
  bodyB: RigidBody;
  manifold: ContactManifold;
}

/** Tuning constants of the sequential-impulse contact solver (Section 6). */
export interface ContactSolverOptions {
  /** Baumgarte stabilization factor β. Defaults to `0.2`. */
  baumgarte?: number;
  /** Penetration slop in metres. Defaults to `0.005`. */
  penetrationSlop?: number;
  /** Minimum closing speed for restitution, in m/s. Defaults to `1`. */
  restitutionThreshold?: number;
}

interface ContactPointConstraint {
  readonly id: number;
  /** World-space contact position (kept for debug overlays). */
  readonly position: Vec3;
  readonly penetration: number;
  readonly normalRow: ConstraintRow;
  readonly tangentRow1: ConstraintRow;
  readonly tangentRow2: ConstraintRow;
  readonly friction: number;
  /** Split-impulse target for Baumgarte penetration correction. */
  positionBias: number;
  /** Accumulated positional (pseudo) impulse. */
  positionImpulse: number;
}

/**
 * One persistent contact manifold expressed as friction constraints: a normal
 * row plus two orthogonal tangent rows per contact point.
 *
 * Baumgarte penetration correction is handled by a separate split-impulse pass
 * (`solvePositions`) so it never injects real kinetic energy. The velocity pass
 * only drives the relative normal velocity to the restitution target and
 * applies Coulomb friction.
 */
export class ContactConstraint {
  public readonly bodyA: RigidBody;
  public readonly bodyB: RigidBody;
  public readonly normal: Vec3;
  public readonly tangent1: Vec3;
  public readonly tangent2: Vec3;
  public readonly points: ContactPointConstraint[] = [];
  public readonly pairKey: string;

  private readonly baumgarte: number;
  private readonly penetrationSlop: number;
  private readonly restitutionThreshold: number;

  constructor(pair: ContactPair, options: Required<ContactSolverOptions>) {
    this.bodyA = pair.bodyA;
    this.bodyB = pair.bodyB;
    this.normal = pair.manifold.normal.clone();
    this.pairKey = ContactCache.pairKey(this.bodyA.id, this.bodyB.id);
    this.baumgarte = options.baumgarte;
    this.penetrationSlop = options.penetrationSlop;
    this.restitutionThreshold = options.restitutionThreshold;

    this.tangent1 = new Vec3();
    this.tangent2 = new Vec3();
    buildTangentBasis(this.normal, this.tangent1, this.tangent2);

    const friction = Math.sqrt(
      Math.max(0, this.bodyA.friction) * Math.max(0, this.bodyB.friction),
    );
    for (const point of pair.manifold.points) {
      this.points.push(this.makePoint(point, friction));
    }
  }

  private makePoint(point: ContactPoint, friction: number): ContactPointConstraint {
    const n = this.normal;
    const t1 = this.tangent1;
    const t2 = this.tangent2;
    const rA = point.position.clone().sub(this.bodyA.position);
    const rB = point.position.clone().sub(this.bodyB.position);

    const normalRow = new ConstraintRow(
      this.bodyA,
      this.bodyB,
      n.clone().negate(),
      rA.clone().cross(n).negate(),
      n.clone(),
      rB.clone().cross(n),
    );
    // Signorini condition: contacts push, never pull.
    normalRow.lowerBound = 0;
    const tangentRow1 = new ConstraintRow(
      this.bodyA,
      this.bodyB,
      t1.clone().negate(),
      rA.clone().cross(t1).negate(),
      t1.clone(),
      rB.clone().cross(t1),
    );
    const tangentRow2 = new ConstraintRow(
      this.bodyA,
      this.bodyB,
      t2.clone().negate(),
      rA.clone().cross(t2).negate(),
      t2.clone(),
      rB.clone().cross(t2),
    );

    return {
      id: point.id,
      position: point.position.clone(),
      penetration: point.penetration,
      normalRow,
      tangentRow1,
      tangentRow2,
      friction,
      positionBias: 0,
      positionImpulse: 0,
    };
  }

  /** Computes effective masses, bias and warm-start impulses. */
  public prepare(dt: number, cache: ContactCache): void {
    const restitution = Math.sqrt(
      Math.max(0, this.bodyA.restitution) * Math.max(0, this.bodyB.restitution),
    );
    for (const point of this.points) {
      const approachingVelocity = point.normalRow.relativeVelocity();
      point.normalRow.prepare();
      point.tangentRow1.prepare();
      point.tangentRow2.prepare();

      point.positionBias =
        (this.baumgarte / dt) * Math.max(0, point.penetration - this.penetrationSlop);
      point.positionImpulse = 0;

      const restitutionBias =
        approachingVelocity < -this.restitutionThreshold
          ? -restitution * approachingVelocity
          : 0;
      point.normalRow.bias = restitutionBias;

      const cached = cache.get(this.pairKey, point.id);
      if (cached !== undefined) {
        point.normalRow.impulse = Math.max(0, cached.normalImpulse);
        // Friction impulses are intentionally *not* warm-started. The tangent
        // basis is rebuilt from the current normal every substep; reusing
        // scalar impulses from a rotated basis injects lateral energy into
        // resting stacks (observed as slow drift). Friction is re-solved from
        // rest each substep instead.
        point.tangentRow1.impulse = 0;
        point.tangentRow2.impulse = 0;
      }
    }
  }

  /** Applies cached impulses before the iteration loop. */
  public warmStart(): void {
    for (const point of this.points) {
      point.normalRow.warmStart();
      point.tangentRow1.warmStart();
      point.tangentRow2.warmStart();
    }
  }

  /** Runs one Gauss-Seidel pass over normal and friction rows. */
  public solve(): void {
    for (const point of this.points) {
      point.normalRow.solve();
      const maxFriction = point.friction * point.normalRow.impulse;
      point.tangentRow1.lowerBound = -maxFriction;
      point.tangentRow1.upperBound = maxFriction;
      point.tangentRow1.solve();
      point.tangentRow2.lowerBound = -maxFriction;
      point.tangentRow2.upperBound = maxFriction;
      point.tangentRow2.solve();
    }
  }

  /**
   * Runs split-impulse position iterations. Pseudo-impulses move bodies apart
   * without changing their real velocities.
   */
  public solvePositions(iterations: number): void {
    for (let iteration = 0; iteration < iterations; iteration++) {
      for (const point of this.points) {
        const relativeVelocity = point.normalRow.relativeVelocityPseudo();
        const delta =
          -point.normalRow.effectiveMass * (relativeVelocity - point.positionBias);
        const previous = point.positionImpulse;
        const next = Math.max(0, previous + delta);
        point.positionImpulse = next;
        const applied = next - previous;
        if (applied !== 0) {
          point.normalRow.applyPseudoImpulse(applied);
        }
      }
    }
  }

  /** Stores the accumulated impulses for next-step warm starting. */
  public store(cache: ContactCache): void {
    for (const point of this.points) {
      const impulse: CachedImpulse = {
        normalImpulse: point.normalRow.impulse,
        tangentImpulse1: point.tangentRow1.impulse,
        tangentImpulse2: point.tangentRow2.impulse,
      };
      cache.store(this.pairKey, point.id, impulse);
    }
  }
}

/** Contact solver wrapper that owns the per-step constraint list. */
export class ContactSolver {
  public readonly constraints: ContactConstraint[] = [];
  private readonly options: Required<ContactSolverOptions>;

  constructor(options: ContactSolverOptions = {}) {
    this.options = {
      baumgarte: options.baumgarte ?? 0.2,
      penetrationSlop: options.penetrationSlop ?? 0.005,
      restitutionThreshold: options.restitutionThreshold ?? 1,
    };
  }

  /** Builds and prepares constraints for every manifold pair. */
  public prepare(pairs: readonly ContactPair[], dt: number, cache: ContactCache): void {
    this.constraints.length = 0;
    for (const pair of pairs) {
      this.constraints.push(new ContactConstraint(pair, this.options));
    }
    for (const constraint of this.constraints) {
      constraint.prepare(dt, cache);
    }
  }

  /** Applies warm-start impulses for all constraints. */
  public warmStart(): void {
    for (const constraint of this.constraints) {
      constraint.warmStart();
    }
  }

  /** Runs `iterations` sequential-impulse passes. */
  public solve(iterations: number): void {
    for (let iteration = 0; iteration < iterations; iteration++) {
      for (const constraint of this.constraints) {
        constraint.solve();
      }
    }
  }

  /** Runs the split-impulse position pass for all constraints. */
  public solvePositions(iterations: number): void {
    for (const constraint of this.constraints) {
      constraint.solvePositions(iterations);
    }
  }

  /** Stores impulses for next-step warm starting. */
  public store(cache: ContactCache): void {
    for (const constraint of this.constraints) {
      constraint.store(cache);
    }
  }
}

/**
 * Builds an orthonormal tangent basis `(t1, t2)` perpendicular to `normal`.
 * Picking the least-aligned world axis keeps the cross product well conditioned.
 */
function buildTangentBasis(normal: Vec3, tangent1: Vec3, tangent2: Vec3): void {
  if (Math.abs(normal.x) > 0.9) {
    tangent1.set(0, 1, 0).cross(normal);
  } else {
    tangent1.set(1, 0, 0).cross(normal);
  }
  tangent1.normalize();
  tangent2.copy(normal).cross(tangent1).normalize();
}
