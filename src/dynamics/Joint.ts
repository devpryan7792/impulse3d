import { Vec3 } from "../math/Vec3.js";
import { BodyType } from "./RigidBody.js";
import type { RigidBody } from "./RigidBody.js";
import { ConstraintRow } from "./ConstraintRow.js";

/** Shared constructor options for anchored joints. */
export interface JointOptions {
  /** Anchor in body A's local frame (default: centre of mass). */
  localAnchorA?: Vec3;
  /** Anchor in body B's local frame (default: centre of mass). */
  localAnchorB?: Vec3;
}

const axisX = new Vec3(1, 0, 0);
const axisY = new Vec3(0, 1, 0);
const axisZ = new Vec3(0, 0, 1);

/**
 * Base class for bilateral velocity constraints between two bodies. A joint
 * owns a fixed set of {@link ConstraintRow}s; `prepare` refreshes the Jacobian
 * blocks, effective masses and positional bias each step, while the accumulated
 * impulses provide warm starting across steps.
 */
export abstract class Joint {
  public readonly bodyA: RigidBody;
  public readonly bodyB: RigidBody;
  public readonly rows: ConstraintRow[] = [];
  public enabled = true;
  /**
   * When `false` (default) the world skips contact generation between the two
   * connected bodies, so overlapping joint hardware never fights the
   * constraint. Set to `true` to opt back in to collisions.
   */
  public collideConnected = false;

  constructor(bodyA: RigidBody, bodyB: RigidBody) {
    if (bodyA === bodyB) {
      throw new RangeError("Joint: a body cannot be joined to itself");
    }
    if (bodyA.type !== BodyType.Dynamic && bodyB.type !== BodyType.Dynamic) {
      throw new RangeError("Joint: at least one body must be dynamic");
    }
    this.bodyA = bodyA;
    this.bodyB = bodyB;
  }

  /** Recomputes Jacobians, effective masses and bias for the current state. */
  public abstract prepare(dt: number, baumgarte: number): void;

  /** Applies cached impulses before the iteration loop. */
  public warmStart(): void {
    for (const row of this.rows) {
      row.warmStart();
    }
  }

  /** Runs one Gauss-Seidel pass over the joint rows. */
  public solve(): void {
    for (const row of this.rows) {
      row.solve();
    }
  }

  /** `true` when the constraint has an awake dynamic or kinematic side. */
  public isActive(): boolean {
    return (
      (this.bodyA.isDynamic && this.bodyA.isAwake) ||
      (this.bodyB.isDynamic && this.bodyB.isAwake) ||
      this.bodyA.isKinematic ||
      this.bodyB.isKinematic
    );
  }
}

/** Ball-and-socket (spherical) joint; constrains the three linear DOFs. */
export class BallSocketJoint extends Joint {
  public readonly localAnchorA: Vec3;
  public readonly localAnchorB: Vec3;

  private readonly worldAnchorA = new Vec3();
  private readonly worldAnchorB = new Vec3();
  private readonly rA = new Vec3();
  private readonly rB = new Vec3();
  private readonly error = new Vec3();

  constructor(bodyA: RigidBody, bodyB: RigidBody, options: JointOptions = {}) {
    super(bodyA, bodyB);
    this.localAnchorA = options.localAnchorA === undefined ? new Vec3() : options.localAnchorA.clone();
    this.localAnchorB = options.localAnchorB === undefined ? new Vec3() : options.localAnchorB.clone();
    for (let i = 0; i < 3; i++) {
      this.rows.push(
        new ConstraintRow(bodyA, bodyB, new Vec3(), new Vec3(), new Vec3(), new Vec3()),
      );
    }
  }

  public prepare(dt: number, baumgarte: number): void {
    this.bodyA.transform.transformPoint(this.worldAnchorA, this.localAnchorA);
    this.bodyB.transform.transformPoint(this.worldAnchorB, this.localAnchorB);
    this.error.copy(this.worldAnchorB).sub(this.worldAnchorA);
    this.rA.copy(this.worldAnchorA).sub(this.bodyA.position);
    this.rB.copy(this.worldAnchorB).sub(this.bodyB.position);

    const axes = [axisX, axisY, axisZ];
    for (let i = 0; i < 3; i++) {
      const axis = axes[i];
      const row = this.rows[i];
      row.linearA.copy(axis).negate();
      row.angularA.copy(this.rA).cross(axis).negate();
      row.linearB.copy(axis);
      row.angularB.copy(this.rB).cross(axis);
      row.bias = -(baumgarte / dt) * this.error.dot(axis);
      row.prepare();
    }
  }
}

/** Distance (rigid rod) joint; constrains the scalar distance between anchors. */
export class DistanceJoint extends Joint {
  public readonly localAnchorA: Vec3;
  public readonly localAnchorB: Vec3;
  public readonly restLength: number;

  private readonly worldAnchorA = new Vec3();
  private readonly worldAnchorB = new Vec3();
  private readonly rA = new Vec3();
  private readonly rB = new Vec3();
  private readonly axis = new Vec3();

  constructor(
    bodyA: RigidBody,
    bodyB: RigidBody,
    options: JointOptions & { restLength?: number } = {},
  ) {
    super(bodyA, bodyB);
    this.localAnchorA = options.localAnchorA === undefined ? new Vec3() : options.localAnchorA.clone();
    this.localAnchorB = options.localAnchorB === undefined ? new Vec3() : options.localAnchorB.clone();
    this.bodyA.transform.transformPoint(this.worldAnchorA, this.localAnchorA);
    this.bodyB.transform.transformPoint(this.worldAnchorB, this.localAnchorB);
    const measured = this.worldAnchorA.distanceTo(this.worldAnchorB);
    this.restLength = options.restLength ?? measured;
    if (!Number.isFinite(this.restLength) || this.restLength < 0) {
      throw new RangeError("DistanceJoint: restLength must be finite and non-negative");
    }
    this.rows.push(
      new ConstraintRow(bodyA, bodyB, new Vec3(), new Vec3(), new Vec3(), new Vec3()),
    );
  }

  public prepare(dt: number, baumgarte: number): void {
    this.bodyA.transform.transformPoint(this.worldAnchorA, this.localAnchorA);
    this.bodyB.transform.transformPoint(this.worldAnchorB, this.localAnchorB);
    const delta = this.worldAnchorB.clone().sub(this.worldAnchorA);
    const distance = delta.length();
    if (distance > 1e-12) {
      this.axis.copy(delta).scale(1 / distance);
    } else {
      this.axis.set(1, 0, 0);
    }
    this.rA.copy(this.worldAnchorA).sub(this.bodyA.position);
    this.rB.copy(this.worldAnchorB).sub(this.bodyB.position);

    const row = this.rows[0];
    row.linearA.copy(this.axis).negate();
    row.angularA.copy(this.rA).cross(this.axis).negate();
    row.linearB.copy(this.axis);
    row.angularB.copy(this.rB).cross(this.axis);
    row.bias = -(baumgarte / dt) * (distance - this.restLength);
    row.prepare();
  }
}

/** Options for {@link HingeJoint}. */
export interface HingeJointOptions extends JointOptions {
  /** Hinge axis in body A's local frame (default `+X`). */
  localAxisA?: Vec3;
  /** Hinge axis in body B's local frame (default `+X`). */
  localAxisB?: Vec3;
}

/**
 * Revolute (hinge) joint: a ball-and-socket anchor plus two angular rows that
 * remove the two rotational DOFs perpendicular to the hinge axis, leaving one
 * free angular DOF. The axes are kept parallel with a Baumgarte bias.
 */
export class HingeJoint extends BallSocketJoint {
  public readonly localAxisA: Vec3;
  public readonly localAxisB: Vec3;

  private readonly worldAxisA = new Vec3();
  private readonly worldAxisB = new Vec3();
  private readonly perpendicular1 = new Vec3();
  private readonly perpendicular2 = new Vec3();
  private readonly axisError = new Vec3();

  constructor(bodyA: RigidBody, bodyB: RigidBody, options: HingeJointOptions = {}) {
    super(bodyA, bodyB, options);
    this.localAxisA = options.localAxisA === undefined ? new Vec3(1, 0, 0) : options.localAxisA.clone();
    this.localAxisB = options.localAxisB === undefined ? new Vec3(1, 0, 0) : options.localAxisB.clone();
    if (this.localAxisA.lengthSq() < 1e-12 || this.localAxisB.lengthSq() < 1e-12) {
      throw new RangeError("HingeJoint: hinge axis must be non-zero");
    }
    this.localAxisA.normalize();
    this.localAxisB.normalize();
    for (let i = 0; i < 2; i++) {
      this.rows.push(
        new ConstraintRow(bodyA, bodyB, new Vec3(), new Vec3(), new Vec3(), new Vec3()),
      );
    }
  }

  public override prepare(dt: number, baumgarte: number): void {
    super.prepare(dt, baumgarte);
    this.bodyA.transform.transformDirection(this.worldAxisA, this.localAxisA).normalize();
    this.bodyB.transform.transformDirection(this.worldAxisB, this.localAxisB).normalize();
    this.axisError.copy(this.worldAxisA).cross(this.worldAxisB);

    // Build a stable orthonormal frame perpendicular to the hinge axis.
    const axis = this.worldAxisA;
    const reference =
      Math.abs(axis.x) < Math.abs(axis.y)
        ? Math.abs(axis.x) < Math.abs(axis.z)
          ? axisX
          : axisZ
        : Math.abs(axis.y) < Math.abs(axis.z)
          ? axisY
          : axisZ;
    this.perpendicular1.copy(axis).cross(reference).normalize();
    this.perpendicular2.copy(axis).cross(this.perpendicular1).normalize();

    const perpendiculars = [this.perpendicular1, this.perpendicular2];
    for (let i = 0; i < 2; i++) {
      const direction = perpendiculars[i];
      const row = this.rows[3 + i];
      row.linearA.setZero();
      row.angularA.copy(direction).negate();
      row.linearB.setZero();
      row.angularB.copy(direction);
      row.bias = -(baumgarte / dt) * this.axisError.dot(direction);
      row.prepare();
    }
  }
}
