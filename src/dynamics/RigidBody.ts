import { Mat3 } from "../math/Mat3.js";
import { Quat } from "../math/Quat.js";
import { Transform } from "../math/Transform.js";
import { Vec3 } from "../math/Vec3.js";
import { AABB } from "../collision/AABB.js";
import { Box } from "../collision/shapes/Box.js";
import { Capsule } from "../collision/shapes/Capsule.js";
import { ConvexPolytope } from "../collision/shapes/ConvexPolytope.js";
import { Shape, ShapeType } from "../collision/shapes/Shape.js";
import { Sphere } from "../collision/shapes/Sphere.js";

/** Dynamic state discriminant for a rigid body. */
export const BodyType = {
  /** Infinite mass, never integrated, never sleeps. */
  Static: 0,
  /** Infinite mass with a prescribed velocity (moving platforms, motors). */
  Kinematic: 1,
  /** Finite mass, affected by forces, contacts and joints. */
  Dynamic: 2,
} as const;

export type BodyType = (typeof BodyType)[keyof typeof BodyType];

/** Constructor options for {@link RigidBody}. */
export interface RigidBodyOptions {
  /** Collision geometry. */
  shape: Shape;
  /** Defaults to `Dynamic` when `mass > 0`, otherwise `Static`. */
  type?: BodyType;
  /** Mass in kilograms. Defaults to `1` for bounded dynamic shapes. */
  mass?: number;
  position?: Vec3;
  rotation?: Quat;
  linearVelocity?: Vec3;
  angularVelocity?: Vec3;
  /** Coulomb friction coefficient (default `0.5`). */
  friction?: number;
  /** Restitution coefficient in `[0, 1]` (default `0`). */
  restitution?: number;
  /** Velocity damping rate in 1/s (default `0`, disabled). */
  linearDamping?: number;
  /** Angular damping rate in 1/s (default `0`, disabled). */
  angularDamping?: number;
  /** Arbitrary renderer/application payload. */
  userData?: unknown;
}

/** Scratch world inertia matrix for {@link RigidBody.updateWorldInertia}. */
const scratchRotation = new Mat3();
const scratchWorldInertia = new Mat3();
const scratchWorldInverse = new Mat3();
const scratchVector = new Vec3();
const scratchAngularMomentum = new Vec3();
const scratchGyro = new Vec3();
const scratchTorque = new Vec3();
const scratchAngularAcceleration = new Vec3();
const scratchTotalAngular = new Vec3();

let nextBodyId = 1;

/**
 * Returns the body-space inertia tensor of `shape` for a given mass.
 *
 * Sphere, box and capsule use the closed forms from Section 2.3. A convex
 * polytope has no face list in this engine, so its inertia is approximated by
 * the box spanned by its local AABB (documented approximation).
 *
 * @throws RangeError when a dynamic body has a plane or zero-volume shape.
 */
export function computeLocalInertia(shape: Shape, mass: number): Mat3 {
  switch (shape.type) {
    case ShapeType.Sphere:
      return (shape as Sphere).computeLocalInertia(mass);
    case ShapeType.Box:
      return (shape as Box).computeLocalInertia(mass);
    case ShapeType.Capsule:
      return (shape as Capsule).computeLocalInertia(mass);
    case ShapeType.ConvexPolytope: {
      const polytope = shape as ConvexPolytope;
      const min = new Vec3(Infinity, Infinity, Infinity);
      const max = new Vec3(-Infinity, -Infinity, -Infinity);
      for (const vertex of polytope.vertices) {
        min.x = Math.min(min.x, vertex.x);
        min.y = Math.min(min.y, vertex.y);
        min.z = Math.min(min.z, vertex.z);
        max.x = Math.max(max.x, vertex.x);
        max.y = Math.max(max.y, vertex.y);
        max.z = Math.max(max.z, vertex.z);
      }
      const halfExtents = max.clone().sub(min).scale(0.5);
      if (halfExtents.x <= 0 || halfExtents.y <= 0 || halfExtents.z <= 0) {
        throw new RangeError("computeLocalInertia: convex polytope has zero volume");
      }
      return new Box(halfExtents).computeLocalInertia(mass);
    }
    default:
      throw new RangeError("computeLocalInertia: a plane cannot be a dynamic body");
  }
}

/**
 * A rigid body with a 13-dimensional state (Section 2.1).
 *
 * The canonical state lives in {@link position} and {@link rotation};
 * {@link transform} is a `Transform` view over those same objects so the
 * narrowphase can consume it without copying. Velocities and forces are
 * world-space vectors. Body-space inertia is fixed at construction and rotated
 * into world space by {@link updateWorldInertia}.
 */
export class RigidBody {
  /** Unique, monotonically increasing body id. */
  public readonly id: number;
  public readonly shape: Shape;
  public readonly type: BodyType;

  /** Center-of-mass position (canonical state). */
  public readonly position: Vec3;
  /** Orientation (canonical state, always normalized). */
  public readonly rotation: Quat;
  /** View over {@link position} and {@link rotation}. */
  public readonly transform: Transform;

  public readonly linearVelocity: Vec3;
  public readonly angularVelocity: Vec3;

  /**
   * Pseudo-velocities used by the split-impulse position solver. They affect
   * position integration only and are cleared after every step, so Baumgarte
   * penetration correction never injects real kinetic energy.
   */
  public readonly pseudoLinearVelocity = new Vec3();
  public readonly pseudoAngularVelocity = new Vec3();

  /** Accumulated force for the current step. */
  public readonly force = new Vec3();
  /** Accumulated torque for the current step. */
  public readonly torque = new Vec3();

  /** Mass in kilograms; `0` for static and kinematic bodies. */
  public readonly mass: number;
  /** `1 / mass`, or `0` for infinite-mass bodies. */
  public readonly inverseMass: number;

  /** Body-space inertia tensor. Zero for infinite-mass bodies. */
  public readonly localInertia: Mat3;
  /** Body-space inverse inertia tensor. Zero for infinite-mass bodies. */
  public readonly localInverseInertia: Mat3;
  /** World-space inertia tensor (`R I_body Rᵀ`). */
  public readonly worldInertia = new Mat3();
  /** World-space inverse inertia tensor (`R I_body⁻¹ Rᵀ`). */
  public readonly worldInverseInertia = new Mat3();

  public friction: number;
  public restitution: number;
  /** Velocity damping rate in 1/s; `0` disables damping. */
  public linearDamping: number;
  /** Angular damping rate in 1/s; `0` disables damping. */
  public angularDamping: number;
  public userData: unknown;

  /** Seconds spent continuously below the island sleep energy threshold. */
  public sleepTime = 0;

  private awakeState: boolean;

  constructor(options: RigidBodyOptions) {
    if (options === null || typeof options !== "object") {
      throw new RangeError("RigidBody: options object is required");
    }
    const shape = options.shape;
    if (shape === undefined || shape === null) {
      throw new RangeError("RigidBody: shape is required");
    }
    const requestedMass = options.mass === undefined ? 1 : options.mass;
    if (!Number.isFinite(requestedMass) || requestedMass < 0) {
      throw new RangeError(`RigidBody: mass must be finite and non-negative, got ${requestedMass}`);
    }
    const type =
      options.type ??
      (requestedMass > 0 ? BodyType.Dynamic : BodyType.Static);
    if (type !== BodyType.Static && type !== BodyType.Kinematic && type !== BodyType.Dynamic) {
      throw new RangeError(`RigidBody: unknown body type ${String(type)}`);
    }
    if (type === BodyType.Dynamic && requestedMass <= 0) {
      throw new RangeError("RigidBody: a dynamic body requires a positive mass");
    }
    if (type === BodyType.Dynamic && shape.type === ShapeType.Plane) {
      throw new RangeError("RigidBody: a plane cannot be a dynamic body");
    }
    if (options.friction !== undefined && (!Number.isFinite(options.friction) || options.friction < 0)) {
      throw new RangeError("RigidBody: friction must be finite and non-negative");
    }
    if (
      options.restitution !== undefined &&
      (!Number.isFinite(options.restitution) || options.restitution < 0)
    ) {
      throw new RangeError("RigidBody: restitution must be finite and non-negative");
    }
    if (
      options.linearDamping !== undefined &&
      (!Number.isFinite(options.linearDamping) || options.linearDamping < 0)
    ) {
      throw new RangeError("RigidBody: linearDamping must be finite and non-negative");
    }
    if (
      options.angularDamping !== undefined &&
      (!Number.isFinite(options.angularDamping) || options.angularDamping < 0)
    ) {
      throw new RangeError("RigidBody: angularDamping must be finite and non-negative");
    }

    this.id = nextBodyId++;
    this.shape = shape;
    this.type = type;
    this.position = options.position === undefined ? new Vec3() : options.position.clone();
    this.rotation = options.rotation === undefined ? new Quat() : options.rotation.clone();
    this.rotation.normalize();
    this.transform = new Transform(this.position, this.rotation);
    this.linearVelocity =
      options.linearVelocity === undefined ? new Vec3() : options.linearVelocity.clone();
    this.angularVelocity =
      options.angularVelocity === undefined ? new Vec3() : options.angularVelocity.clone();
    this.friction = options.friction ?? 0.5;
    this.restitution = options.restitution ?? 0;
    this.linearDamping = options.linearDamping ?? 0;
    this.angularDamping = options.angularDamping ?? 0;
    this.userData = options.userData ?? null;

    if (type === BodyType.Dynamic) {
      this.mass = requestedMass;
      this.inverseMass = 1 / requestedMass;
      this.localInertia = computeLocalInertia(shape, requestedMass);
      this.localInverseInertia = new Mat3();
      if (!Mat3.invert(this.localInverseInertia, this.localInertia)) {
        throw new RangeError("RigidBody: shape has a singular inertia tensor");
      }
      this.awakeState = true;
    } else {
      // Infinite mass: both inertia tensors are zero so Jacobian products are
      // no-ops for static and kinematic bodies.
      this.mass = 0;
      this.inverseMass = 0;
      this.localInertia = Mat3.zero(new Mat3());
      this.localInverseInertia = Mat3.zero(new Mat3());
      this.awakeState = type === BodyType.Kinematic;
    }
    this.updateWorldInertia();
  }

  /** `true` for dynamic bodies that are not sleeping; kinematic bodies are always active. */
  public get isAwake(): boolean {
    return this.awakeState;
  }

  /** `true` for dynamic bodies that are currently sleeping. */
  public get isSleeping(): boolean {
    return this.type === BodyType.Dynamic && !this.awakeState;
  }

  /** `true` for {@link BodyType.Dynamic} bodies. */
  public get isDynamic(): boolean {
    return this.type === BodyType.Dynamic;
  }

  /** `true` for {@link BodyType.Kinematic} bodies. */
  public get isKinematic(): boolean {
    return this.type === BodyType.Kinematic;
  }

  /**
   * Rotates the body-space inertia tensors into world space:
   * `I_world = R I_body Rᵀ`, `I_world⁻¹ = R I_body⁻¹ Rᵀ`.
   */
  public updateWorldInertia(): void {
    if (this.type !== BodyType.Dynamic) {
      Mat3.zero(this.worldInertia);
      Mat3.zero(this.worldInverseInertia);
      return;
    }
    Quat.toMat3(scratchRotation, this.rotation);
    Mat3.transformInertia(scratchWorldInertia, scratchRotation, this.localInertia);
    Mat3.transformInertia(scratchWorldInverse, scratchRotation, this.localInverseInertia);
    this.worldInertia.copy(scratchWorldInertia);
    this.worldInverseInertia.copy(scratchWorldInverse);
  }

  /** Tight world-space AABB of the body's shape. */
  public computeAABB(out: AABB = new AABB()): AABB {
    return this.shape.computeAABB(this.transform, out);
  }

  /** Wakes a sleeping dynamic body and resets its sleep timer. */
  public wake(): void {
    if (this.type !== BodyType.Dynamic || this.awakeState) {
      return;
    }
    this.awakeState = true;
    this.sleepTime = 0;
  }

  /** Puts a dynamic body to sleep and zeroes its velocities. */
  public sleep(): void {
    if (this.type !== BodyType.Dynamic) {
      return;
    }
    this.awakeState = false;
    this.sleepTime = 0;
    this.linearVelocity.setZero();
    this.angularVelocity.setZero();
    this.pseudoLinearVelocity.setZero();
    this.pseudoAngularVelocity.setZero();
  }

  /**
   * Adds a force at an optional world point. Off-center forces also add the
   * corresponding torque. Applying force wakes a sleeping dynamic body.
   */
  public applyForce(force: Vec3, point?: Vec3): void {
    if (this.type !== BodyType.Dynamic) {
      return;
    }
    this.wake();
    this.force.add(force);
    if (point !== undefined) {
      const r = point.clone().sub(this.position);
      this.torque.add(r.cross(force));
    }
  }

  /** Adds a pure torque for the current step. */
  public applyTorque(torque: Vec3): void {
    if (this.type !== BodyType.Dynamic) {
      return;
    }
    this.wake();
    this.torque.add(torque);
  }

  /**
   * Applies an instantaneous world-space impulse at an optional world point.
   * Off-center impulses also change angular velocity.
   */
  public applyImpulse(impulse: Vec3, point?: Vec3): void {
    if (this.type !== BodyType.Dynamic) {
      return;
    }
    this.wake();
    this.linearVelocity.addScaled(impulse, this.inverseMass);
    if (point !== undefined) {
      const r = point.clone().sub(this.position);
      const angularImpulse = r.cross(impulse);
      Mat3.transformVector(scratchVector, this.worldInverseInertia, angularImpulse);
      this.angularVelocity.add(scratchVector);
    }
  }

  /** Applies an instantaneous angular impulse. */
  public applyAngularImpulse(impulse: Vec3): void {
    if (this.type !== BodyType.Dynamic) {
      return;
    }
    this.wake();
    Mat3.transformVector(scratchVector, this.worldInverseInertia, impulse);
    this.angularVelocity.add(scratchVector);
  }

  /** World-space velocity of the point currently at `point`. */
  public getVelocityAtPoint(point: Vec3, out: Vec3 = new Vec3()): Vec3 {
    const r = point.clone().sub(this.position);
    return out.copy(this.angularVelocity).cross(r).add(this.linearVelocity);
  }

  /**
   * Symplectic force integration (Section 2.2): gravity and accumulated forces
   * update linear velocity, while torque and the gyroscopic term update angular
   * velocity. Kinematic and sleeping bodies are untouched.
   */
  public integrateForces(dt: number, gravity: Vec3, forceScale = 1): void {
    if (this.type !== BodyType.Dynamic || !this.awakeState) {
      return;
    }
    this.updateWorldInertia();
    this.linearVelocity.addScaled(gravity, dt);
    this.linearVelocity.addScaled(this.force, this.inverseMass * dt * forceScale);

    Mat3.transformVector(scratchAngularMomentum, this.worldInertia, this.angularVelocity);
    scratchGyro.copy(this.angularVelocity).cross(scratchAngularMomentum);
    scratchTorque.copy(this.torque).scale(forceScale).sub(scratchGyro);
    Mat3.transformVector(
      scratchAngularAcceleration,
      this.worldInverseInertia,
      scratchTorque,
    );
    this.angularVelocity.addScaled(scratchAngularAcceleration, dt);

    // Exponential-ish damping: v *= 1 / (1 + c·dt). Per-substep application
    // composes to approximately exp(-c·dt_total) over a public step.
    if (this.linearDamping > 0) {
      this.linearVelocity.scale(1 / (1 + this.linearDamping * dt));
    }
    if (this.angularDamping > 0) {
      this.angularVelocity.scale(1 / (1 + this.angularDamping * dt));
    }
  }

  /**
   * Symplectic position integration. Dynamic awake bodies and kinematic bodies
   * move; static and sleeping bodies do not.
   */
  public integrateVelocity(dt: number): void {
    const moves = this.type === BodyType.Kinematic || (this.type === BodyType.Dynamic && this.awakeState);
    if (!moves) {
      return;
    }
    this.position
      .addScaled(this.linearVelocity, dt)
      .addScaled(this.pseudoLinearVelocity, dt);
    scratchTotalAngular.copy(this.angularVelocity).add(this.pseudoAngularVelocity);
    this.rotation.integrateAngularVelocity(scratchTotalAngular, dt);
    this.pseudoLinearVelocity.setZero();
    this.pseudoAngularVelocity.setZero();
    this.updateWorldInertia();
  }

  /** Kinetic energy `½m|v|² + ½ω·I_world ω` in joules. */
  public kineticEnergy(): number {
    if (this.type !== BodyType.Dynamic) {
      return 0;
    }
    this.updateWorldInertia();
    Mat3.transformVector(scratchAngularMomentum, this.worldInertia, this.angularVelocity);
    return (
      0.5 * this.mass * this.linearVelocity.lengthSq() +
      0.5 * this.angularVelocity.dot(scratchAngularMomentum)
    );
  }

  /** Clears per-step force accumulators. */
  public clearForces(): void {
    this.force.setZero();
    this.torque.setZero();
  }
}
