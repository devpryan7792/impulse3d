import { AABB } from "../collision/AABB.js";
import { AABBTree } from "../collision/broadphase/AABBTree.js";
import { collideShapes } from "../collision/narrowphase/Narrowphase.js";
import { ContactCache } from "../collision/manifold/ContactManifold.js";
import { Vec3 } from "../math/Vec3.js";
import { ContactSolver } from "./ContactSolver.js";
import type { ContactPair } from "./ContactSolver.js";
import type { Joint } from "./Joint.js";
import type { RigidBody } from "./RigidBody.js";

/** Tuning options for {@link World}. */
export interface WorldOptions {
  /** Gravity acceleration (default `(0, -9.81, 0)`). */
  gravity?: Vec3;
  /** Sequential-impulse iterations per substep (default `10`). */
  solverIterations?: number;
  /**
   * Internal substeps per `step` call (default `8`). Substepping keeps tall
   * stacks stable at 60 Hz without raising the external API rate; lower it to
   * trade stability for throughput.
   */
  substeps?: number;
  /** Split-impulse position-correction iterations per substep (default `3`). */
  positionIterations?: number;
  /** Baumgarte factor β (default `0.2`). */
  baumgarte?: number;
  /** Penetration slop in metres (default `0.005`). */
  penetrationSlop?: number;
  /** Minimum closing speed for restitution (default `1` m/s). */
  restitutionThreshold?: number;
  /** Island kinetic-energy threshold in joules (default `1e-4`). */
  sleepEnergyThreshold?: number;
  /** Continuous low-energy duration before sleeping, in seconds (default `0.5`). */
  sleepTime?: number;
  /** Static broadphase margin β (default `0.05` m). */
  broadphaseMargin?: number;
  /** Velocity fattening scale α (default `1`). */
  broadphaseVelocityScale?: number;
}

/** Diagnostics snapshot returned by {@link World.getStats}. */
export interface WorldStats {
  stepCount: number;
  bodyCount: number;
  awakeBodies: number;
  sleepingBodies: number;
  /** Leaf pairs reported by the broadphase during the last substep. */
  broadphasePairs: number;
  /** Pairs that produced a narrowphase manifold during the last substep. */
  manifoldPairs: number;
  /** Total manifold contact points solved during the last substep. */
  contactPoints: number;
  jointCount: number;
}

const scratchAABB = new AABB();
const scratchFatAABB = new AABB();

function isActiveBody(body: RigidBody): boolean {
  return (body.isDynamic && body.isAwake) || body.isKinematic;
}

/**
 * Rigid-body world: broadphase, narrowphase, sequential-impulse contact and
 * joint solving, split-impulse position correction, islands and sleeping
 * (Sections 6–8).
 *
 * A public step is divided into `substeps` smaller steps. Each substep
 * integrates forces, refreshes the broadphase, rebuilds manifolds, warm-starts
 * and solves contacts/joints, corrects penetration, and integrates positions.
 * User forces are distributed across the substeps so their total impulse over
 * the external step is unchanged.
 */
export class World {
  public readonly gravity: Vec3;
  public readonly broadphase: AABBTree<RigidBody>;
  public readonly contactCache = new ContactCache();
  public readonly bodies: RigidBody[] = [];
  public readonly joints: Joint[] = [];
  public readonly contactSolver: ContactSolver;
  public readonly substeps: number;
  public solverIterations: number;
  public positionIterations: number;

  private readonly baumgarte: number;
  private readonly sleepEnergyThreshold: number;
  private readonly sleepTime: number;
  private readonly broadphaseMargin: number;
  private readonly broadphaseVelocityScale: number;
  private readonly bodySet = new Set<RigidBody>();

  private stepCount = 0;
  private lastBroadphasePairs = 0;
  private lastManifoldPairs = 0;
  private lastContactPoints = 0;

  constructor(options: WorldOptions = {}) {
    const gravity =
      options.gravity === undefined ? new Vec3(0, -9.81, 0) : options.gravity.clone();
    if (
      !Number.isFinite(gravity.x) ||
      !Number.isFinite(gravity.y) ||
      !Number.isFinite(gravity.z)
    ) {
      throw new RangeError("World: gravity must be finite");
    }
    const solverIterations = options.solverIterations ?? 10;
    if (!Number.isInteger(solverIterations) || solverIterations < 1) {
      throw new RangeError("World: solverIterations must be an integer >= 1");
    }
    this.substeps = options.substeps ?? 8;
    if (!Number.isInteger(this.substeps) || this.substeps < 1) {
      throw new RangeError("World: substeps must be an integer >= 1");
    }
    const positionIterations = options.positionIterations ?? 3;
    if (!Number.isInteger(positionIterations) || positionIterations < 0) {
      throw new RangeError("World: positionIterations must be an integer >= 0");
    }
    const baumgarte = options.baumgarte ?? 0.2;
    if (!Number.isFinite(baumgarte) || baumgarte < 0) {
      throw new RangeError("World: baumgarte must be finite and non-negative");
    }
    const penetrationSlop = options.penetrationSlop ?? 0.005;
    if (!Number.isFinite(penetrationSlop) || penetrationSlop < 0) {
      throw new RangeError("World: penetrationSlop must be finite and non-negative");
    }
    const restitutionThreshold = options.restitutionThreshold ?? 1;
    if (!Number.isFinite(restitutionThreshold) || restitutionThreshold < 0) {
      throw new RangeError(
        "World: restitutionThreshold must be finite and non-negative",
      );
    }
    this.sleepEnergyThreshold = options.sleepEnergyThreshold ?? 1e-4;
    if (!Number.isFinite(this.sleepEnergyThreshold) || this.sleepEnergyThreshold < 0) {
      throw new RangeError(
        "World: sleepEnergyThreshold must be finite and non-negative",
      );
    }
    this.sleepTime = options.sleepTime ?? 0.5;
    if (!Number.isFinite(this.sleepTime) || this.sleepTime < 0) {
      throw new RangeError("World: sleepTime must be finite and non-negative");
    }
    this.broadphaseMargin = options.broadphaseMargin ?? 0.05;
    if (!Number.isFinite(this.broadphaseMargin) || this.broadphaseMargin < 0) {
      throw new RangeError("World: broadphaseMargin must be finite and non-negative");
    }
    this.broadphaseVelocityScale = options.broadphaseVelocityScale ?? 1;
    if (!Number.isFinite(this.broadphaseVelocityScale) || this.broadphaseVelocityScale < 0) {
      throw new RangeError(
        "World: broadphaseVelocityScale must be finite and non-negative",
      );
    }

    this.gravity = gravity;
    this.solverIterations = solverIterations;
    this.positionIterations = positionIterations;
    this.baumgarte = baumgarte;
    this.broadphase = new AABBTree<RigidBody>({
      margin: this.broadphaseMargin,
      velocityScale: this.broadphaseVelocityScale,
    });
    this.contactSolver = new ContactSolver({
      baumgarte,
      penetrationSlop,
      restitutionThreshold,
    });
  }

  /** Adds a body and inserts its fattened AABB into the broadphase. */
  public addBody(body: RigidBody): RigidBody {
    if (this.bodySet.has(body)) {
      throw new RangeError(`World.addBody: body ${body.id} already belongs to this world`);
    }
    body.computeAABB(scratchAABB);
    AABB.fatten(
      scratchFatAABB,
      scratchAABB,
      body.linearVelocity,
      1 / 60,
      this.broadphaseVelocityScale,
      this.broadphaseMargin,
    );
    this.broadphase.insertLeaf(body.id, scratchFatAABB, body);
    this.bodySet.add(body);
    this.bodies.push(body);
    return body;
  }

  /** Removes a body, its broadphase leaf and any joints attached to it. */
  public removeBody(body: RigidBody): boolean {
    if (!this.bodySet.has(body)) {
      return false;
    }
    this.broadphase.removeLeaf(body.id);
    this.bodySet.delete(body);
    this.bodies.splice(this.bodies.indexOf(body), 1);
    for (let i = this.joints.length - 1; i >= 0; i--) {
      if (this.joints[i].bodyA === body || this.joints[i].bodyB === body) {
        this.joints.splice(i, 1);
      }
    }
    return true;
  }

  /** Adds a joint whose bodies both belong to this world. */
  public addJoint(joint: Joint): Joint {
    if (!this.bodySet.has(joint.bodyA) || !this.bodySet.has(joint.bodyB)) {
      throw new RangeError("World.addJoint: both bodies must belong to this world");
    }
    if (this.joints.includes(joint)) {
      throw new RangeError("World.addJoint: joint already belongs to this world");
    }
    this.joints.push(joint);
    return joint;
  }

  /** Removes a joint. Returns `false` when it was not registered. */
  public removeJoint(joint: Joint): boolean {
    const index = this.joints.indexOf(joint);
    if (index < 0) {
      return false;
    }
    this.joints.splice(index, 1);
    return true;
  }

  /** Removes every body and joint and resets the broadphase and cache. */
  public clear(): void {
    this.bodies.length = 0;
    this.joints.length = 0;
    this.bodySet.clear();
    this.broadphase.clear();
    this.contactCache.clear();
    this.stepCount = 0;
    this.lastBroadphasePairs = 0;
    this.lastManifoldPairs = 0;
    this.lastContactPoints = 0;
  }

  /** Advances the simulation by `dt` seconds. */
  public step(dt: number): void {
    if (!Number.isFinite(dt) || dt <= 0) {
      throw new RangeError(`World.step: dt must be finite and positive, got ${dt}`);
    }
    this.stepCount++;
    const subDt = dt / this.substeps;
    const forceScale = 1 / this.substeps;
    for (let i = 0; i < this.substeps; i++) {
      this.substep(subDt, forceScale);
    }
    for (const body of this.bodies) {
      body.clearForces();
    }
  }

  /** Returns a diagnostics snapshot. */
  public getStats(): WorldStats {
    let awakeBodies = 0;
    let sleepingBodies = 0;
    for (const body of this.bodies) {
      if (body.isDynamic && body.isAwake) {
        awakeBodies++;
      } else if (body.isSleeping) {
        sleepingBodies++;
      }
    }
    return {
      stepCount: this.stepCount,
      bodyCount: this.bodies.length,
      awakeBodies,
      sleepingBodies,
      broadphasePairs: this.lastBroadphasePairs,
      manifoldPairs: this.lastManifoldPairs,
      contactPoints: this.lastContactPoints,
      jointCount: this.joints.length,
    };
  }

  /** One internal sub-step of the simulation pipeline. */
  private substep(dt: number, forceScale: number): void {
    // 1. Symplectic force integration (fractional user forces per substep).
    for (const body of this.bodies) {
      body.integrateForces(dt, this.gravity, forceScale);
    }

    // 2. Broadphase refresh and pair query.
    for (const body of this.bodies) {
      const tight = body.computeAABB(scratchAABB);
      this.broadphase.updateLeaf(body.id, tight, body.linearVelocity, dt);
    }

    const contactPairs: ContactPair[] = [];
    let broadphasePairs = 0;
    const jointPairKeys = new Set<string>();
    for (const joint of this.joints) {
      if (!joint.collideConnected) {
        jointPairKeys.add(ContactCache.pairKey(joint.bodyA.id, joint.bodyB.id));
      }
    }
    this.broadphase.queryOverlaps((leafA, leafB) => {
      broadphasePairs++;
      const bodyA = leafA.payload;
      const bodyB = leafB.payload;
      if (jointPairKeys.has(ContactCache.pairKey(bodyA.id, bodyB.id))) {
        return;
      }
      if (!bodyA.isDynamic && !bodyB.isDynamic) {
        return;
      }
      if (bodyA.isSleeping && bodyB.isSleeping) {
        return;
      }
      const manifold = collideShapes(
        bodyA.shape,
        bodyA.transform,
        bodyB.shape,
        bodyB.transform,
      );
      if (manifold === null) {
        return;
      }
      // Canonicalize orientation by body id so cached normal/tangent impulses
      // stay in the same frame even when broadphase traversal order changes.
      if (bodyA.id > bodyB.id) {
        manifold.normal.negate();
        contactPairs.push({ bodyA: bodyB, bodyB: bodyA, manifold });
      } else {
        contactPairs.push({ bodyA, bodyB, manifold });
      }
      if (isActiveBody(bodyA)) {
        this.wakeIsland(bodyB);
      } else if (isActiveBody(bodyB)) {
        this.wakeIsland(bodyA);
      }
    });

    // 3. Joints can also connect an active body to a sleeping island.
    for (const joint of this.joints) {
      if (!joint.isActive()) {
        continue;
      }
      if (joint.bodyA.isSleeping) {
        this.wakeIsland(joint.bodyA);
      }
      if (joint.bodyB.isSleeping) {
        this.wakeIsland(joint.bodyB);
      }
    }

    // 4. Build contact constraints for pairs with at least one active side.
    this.contactCache.beginStep();
    const activePairs: ContactPair[] = [];
    for (const pair of contactPairs) {
      if (!isActiveBody(pair.bodyA) && !isActiveBody(pair.bodyB)) {
        continue;
      }
      activePairs.push(pair);
    }
    activePairs.sort((a, b) => {
      const ya = Math.min(a.bodyA.position.y, a.bodyB.position.y);
      const yb = Math.min(b.bodyA.position.y, b.bodyB.position.y);
      return ya - yb;
    });
    this.contactSolver.prepare(activePairs, dt, this.contactCache);

    const activeJoints: Joint[] = [];
    for (const joint of this.joints) {
      if (!joint.enabled || !joint.isActive()) {
        continue;
      }
      activeJoints.push(joint);
      joint.prepare(dt, this.baumgarte);
    }

    // 5. Warm start and sequential-impulse iterations.
    this.contactSolver.warmStart();
    for (const joint of activeJoints) {
      joint.warmStart();
    }
    for (let iteration = 0; iteration < this.solverIterations; iteration++) {
      this.contactSolver.solve(1);
      for (const joint of activeJoints) {
        joint.solve();
      }
    }

    // 6. Split-impulse position correction and warm-start storage.
    this.contactSolver.solvePositions(this.positionIterations);
    this.contactSolver.store(this.contactCache);

    // 7. Symplectic position integration (real + pseudo velocities).
    for (const body of this.bodies) {
      body.integrateVelocity(dt);
    }

    // 8. Island sleep state.
    this.updateSleeping(dt, activePairs, activeJoints);

    this.lastBroadphasePairs = broadphasePairs;
    this.lastManifoldPairs = contactPairs.length;
    this.lastContactPoints = this.contactSolver.constraints.reduce(
      (sum, constraint) => sum + constraint.points.length,
      0,
    );
  }

  /**
   * Wakes a sleeping body and every dynamic body reachable through joints.
   * Contact islands are woken pairwise by the broadphase as new manifolds
   * appear, so joints are the only persistent edges that need walking here.
   */
  private wakeIsland(body: RigidBody): void {
    if (!body.isDynamic || body.isAwake) {
      return;
    }
    const stack: RigidBody[] = [body];
    const visited = new Set<RigidBody>();
    while (stack.length > 0) {
      const current = stack.pop()!;
      if (visited.has(current)) {
        continue;
      }
      visited.add(current);
      current.wake();
      for (const joint of this.joints) {
        const other =
          joint.bodyA === current
            ? joint.bodyB
            : joint.bodyB === current
              ? joint.bodyA
              : null;
        if (other !== null && other.isDynamic && other.isSleeping) {
          stack.push(other);
        }
      }
    }
  }

  /** Builds connected components and applies the island sleep state machine. */
  private updateSleeping(
    dt: number,
    contactPairs: readonly ContactPair[],
    activeJoints: readonly Joint[],
  ): void {
    const dynamics = this.bodies.filter((body) => body.isDynamic);
    const index = new Map<RigidBody, number>();
    for (let i = 0; i < dynamics.length; i++) {
      index.set(dynamics[i], i);
    }
    const parent = dynamics.map((_, i) => i);

    const find = (start: number): number => {
      let root = start;
      while (parent[root] !== root) {
        root = parent[root];
      }
      while (parent[start] !== start) {
        const next = parent[start];
        parent[start] = root;
        start = next;
      }
      return root;
    };
    const union = (a: RigidBody, b: RigidBody): void => {
      const rootA = find(index.get(a)!);
      const rootB = find(index.get(b)!);
      if (rootA !== rootB) {
        parent[rootB] = rootA;
      }
    };

    for (const pair of contactPairs) {
      if (pair.bodyA.isDynamic && pair.bodyB.isDynamic) {
        union(pair.bodyA, pair.bodyB);
      }
    }
    for (const joint of activeJoints) {
      if (joint.bodyA.isDynamic && joint.bodyB.isDynamic) {
        union(joint.bodyA, joint.bodyB);
      }
    }

    for (const body of dynamics) {
      if (!body.isAwake) {
        continue;
      }
      if (body.kineticEnergy() < this.sleepEnergyThreshold) {
        body.sleepTime += dt;
      } else {
        body.sleepTime = 0;
      }
    }

    const islands = new Map<number, RigidBody[]>();
    for (let i = 0; i < dynamics.length; i++) {
      const root = find(i);
      const members = islands.get(root);
      if (members === undefined) {
        islands.set(root, [dynamics[i]]);
      } else {
        members.push(dynamics[i]);
      }
    }

    for (const members of islands.values()) {
      let allSleeping = true;
      let allReady = true;
      for (const body of members) {
        if (!body.isSleeping) {
          allSleeping = false;
          if (body.sleepTime < this.sleepTime) {
            allReady = false;
          }
        }
      }
      if (allSleeping) {
        continue;
      }
      if (allReady) {
        for (const body of members) {
          body.sleep();
        }
      }
    }
  }
}
