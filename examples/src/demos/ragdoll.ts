import { BallSocketJoint, Capsule, RigidBody, Sphere, Vec3 } from "impulse3d";
import { addBox, addGround, addStaticBox, paint } from "./helpers.js";
import type { DemoContext, DemoDefinition } from "./types.js";

const CENTER_X = 7.15;
const STEP_COUNT = 6;
const STEP_HEIGHT = 0.25;
const STEP_DEPTH = 1.3;

/** Builds the staircase the ragdoll tumbles down. */
function buildStaircase(context: DemoContext): void {
  const { world } = context;
  for (let i = 0; i < STEP_COUNT; i++) {
    const height = STEP_HEIGHT * (i + 1);
    addStaticBox(
      world,
      new Vec3(STEP_DEPTH / 2, height / 2, 1.2),
      new Vec3(STEP_DEPTH / 2 + i * STEP_DEPTH, height / 2, 0),
    );
  }
}

/** Builds a jointed humanoid standing on the top step. */
function buildRagdoll(context: DemoContext): void {
  const { world } = context;
  const px = CENTER_X;

  const parts: Record<string, RigidBody> = {};
  const addPart = (name: string, body: RigidBody): void => {
    parts[name] = body;
  };

  addPart(
    "pelvis",
    addBox(world, new Vec3(0.18, 0.12, 0.1), new Vec3(px, 2.62, 0), {
      mass: 8,
      color: 0x4c566a,
      friction: 0.6,
    }),
  );
  addPart(
    "torso",
    addBox(world, new Vec3(0.22, 0.3, 0.12), new Vec3(px, 3.04, 0), {
      mass: 14,
      color: 0x5e81ac,
      friction: 0.6,
    }),
  );
  addPart(
    "head",
    world.addBody(
      paint(
        new RigidBody({
          shape: new Sphere(0.13),
          mass: 4,
          position: new Vec3(px, 3.47, 0),
          friction: 0.6,
        }),
        0xd08770,
      ),
    ),
  );

  for (const [side, sign] of [
    ["Left", -1],
    ["Right", 1],
  ] as const) {
    addPart(
      `upperArm${side}`,
      world.addBody(
        paint(
          new RigidBody({
            shape: new Capsule(0.06, 0.12),
            mass: 2,
            position: new Vec3(px + sign * 0.28, 3.14, 0),
            friction: 0.6,
          }),
          0x81a1c1,
        ),
      ),
    );
    addPart(
      `forearm${side}`,
      world.addBody(
        paint(
          new RigidBody({
            shape: new Capsule(0.05, 0.11),
            mass: 1.5,
            position: new Vec3(px + sign * 0.28, 2.8, 0),
            friction: 0.6,
          }),
          0x81a1c1,
        ),
      ),
    );
    addPart(
      `thigh${side}`,
      world.addBody(
        paint(
          new RigidBody({
            shape: new Capsule(0.08, 0.18),
            mass: 4,
            position: new Vec3(px + sign * 0.1, 2.24, 0),
            friction: 0.6,
          }),
          0x8fbcbb,
        ),
      ),
    );
    addPart(
      `shin${side}`,
      world.addBody(
        paint(
          new RigidBody({
            shape: new Capsule(0.07, 0.17),
            mass: 3,
            position: new Vec3(px + sign * 0.1, 1.74, 0),
            friction: 0.6,
          }),
          0x8fbcbb,
        ),
      ),
    );
  }

  const joint = (
    a: string,
    localA: Vec3,
    b: string,
    localB: Vec3,
  ): BallSocketJoint =>
    new BallSocketJoint(parts[a], parts[b], { localAnchorA: localA, localAnchorB: localB });

  world.addJoint(joint("pelvis", new Vec3(0, 0.12, 0), "torso", new Vec3(0, -0.3, 0)));
  world.addJoint(joint("torso", new Vec3(0, 0.3, 0), "head", new Vec3(0, -0.13, 0)));
  for (const [side, sign] of [
    ["Left", -1],
    ["Right", 1],
  ] as const) {
    world.addJoint(
      joint("torso", new Vec3(sign * 0.22, 0.28, 0), `upperArm${side}`, new Vec3(0, 0.18, 0)),
    );
    world.addJoint(
      joint(`upperArm${side}`, new Vec3(0, -0.18, 0), `forearm${side}`, new Vec3(0, 0.16, 0)),
    );
    world.addJoint(
      joint("pelvis", new Vec3(sign * 0.1, -0.12, 0), `thigh${side}`, new Vec3(0, 0.26, 0)),
    );
    world.addJoint(
      joint(`thigh${side}`, new Vec3(0, -0.26, 0), `shin${side}`, new Vec3(0, 0.24, 0)),
    );
  }

  // A shove at the hips starts the tumble towards the lower steps.
  parts.torso.linearVelocity.set(-1.6, 0, 0);
  parts.torso.angularVelocity.set(0, 0, 1.4);
}

function build(context: DemoContext): void {
  addGround(context.world, 0.7);
  buildStaircase(context);
  buildRagdoll(context);
}

export const ragdollDemo: DemoDefinition = {
  id: "ragdoll",
  name: "Ragdoll",
  description:
    "A twelve-body humanoid with ball-socket hips, knees, shoulders and elbows drops down a staircase. Joint-connected pairs skip contacts, so the limbs articulate without fighting the solver.",
  camera: {
    position: [14.5, 6.5, 12],
    target: [3.6, 1.6, 0],
  },
  build,
};
