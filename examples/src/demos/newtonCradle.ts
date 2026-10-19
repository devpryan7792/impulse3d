import * as THREE from "three";
import { DistanceJoint, RigidBody, Sphere, Vec3 } from "impulse3d";
import { addStaticBox, paint } from "./helpers.js";
import type { DemoContext, DemoDefinition } from "./types.js";

const PENDULUM_LENGTH = 1;
const BALL_RADIUS = 0.25;
const BALL_COUNT = 5;
const BALL_SPACING = 0.51;
const LIFT_ANGLE = 0.6;
const ANCHOR_Y = 4;

interface CradleRig {
  readonly anchors: Vec3[];
  readonly balls: RigidBody[];
  readonly lines: THREE.LineSegments;
}

let rig: CradleRig | null = null;

/**
 * Five equal spheres hang from a static beam. The leftmost one starts lifted;
 * gravity accelerates it into the row, where the perfectly elastic, equal-mass
 * collisions hand the whole momentum to the far sphere.
 */
function build(context: DemoContext): void {
  const { world } = context;

  // A simple portal frame so the pendulums have a visible anchor.
  const postHalf = new Vec3(0.12, 2, 0.12);
  addStaticBox(world, postHalf, new Vec3(-2.1, 2, 0));
  addStaticBox(world, postHalf, new Vec3(2.1, 2, 0));
  const beam = addStaticBox(world, new Vec3(2.3, 0.1, 0.1), new Vec3(0, ANCHOR_Y, 0));

  const anchors: Vec3[] = [];
  const balls: RigidBody[] = [];
  const anchorY = beam.position.y;
  for (let i = 0; i < BALL_COUNT; i++) {
    const x = (i - (BALL_COUNT - 1) / 2) * BALL_SPACING;
    const lifted = i === 0;
    const angle = lifted ? -LIFT_ANGLE : 0;
    const position = new Vec3(
      x + Math.sin(angle) * PENDULUM_LENGTH,
      anchorY - Math.cos(angle) * PENDULUM_LENGTH,
      0,
    );
    const ball = world.addBody(
      new RigidBody({
        shape: new Sphere(BALL_RADIUS),
        mass: 1,
        position,
        restitution: 1,
        friction: 0,
        linearDamping: 0.004,
        angularDamping: 0.08,
      }),
    );
    paint(ball, 0x81a1c1);
    balls.push(ball);
    anchors.push(new Vec3(x, anchorY, 0));

    world.addJoint(
      new DistanceJoint(beam, ball, {
        localAnchorA: new Vec3(x, 0, 0),
        restLength: PENDULUM_LENGTH,
      }),
    );
  }

  // Visual suspension lines: the distance joints themselves are invisible.
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.BufferAttribute(new Float32Array(BALL_COUNT * 2 * 3), 3),
  );
  const lines = new THREE.LineSegments(
    geometry,
    new THREE.LineBasicMaterial({ color: 0x5c6b82 }),
  );
  lines.frustumCulled = false;
  context.group.add(lines);
  rig = { anchors, balls, lines };
}

/** Copies the pendulum anchors and ball centres into the line buffer. */
function update(_dt: number, _context: DemoContext): void {
  if (rig === null) {
    return;
  }
  const attribute = rig.lines.geometry.getAttribute("position") as THREE.BufferAttribute;
  const array = attribute.array as Float32Array;
  let offset = 0;
  for (let i = 0; i < rig.balls.length; i++) {
    const anchor = rig.anchors[i];
    const ball = rig.balls[i].position;
    array[offset++] = anchor.x;
    array[offset++] = anchor.y;
    array[offset++] = anchor.z;
    array[offset++] = ball.x;
    array[offset++] = ball.y;
    array[offset++] = ball.z;
  }
  attribute.needsUpdate = true;
}

export const newtonCradleDemo: DemoDefinition = {
  id: "cradle",
  name: "Newton's Cradle",
  description:
    "Five equal spheres on distance joints with restitution 1. The lifted sphere's momentum passes through the row and kicks the far sphere out — the solver reflects the closing velocity without measurable loss.",
  camera: {
    position: [0, 3.4, 7.5],
    target: [0, 3.1, 0],
  },
  build,
  update,
};
