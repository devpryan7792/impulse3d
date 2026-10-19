import { DistanceJoint, RigidBody, Sphere, Vec3 } from "impulse3d";
import { addBox, addGround, addStaticBox, paint } from "./helpers.js";
import type { DemoContext, DemoDefinition } from "./types.js";

const WALL_ROWS = 10;
const WALL_COLUMNS = 8;
const BRICK_HALF = new Vec3(0.35, 0.18, 0.12);
const BALL_RADIUS = 0.6;
const PENDULUM_LENGTH = 3;

/** Static portal frame that carries the wrecking ball. */
function buildFrame(context: DemoContext): RigidBody {
  const { world } = context;
  const postHalf = new Vec3(0.12, 2.5, 0.12);
  addStaticBox(world, postHalf, new Vec3(-1.6, 2.5, 2.5));
  addStaticBox(world, postHalf, new Vec3(1.6, 2.5, 2.5));
  return addStaticBox(world, new Vec3(1.75, 0.12, 0.12), new Vec3(0, 5, 2.5));
}

/** Running-bond brick wall. */
function buildWall(context: DemoContext): void {
  const { world } = context;
  for (let row = 0; row < WALL_ROWS; row++) {
    for (let column = 0; column < WALL_COLUMNS; column++) {
      const position = new Vec3(
        -2.45 + column * 0.7 + (row % 2) * 0.35,
        BRICK_HALF.y + row * BRICK_HALF.y * 2,
        0,
      );
      addBox(world, BRICK_HALF, position, {
        friction: 0.7,
        color: row % 2 === 0 ? 0xbf616a : 0xd08770,
      });
    }
  }
}

function build(context: DemoContext): void {
  const { world } = context;
  addGround(world, 0.7);
  const beam = buildFrame(context);
  buildWall(context);

  const ball = world.addBody(
    paint(
      new RigidBody({
        shape: new Sphere(BALL_RADIUS),
        mass: 220,
        position: new Vec3(0, 5 - PENDULUM_LENGTH, 2.5),
        linearVelocity: new Vec3(0, 0, -6),
        friction: 0.4,
        restitution: 0.15,
        linearDamping: 0.01,
      }),
      0x2e3440,
    ),
  );
  world.addJoint(new DistanceJoint(beam, ball, { restLength: PENDULUM_LENGTH }));
}

export const demolitionDemo: DemoDefinition = {
  id: "demolition",
  name: "Demolition",
  description:
    "An 80-brick running-bond wall meets a 220 kg wrecking ball on a distance joint. Grab the ball with the pointer to wind it back for another swing.",
  camera: {
    position: [10.5, 5.5, 13],
    target: [0, 2.2, 0.8],
  },
  build,
};
