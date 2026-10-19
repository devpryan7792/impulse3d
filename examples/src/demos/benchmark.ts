import { RigidBody, Sphere, Vec3 } from "impulse3d";
import { addGround, addStaticBox, paint } from "./helpers.js";
import type { DemoContext, DemoDefinition } from "./types.js";

const COLUMNS = 40;
const ROWS = 25;
const BODY_COUNT = COLUMNS * ROWS;
const RADIUS = 0.22;
const SPACING = RADIUS * 2 + 0.12;
const PALETTE = [0x88c0d0, 0x81a1c1, 0x8fbcbb, 0xa3be8c, 0xd08770] as const;

/**
 * 1000-body stress test.
 *
 * A shallow field of 1000 spheres drops into a walled arena. Sphere contacts
 * are single-point analytic manifolds, which keeps the narrowphase and solver
 * work per body low enough to hold the 60 Hz budget on a desktop CPU. This
 * scene deliberately trades stability margin for throughput: one substep and
 * four solver iterations instead of the defaults (8 and 10). The HUD reports
 * the achieved frame rate and simulation step time.
 */
function build(context: DemoContext): void {
  const { world } = context;
  addGround(world, 0.5);

  const halfWidth = (COLUMNS / 2) * SPACING;
  const halfDepth = (ROWS / 2) * SPACING;
  addStaticBox(world, new Vec3(0.2, 0.6, halfDepth + 0.4), new Vec3(-halfWidth - 0.2, 0.6, 0));
  addStaticBox(world, new Vec3(0.2, 0.6, halfDepth + 0.4), new Vec3(halfWidth + 0.2, 0.6, 0));
  addStaticBox(world, new Vec3(halfWidth + 0.4, 0.6, 0.2), new Vec3(0, 0.6, -halfDepth - 0.2));
  addStaticBox(world, new Vec3(halfWidth + 0.4, 0.6, 0.2), new Vec3(0, 0.6, halfDepth + 0.2));

  let index = 0;
  for (let row = 0; row < ROWS; row++) {
    for (let column = 0; column < COLUMNS; column++) {
      const x = (column - (COLUMNS - 1) / 2) * SPACING + ((row % 2) * SPACING) / 2;
      const z = (row - (ROWS - 1) / 2) * SPACING;
      world.addBody(
        paint(
          new RigidBody({
            shape: new Sphere(RADIUS),
            mass: 1,
            position: new Vec3(x, 1.5, z),
            // A little lateral energy makes the field collide and spread.
            linearVelocity: new Vec3((column % 5) - 2, 0, (row % 3) - 1),
            friction: 0.5,
            restitution: 0.2,
            linearDamping: 0.2,
            angularDamping: 0.25,
          }),
          PALETTE[index % PALETTE.length],
        ),
      );
      index++;
    }
  }
}

export const benchmarkDemo: DemoDefinition = {
  id: "benchmark",
  name: "1000-Body Benchmark",
  description: `A shallow field of ${BODY_COUNT} spheres drops into a walled arena, tuned to one substep and four solver iterations with shadows off to sustain the 60 FPS budget. Watch fps and step ms in the HUD.`,
  camera: {
    position: [14, 10, 16],
    target: [0, 1, 0],
  },
  worldOptions: {
    substeps: 1,
    solverIterations: 4,
    broadphaseMargin: 0.01,
  },
  castShadows: false,
  build,
};
