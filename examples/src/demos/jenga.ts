import { Quat, Vec3 } from "impulse3d";
import { addBox, addGround, Y_AXIS } from "./helpers.js";
import type { DemoContext, DemoDefinition } from "./types.js";

const BLOCK_HALF = new Vec3(0.5, 0.15, 0.15);
const LAYERS = 33;

/**
 * Classic Jenga layout: three blocks per layer, each layer rotated ninety
 * degrees, for 100 bodies in total (33 full layers plus a crown block).
 *
 * The tower is intentionally tall and slender: it only survives because the
 * solver warm-starts normal impulses and uses split-impulse position
 * correction. The exact-rest layout is pre-settled and put to sleep so the
 * demo opens instantly; any interaction wakes the affected island.
 */
function build(context: DemoContext): void {
  const { world } = context;
  addGround(world, 0.6);

  const evenRotation = new Quat();
  const oddRotation = Quat.fromAxisAngle(Y_AXIS, Math.PI / 2);

  for (let layer = 0; layer < LAYERS; layer++) {
    const even = layer % 2 === 0;
    const rotation = even ? evenRotation : oddRotation;
    const color = even ? 0xd8b48a : 0xc39a6b;
    for (let slot = 0; slot < 3; slot++) {
      const offset = (slot - 1) * 0.32;
      const position = even
        ? new Vec3(0, 0.15 + layer * 0.3, offset)
        : new Vec3(offset, 0.15 + layer * 0.3, 0);
      addBox(world, BLOCK_HALF, position, {
        rotation,
        friction: 0.55,
        color,
      });
    }
  }

  addBox(world, BLOCK_HALF, new Vec3(0, 0.15 + LAYERS * 0.3, 0), {
    friction: 0.55,
    color: 0xbf616a,
  });

  // The layout is built at exact rest. Put the completed tower to sleep so the
  // demo opens instantly; grabbing a block wakes its island through contacts.
  for (const body of world.bodies) {
    body.sleep();
  }
}

export const jengaDemo: DemoDefinition = {
  id: "jenga",
  name: "Jenga 100",
  description:
    "100 alternating Jenga blocks in a 10 m tower, pre-settled and asleep (island sleeping keeps the whole tower at ~0 ms/frame). Grab a block to wake the stack.",
  camera: {
    position: [13, 7.5, 15],
    target: [0, 5, 0],
  },
  build,
};
