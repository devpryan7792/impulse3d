import * as THREE from "three";
import { RigidBody, Sphere, Vec3, World } from "impulse3d";
import { MouseSpring } from "./MouseSpring.js";
import { ThreeVisualizer } from "./ThreeVisualizer.js";
import { demos } from "./demos/index.js";
import type { DemoContext } from "./demos/types.js";

/**
 * Headless demo smoke test: builds every scene without a WebGL context, runs
 * the fixed simulation loop and checks for non-finite state, runaway bodies
 * and demo-specific invariants. Run with `npm run demo:smoke`.
 */
const TICKS = 480;

function assert(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(message);
  }
}

function finiteState(world: World): boolean {
  for (const body of world.bodies) {
    if (!body.position.isFinite() || !body.linearVelocity.isFinite()) {
      return false;
    }
    if (!Number.isFinite(body.angularVelocity.x) || !Number.isFinite(body.rotation.w)) {
      return false;
    }
  }
  return true;
}

for (const demo of demos) {
  const scene = new THREE.Scene();
  const group = new THREE.Group();
  scene.add(group);
  const world = new World(demo.worldOptions);
  const visualizer = new ThreeVisualizer(scene);
  const context: DemoContext = { world, scene, group, visualizer };

  demo.build(context);
  visualizer.addMissingBodies(world);

  const startPositions = world.bodies.map((body) => body.position.clone());
  if (demo.id === "vehicle") {
    demo.onKey?.("KeyW", true, context);
  }

  let nanAt = -1;
  let maxCoordinate = 0;
  let maxDynamicX = -Infinity;
  let vehicleHeading = 0;
  const started = performance.now();
  for (let tick = 0; tick < TICKS; tick++) {
    if (demo.id === "vehicle" && tick === 240) {
      demo.onKey?.("KeyA", true, context);
    }
    demo.update?.(1 / 60, context);
    world.step(1 / 60);
    if (nanAt < 0 && !finiteState(world)) {
      nanAt = tick;
    }
    for (const body of world.bodies) {
      maxCoordinate = Math.max(
        maxCoordinate,
        Math.abs(body.position.x),
        Math.abs(body.position.y),
        Math.abs(body.position.z),
      );
      if (body.isDynamic) {
        maxDynamicX = Math.max(maxDynamicX, body.position.x);
        if (demo.id === "vehicle") {
          const forward = body.transform.transformDirection(new Vec3(), new Vec3(1, 0, 0));
          vehicleHeading = (Math.atan2(forward.z, forward.x) * 180) / Math.PI;
        }
      }
    }
  }
  const elapsed = performance.now() - started;

  // Exercise the debug overlay paths (AABB and contact-normal buffers).
  visualizer.setDebug(true);
  visualizer.syncTransforms(world);

  const stats = world.getStats();
  let moved = 0;
  for (let i = 0; i < world.bodies.length; i++) {
    if (world.bodies[i].position.distanceTo(startPositions[i]) > 0.5) {
      moved++;
    }
  }
  let maxTilt = 0;
  let top = -Infinity;
  for (const body of world.bodies) {
    if (!body.isDynamic) {
      continue;
    }
    top = Math.max(top, body.position.y);
    const up = body.transform.transformDirection(new Vec3(), new Vec3(0, 1, 0));
    maxTilt = Math.max(maxTilt, Math.acos(Math.min(1, Math.abs(up.y))) * (180 / Math.PI));
  }

  console.log(
    `${demo.id.padEnd(11)} bodies=${String(stats.bodyCount).padStart(4)}` +
      ` awake=${String(stats.awakeBodies).padStart(4)}` +
      ` sleeping=${String(stats.sleepingBodies).padStart(4)}` +
      ` contacts=${String(stats.contactPoints).padStart(4)}` +
      ` ms/tick=${(elapsed / TICKS).toFixed(2).padStart(6)}` +
      ` moved=${String(moved).padStart(4)} maxCoord=${maxCoordinate.toFixed(1).padStart(6)}` +
      ` maxTilt=${maxTilt.toFixed(0).padStart(3)}°` +
      (demo.id === "vehicle" ? ` heading=${vehicleHeading.toFixed(0)}°` : ""),
  );

  assert(nanAt < 0, `${demo.id}: non-finite state at tick ${nanAt}`);
  assert(maxCoordinate < 500, `${demo.id}: runaway body at ${maxCoordinate}`);
  switch (demo.id) {
    case "jenga": {
      assert(top > 9, `jenga: tower collapsed (top ${top.toFixed(2)})`);
      break;
    }
    case "cradle": {
      assert(
        maxDynamicX > 1.3,
        `cradle: momentum did not reach the far sphere (${maxDynamicX.toFixed(2)})`,
      );
      break;
    }
    case "ragdoll": {
      assert(moved >= 4 && maxTilt > 45, `ragdoll: did not topple (${moved} moved, ${maxTilt.toFixed(0)}°)`);
      break;
    }
    case "vehicle": {
      let travelled = 0;
      for (let i = 0; i < world.bodies.length; i++) {
        const body = world.bodies[i];
        if (body.isDynamic) {
          travelled = Math.max(travelled, body.position.distanceTo(startPositions[i]));
        }
      }
      assert(travelled > 2, `vehicle: did not drive (${travelled.toFixed(2)} m)`);
      assert(Math.abs(vehicleHeading) > 5, `vehicle: did not steer (${vehicleHeading.toFixed(1)}°)`);
      break;
    }
    case "demolition": {
      assert(moved > 5, `demolition: wall did not break (${moved} bodies moved)`);
      break;
    }
    case "benchmark": {
      assert(stats.sleepingBodies > 0, "benchmark: no bodies settled");
      break;
    }
    default:
      break;
  }

  visualizer.dispose(scene);
}

console.log("demo smoke test passed");

// --- MouseSpring: raycast pick, drag target and spring force ---------------
{
  const scene = new THREE.Scene();
  const world = new World();
  const visualizer = new ThreeVisualizer(scene);
  const body = world.addBody(
    new RigidBody({ shape: new Sphere(0.5), mass: 1, position: new Vec3(0, 0, 0) }),
  );
  const mesh = visualizer.addBody(body);

  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 5);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();

  const element = {
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 100, height: 100 }),
  } as unknown as HTMLElement;
  const spring = new MouseSpring(camera, element);

  const center = { clientX: 50, clientY: 50, pointerId: 1 } as unknown as PointerEvent;
  assert(spring.begin(center, [mesh]), "mouse spring: raycast missed the sphere");
  spring.move({ clientX: 72, clientY: 50, pointerId: 1 } as unknown as PointerEvent);
  for (let i = 0; i < 30; i++) {
    spring.applyForces(world);
    world.step(1 / 60);
  }
  assert(body.position.x > 0.2, `mouse spring: body did not follow (${body.position.x.toFixed(2)})`);
  spring.release();
  assert(spring.grabbed === null, "mouse spring: release failed");
  visualizer.dispose(scene);
}

console.log("mouse spring smoke test passed");
