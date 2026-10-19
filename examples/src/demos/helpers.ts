import { BodyType, Box, Plane, Quat, RigidBody, Vec3 } from "impulse3d";
import type { World } from "impulse3d";

/** Options accepted by {@link addBox}. */
export interface BoxOptions {
  readonly mass?: number;
  readonly rotation?: Quat;
  readonly friction?: number;
  readonly restitution?: number;
  readonly linearDamping?: number;
  readonly angularDamping?: number;
  readonly color?: number;
}

interface BodyVisual {
  color?: number;
}

/** Attaches a display colour to a body without touching simulation state. */
export function paint(body: RigidBody, color: number): RigidBody {
  body.userData = { color } satisfies BodyVisual;
  return body;
}

/** Reads the display colour attached by {@link paint}, if any. */
export function colorFor(body: RigidBody): number | undefined {
  const visual = body.userData as BodyVisual | null;
  return visual?.color;
}

/** Adds an infinite static ground plane. */
export function addGround(world: World, friction = 0.6): RigidBody {
  return world.addBody(
    new RigidBody({ shape: new Plane(), type: BodyType.Static, friction }),
  );
}

/** Adds a static box obstacle. */
export function addStaticBox(
  world: World,
  halfExtents: Vec3,
  position: Vec3,
  rotation?: Quat,
): RigidBody {
  return world.addBody(
    new RigidBody({
      shape: new Box(halfExtents),
      type: BodyType.Static,
      position,
      rotation: rotation ?? new Quat(),
    }),
  );
}

/** Adds a dynamic box with the common demo defaults. */
export function addBox(
  world: World,
  halfExtents: Vec3,
  position: Vec3,
  options: BoxOptions = {},
): RigidBody {
  const body = world.addBody(
    new RigidBody({
      shape: new Box(halfExtents),
      mass: options.mass ?? 1,
      position,
      rotation: options.rotation ?? new Quat(),
      friction: options.friction ?? 0.5,
      restitution: options.restitution ?? 0,
      linearDamping: options.linearDamping ?? 0,
      angularDamping: options.angularDamping ?? 0,
    }),
  );
  if (options.color !== undefined) {
    paint(body, options.color);
  }
  return body;
}

/** Shared world up axis for demo layouts. */
export const Y_AXIS = new Vec3(0, 1, 0);
