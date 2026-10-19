import type * as THREE from "three";
import type { World, WorldOptions } from "impulse3d";
import type { ThreeVisualizer } from "../ThreeVisualizer.js";

/** Everything a demo may use while constructing its scene. */
export interface DemoContext {
  readonly world: World;
  readonly scene: THREE.Scene;
  /** Parent for visual-only objects (wheels, markers); cleared between demos. */
  readonly group: THREE.Group;
  readonly visualizer: ThreeVisualizer;
}

/** Camera placement for a demo, in world units. */
export interface DemoCamera {
  readonly position: readonly [number, number, number];
  readonly target: readonly [number, number, number];
}

/** A self-contained interactive scene shown by the demo browser. */
export interface DemoDefinition {
  readonly id: string;
  readonly name: string;
  readonly description: string;
  readonly camera: DemoCamera;
  /** World tuning for this scene (defaults to the engine defaults). */
  readonly worldOptions?: WorldOptions;
  /** Render dynamic bodies with shadows (default `true`). */
  readonly castShadows?: boolean;
  /** Populates the world and any visual-only objects. */
  build(context: DemoContext): void;
  /** Called once per fixed simulation step, before `world.step`. */
  update?(dt: number, context: DemoContext): void;
  /** Keyboard input forwarded from the page (also fired for key releases). */
  onKey?(code: string, down: boolean, context: DemoContext): void;
}
