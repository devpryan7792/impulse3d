import * as THREE from "three";
import { ConvexGeometry } from "three/examples/jsm/geometries/ConvexGeometry.js";
import { BodyType, ShapeType } from "impulse3d";
import type {
  Box,
  Capsule,
  ConvexPolytope,
  RigidBody,
  Shape,
  Sphere,
  World,
} from "impulse3d";

/** Colours and debug-overlay defaults for {@link ThreeVisualizer}. */
export interface VisualizerOptions {
  /** Colour used for dynamic bodies unless overridden per body. */
  dynamicColor?: number;
  /** Colour used for static and kinematic bodies. */
  staticColor?: number;
  /** Show the AABB/contact-normal overlay from the start (default `false`). */
  debug?: boolean;
}

interface TrackedBody {
  readonly body: RigidBody;
  readonly mesh: THREE.Mesh;
  readonly geometry: THREE.BufferGeometry;
  readonly material: THREE.MeshStandardMaterial;
}

/** The twelve edges of a unit box, as index pairs into its eight corners. */
const AABB_EDGES: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [2, 3], [4, 5], [6, 7],
  [0, 2], [1, 3], [4, 6], [5, 7],
  [0, 4], [1, 5], [2, 6], [3, 7],
];

/** Unit-cube corner offsets in bitset order `(x, y, z)`. */
const AABB_CORNERS: ReadonlyArray<readonly [number, number, number]> = [
  [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
  [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
];

/**
 * Maps an Impulse3D shape onto an equivalent Three.js geometry.
 *
 * The engine's shapes are all convex primitives with an explicit support
 * mapping, so this is a pure visual translation and never feeds back into
 * simulation state.
 */
function geometryFor(shape: Shape): THREE.BufferGeometry {
  switch (shape.type) {
    case ShapeType.Sphere: {
      const sphere = shape as Sphere;
      return new THREE.SphereGeometry(sphere.radius, 20, 14);
    }
    case ShapeType.Box: {
      const box = shape as Box;
      return new THREE.BoxGeometry(
        box.halfExtents.x * 2,
        box.halfExtents.y * 2,
        box.halfExtents.z * 2,
      );
    }
    case ShapeType.Capsule: {
      const capsule = shape as Capsule;
      return new THREE.CapsuleGeometry(
        capsule.radius,
        capsule.halfHeight * 2,
        8,
        18,
      );
    }
    case ShapeType.Plane: {
      const plane = new THREE.PlaneGeometry(80, 80);
      // The engine plane's normal is local +Y; Three's PlaneGeometry faces +Z.
      plane.rotateX(-Math.PI / 2);
      return plane;
    }
    case ShapeType.ConvexPolytope: {
      const polytope = shape as ConvexPolytope;
      return new ConvexGeometry(
        polytope.vertices.map((v) => new THREE.Vector3(v.x, v.y, v.z)),
      );
    }
    default:
      throw new RangeError(`ThreeVisualizer: unsupported shape type ${String(shape.type)}`);
  }
}

/**
 * Keeps one Three.js mesh per rigid body in sync with the simulation and
 * renders an optional wireframe debug overlay (AABBs and contact normals).
 *
 * The visualizer never mutates physics state; `syncTransforms` is a one-way
 * copy of body transforms into the scene graph.
 */
export class ThreeVisualizer {
  /** Scene-graph parent of every body mesh. */
  public readonly group = new THREE.Group();

  private readonly tracked: TrackedBody[] = [];
  private readonly dynamicColor: number;
  private readonly staticColor: number;

  private readonly debugGroup = new THREE.Group();
  private readonly aabbGeometry = new THREE.BufferGeometry();
  private readonly aabbLines: THREE.LineSegments;
  private readonly normalGeometry = new THREE.BufferGeometry();
  private readonly normalLines: THREE.LineSegments;
  private aabbBufferSize = 0;
  private normalBufferSize = 0;
  private debugEnabled: boolean;

  public constructor(scene: THREE.Scene, options: VisualizerOptions = {}) {
    this.dynamicColor = options.dynamicColor ?? 0xd8dee9;
    this.staticColor = options.staticColor ?? 0x4c566a;

    this.group.name = "impulse3d-bodies";
    scene.add(this.group);

    this.aabbLines = new THREE.LineSegments(
      this.aabbGeometry,
      new THREE.LineBasicMaterial({ color: 0x88c0d0, transparent: true, opacity: 0.55 }),
    );
    this.aabbLines.frustumCulled = false;
    this.normalLines = new THREE.LineSegments(
      this.normalGeometry,
      new THREE.LineBasicMaterial({ color: 0xbf616a }),
    );
    this.normalLines.frustumCulled = false;
    this.debugGroup.name = "impulse3d-debug";
    this.debugGroup.add(this.aabbLines, this.normalLines);
    this.debugGroup.visible = false;
    scene.add(this.debugGroup);

    this.debugEnabled = options.debug ?? false;
    this.debugGroup.visible = this.debugEnabled;
  }

  /** Meshes of dynamic bodies, suitable for pointer raycasting. */
  public get pickables(): THREE.Object3D[] {
    return this.tracked.filter((entry) => entry.body.isDynamic).map((entry) => entry.mesh);
  }

  /** Adds a mesh for `body`; `color` overrides the type-based default. */
  public addBody(body: RigidBody, color?: number): THREE.Mesh {
    const geometry = geometryFor(body.shape);
    const material = new THREE.MeshStandardMaterial({
      color: color ?? (body.type === BodyType.Dynamic ? this.dynamicColor : this.staticColor),
      roughness: body.type === BodyType.Dynamic ? 0.62 : 0.92,
      metalness: body.type === BodyType.Dynamic ? 0.08 : 0.02,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = body.type === BodyType.Dynamic;
    mesh.receiveShadow = true;
    mesh.position.set(body.position.x, body.position.y, body.position.z);
    mesh.quaternion.set(body.rotation.x, body.rotation.y, body.rotation.z, body.rotation.w);
    mesh.userData.body = body;
    this.group.add(mesh);
    this.tracked.push({ body, mesh, geometry, material });
    return mesh;
  }

  /** Adds meshes for every body in `world` not tracked yet. */
  public addMissingBodies(
    world: World,
    colorFor?: (body: RigidBody) => number | undefined,
  ): void {
    for (const body of world.bodies) {
      if (this.tracked.some((entry) => entry.body === body)) {
        continue;
      }
      this.addBody(body, colorFor?.(body));
    }
  }

  /** Removes and disposes the mesh of `body`, if tracked. */
  public removeBody(body: RigidBody): void {
    const index = this.tracked.findIndex((entry) => entry.body === body);
    if (index < 0) {
      return;
    }
    const [entry] = this.tracked.splice(index, 1);
    this.group.remove(entry.mesh);
    entry.geometry.dispose();
    entry.material.dispose();
  }

  /** Removes every tracked mesh and releases its GPU resources. */
  public clear(): void {
    for (const entry of this.tracked) {
      this.group.remove(entry.mesh);
      entry.geometry.dispose();
      entry.material.dispose();
    }
    this.tracked.length = 0;
    this.aabbGeometry.setDrawRange(0, 0);
    this.normalGeometry.setDrawRange(0, 0);
  }

  /** Toggles the AABB + contact-normal overlay. */
  public setDebug(enabled: boolean): void {
    this.debugEnabled = enabled;
    this.debugGroup.visible = enabled;
  }

  /** Enables or disables shadow casting for all dynamic body meshes. */
  public setShadows(enabled: boolean): void {
    for (const { body, mesh } of this.tracked) {
      mesh.castShadow = enabled && body.isDynamic;
    }
  }

  /** Returns whether the debug overlay is currently visible. */
  public get debug(): boolean {
    return this.debugEnabled;
  }

  /** Copies every tracked body transform into its mesh. */
  public syncTransforms(world: World): void {
    for (const { body, mesh } of this.tracked) {
      mesh.position.set(body.position.x, body.position.y, body.position.z);
      mesh.quaternion.set(body.rotation.x, body.rotation.y, body.rotation.z, body.rotation.w);
    }
    if (this.debugEnabled) {
      this.updateDebugOverlay(world);
    }
  }

  /** Renders the scene through `renderer`. */
  public render(
    renderer: THREE.WebGLRenderer,
    scene: THREE.Scene,
    camera: THREE.Camera,
  ): void {
    renderer.render(scene, camera);
  }

  /** Releases the body meshes and debug resources from `scene`. */
  public dispose(scene: THREE.Scene): void {
    this.clear();
    scene.remove(this.group);
    scene.remove(this.debugGroup);
    this.aabbGeometry.dispose();
    this.normalGeometry.dispose();
    (this.aabbLines.material as THREE.Material).dispose();
    (this.normalLines.material as THREE.Material).dispose();
  }

  private ensureAabbCapacity(floats: number): void {
    if (this.aabbBufferSize !== 0 && floats <= this.aabbBufferSize) {
      return;
    }
    const size = Math.max(floats, this.aabbBufferSize * 2, 24 * 3 * 16);
    this.aabbGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(size), 3),
    );
    this.aabbBufferSize = size;
  }

  private ensureNormalCapacity(floats: number): void {
    if (this.normalBufferSize !== 0 && floats <= this.normalBufferSize) {
      return;
    }
    const size = Math.max(floats, this.normalBufferSize * 2, 2 * 3 * 64);
    this.normalGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(size), 3),
    );
    this.normalBufferSize = size;
  }

  private updateDebugOverlay(world: World): void {
    // --- AABBs: twelve segments per body -----------------------------------
    this.ensureAabbCapacity(this.tracked.length * AABB_EDGES.length * 2 * 3);
    const aabbAttribute = this.aabbGeometry.getAttribute("position") as THREE.BufferAttribute;
    const aabbArray = aabbAttribute.array as Float32Array;
    let aabbOffset = 0;
    for (const { body } of this.tracked) {
      const bounds = body.computeAABB();
      const { min, max } = bounds;
      for (const [from, to] of AABB_EDGES) {
        const a = AABB_CORNERS[from];
        const b = AABB_CORNERS[to];
        aabbArray[aabbOffset++] = a[0] === 0 ? min.x : max.x;
        aabbArray[aabbOffset++] = a[1] === 0 ? min.y : max.y;
        aabbArray[aabbOffset++] = a[2] === 0 ? min.z : max.z;
        aabbArray[aabbOffset++] = b[0] === 0 ? min.x : max.x;
        aabbArray[aabbOffset++] = b[1] === 0 ? min.y : max.y;
        aabbArray[aabbOffset++] = b[2] === 0 ? min.z : max.z;
      }
    }
    aabbAttribute.needsUpdate = true;
    this.aabbGeometry.setDrawRange(0, aabbOffset / 3);

    // --- Contact normals: one segment per manifold point --------------------
    let points = 0;
    for (const constraint of world.contactSolver.constraints) {
      points += constraint.points.length;
    }
    this.ensureNormalCapacity(points * 2 * 3);
    const normalAttribute = this.normalGeometry.getAttribute("position") as THREE.BufferAttribute;
    const normalArray = normalAttribute.array as Float32Array;
    let normalOffset = 0;
    for (const constraint of world.contactSolver.constraints) {
      for (const point of constraint.points) {
        const { position } = point;
        normalArray[normalOffset++] = position.x;
        normalArray[normalOffset++] = position.y;
        normalArray[normalOffset++] = position.z;
        normalArray[normalOffset++] = position.x + constraint.normal.x * 0.2;
        normalArray[normalOffset++] = position.y + constraint.normal.y * 0.2;
        normalArray[normalOffset++] = position.z + constraint.normal.z * 0.2;
      }
    }
    normalAttribute.needsUpdate = true;
    this.normalGeometry.setDrawRange(0, normalOffset / 3);
  }
}
