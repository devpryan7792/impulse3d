import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { World } from "impulse3d";
import { MouseSpring } from "./MouseSpring.js";
import { ThreeVisualizer } from "./ThreeVisualizer.js";
import { colorFor } from "./demos/helpers.js";
import type { DemoContext, DemoDefinition } from "./demos/types.js";

/** Live counters pushed to the HUD. */
export interface SceneStats {
  readonly fps: number;
  readonly stepMs: number;
  readonly bodies: number;
  readonly awake: number;
  readonly contacts: number;
}

/** Callbacks used by the page shell. */
export interface PhysicsSceneHooks {
  onStats?(stats: SceneStats): void;
}

const FIXED_STEP = 1 / 60;
const MAX_STEPS_PER_FRAME = 4;
const STATS_INTERVAL_MS = 250;

/** Recursively disposes geometries and materials below `root`. */
function disposeObject(root: THREE.Object3D): void {
  root.traverse((object) => {
    if (root === object) {
      return;
    }
    const mesh = object as THREE.Mesh;
    mesh.geometry?.dispose();
    const material = mesh.material;
    if (Array.isArray(material)) {
      for (const entry of material) {
        entry.dispose();
      }
    } else {
      material?.dispose();
    }
  });
}

/**
 * Browser shell for the engine: renderer, camera, lights, orbit controls,
 * pointer grabbing and the fixed-timestep simulation loop.
 */
export class PhysicsScene {
  public world: World;
  public readonly scene = new THREE.Scene();
  public readonly camera: THREE.PerspectiveCamera;
  public readonly renderer: THREE.WebGLRenderer;
  public readonly controls: OrbitControls;

  private demo: DemoDefinition;
  private context: DemoContext;
  private visualizer: ThreeVisualizer;
  private readonly mouseSpring: MouseSpring;
  private readonly hooks: PhysicsSceneHooks;

  private readonly demoGroup = new THREE.Group();
  private readonly canvas: HTMLCanvasElement;
  private readonly resizeHandler = (): void => this.resize();

  private accumulator = 0;
  private lastTime = performance.now();
  private lastStatsTime = 0;
  private frameHandle = 0;
  private paused = false;
  private debugEnabled = false;
  private fps = 60;

  public constructor(
    canvas: HTMLCanvasElement,
    initialDemo: DemoDefinition,
    hooks: PhysicsSceneHooks = {},
  ) {
    this.canvas = canvas;
    this.hooks = hooks;
    this.demo = initialDemo;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;

    this.scene.background = new THREE.Color(0x0b0e14);
    this.scene.fog = new THREE.Fog(0x0b0e14, 34, 80);
    this.scene.add(this.demoGroup);

    this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 220);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.minDistance = 2;
    this.controls.maxDistance = 60;

    const hemisphere = new THREE.HemisphereLight(0xdfe9ff, 0x1a2030, 1.2);
    this.scene.add(hemisphere);

    const sun = new THREE.DirectionalLight(0xffffff, 2.1);
    sun.position.set(9, 16, 7);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.bias = -0.0004;
    const shadowCamera = sun.shadow.camera;
    shadowCamera.left = -18;
    shadowCamera.right = 18;
    shadowCamera.top = 18;
    shadowCamera.bottom = -18;
    shadowCamera.near = 0.5;
    shadowCamera.far = 60;
    this.scene.add(sun);

    this.world = new World();
    this.visualizer = new ThreeVisualizer(this.scene);
    this.context = {
      world: this.world,
      scene: this.scene,
      group: this.demoGroup,
      visualizer: this.visualizer,
    };

    this.mouseSpring = new MouseSpring(this.camera, canvas);
    this.attachPointerHandlers();
    this.attachKeyboardHandlers();
    this.loadDemo(initialDemo);
  }

  /** Swaps in a fresh world and rebuilds `demo`'s scene. */
  public loadDemo(demo: DemoDefinition): void {
    this.demo = demo;
    this.mouseSpring.release();
    this.controls.enabled = true;
    this.paused = false;

    disposeObject(this.demoGroup);
    this.demoGroup.clear();

    this.visualizer.dispose(this.scene);
    this.world = new World(demo.worldOptions);
    this.visualizer = new ThreeVisualizer(this.scene, { debug: this.debugEnabled });
    this.context = {
      world: this.world,
      scene: this.scene,
      group: this.demoGroup,
      visualizer: this.visualizer,
    };

    demo.build(this.context);
    this.visualizer.addMissingBodies(this.world, colorFor);
    this.visualizer.setShadows(demo.castShadows ?? true);

    this.camera.position.set(...demo.camera.position);
    this.controls.target.set(...demo.camera.target);
    this.controls.update();
    this.accumulator = 0;
    this.lastTime = performance.now();
  }

  /** Restarts the active demo. */
  public reset(): void {
    this.loadDemo(this.demo);
  }

  /** Toggles the fixed-step simulation. */
  public togglePause(): void {
    this.paused = !this.paused;
    this.accumulator = 0;
  }

  /** Toggles the AABB/contact-normal overlay. */
  public toggleDebug(): void {
    this.debugEnabled = !this.debugEnabled;
    this.visualizer.setDebug(this.debugEnabled);
  }

  /** Starts the render loop (idempotent). */
  public start(): void {
    if (this.frameHandle !== 0) {
      return;
    }
    this.resize();
    window.addEventListener("resize", this.resizeHandler);
    this.lastTime = performance.now();
    const loop = (now: number): void => {
      this.frameHandle = requestAnimationFrame(loop);
      this.frame(now);
    };
    this.frameHandle = requestAnimationFrame(loop);
  }

  /** Stops the render loop. */
  public stop(): void {
    if (this.frameHandle !== 0) {
      cancelAnimationFrame(this.frameHandle);
      this.frameHandle = 0;
    }
    window.removeEventListener("resize", this.resizeHandler);
  }

  private frame(now: number): void {
    const frameDt = Math.min((now - this.lastTime) / 1000, 0.25);
    this.lastTime = now;
    this.controls.update();

    let stepMs = 0;
    if (!this.paused) {
      this.accumulator += frameDt;
      const started = performance.now();
      let steps = 0;
      while (this.accumulator >= FIXED_STEP && steps < MAX_STEPS_PER_FRAME) {
        this.mouseSpring.applyForces(this.world);
        this.demo.update?.(FIXED_STEP, this.context);
        this.world.step(FIXED_STEP);
        this.accumulator -= FIXED_STEP;
        steps++;
      }
      if (steps > 0) {
        stepMs = (performance.now() - started) / steps;
      }
      if (this.accumulator > FIXED_STEP * MAX_STEPS_PER_FRAME) {
        this.accumulator = 0;
      }
    } else {
      stepMs = 0;
    }

    this.visualizer.syncTransforms(this.world);
    this.visualizer.render(this.renderer, this.scene, this.camera);

    if (frameDt > 0) {
      this.fps += (1 / frameDt - this.fps) * 0.08;
    }
    if (now - this.lastStatsTime >= STATS_INTERVAL_MS) {
      this.lastStatsTime = now;
      const stats = this.world.getStats();
      let dynamicBodies = 0;
      for (const body of this.world.bodies) {
        if (body.isDynamic) {
          dynamicBodies++;
        }
      }
      this.hooks.onStats?.({
        fps: this.fps,
        stepMs,
        bodies: dynamicBodies,
        awake: stats.awakeBodies,
        contacts: stats.contactPoints,
      });
    }
  }

  private resize(): void {
    const width = this.canvas.clientWidth || window.innerWidth;
    const height = this.canvas.clientHeight || window.innerHeight;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / Math.max(height, 1);
    this.camera.updateProjectionMatrix();
  }

  private handleKey(code: string, down: boolean, repeat: boolean): void {
    if (down && !repeat) {
      if (code === "KeyP") {
        this.togglePause();
        return;
      }
      if (code === "KeyR") {
        this.reset();
        return;
      }
      if (code === "KeyB") {
        this.toggleDebug();
        return;
      }
    }
    this.demo.onKey?.(code, down, this.context);
  }

  private attachKeyboardHandlers(): void {
    window.addEventListener("keydown", (event) => {
      if (
        event.code === "Space" ||
        event.code === "ArrowUp" ||
        event.code === "ArrowDown" ||
        event.code === "ArrowLeft" ||
        event.code === "ArrowRight"
      ) {
        event.preventDefault();
      }
      this.handleKey(event.code, true, event.repeat);
    });
    window.addEventListener("keyup", (event) => {
      this.handleKey(event.code, false, false);
    });
  }

  private attachPointerHandlers(): void {
    this.canvas.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) {
        return;
      }
      if (this.mouseSpring.begin(event, this.visualizer.pickables)) {
        this.controls.enabled = false;
        this.canvas.setPointerCapture(event.pointerId);
      }
    });
    this.canvas.addEventListener("pointermove", (event) => {
      this.mouseSpring.move(event);
    });
    const release = (event: PointerEvent): void => {
      this.mouseSpring.release();
      this.controls.enabled = true;
      if (this.canvas.hasPointerCapture(event.pointerId)) {
        this.canvas.releasePointerCapture(event.pointerId);
      }
    };
    this.canvas.addEventListener("pointerup", release);
    this.canvas.addEventListener("pointercancel", release);
  }
}
