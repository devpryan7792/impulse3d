import * as THREE from "three";
import { Vec3 } from "impulse3d";
import type { RigidBody, World } from "impulse3d";

/** Tuning for the pointer-driven spring constraint. */
export interface MouseSpringOptions {
  /** Spring stiffness in N/m (default `900`). */
  stiffness?: number;
  /** Damping coefficient in N·s/m (default `60`). */
  damping?: number;
  /** Clamp for the applied force magnitude in N (default `20000`). */
  maxForce?: number;
}

/**
 * Raycast spring grabber: picks a dynamic body under the pointer and pulls its
 * local grab point towards a target that slides on a camera-facing plane.
 *
 * The spring is a soft constraint (force-based), so heavy bodies feel sluggish
 * and light ones snap — exactly the behaviour expected from a mouse grab.
 */
export class MouseSpring {
  private readonly raycaster = new THREE.Raycaster();
  private readonly pointer = new THREE.Vector2();
  private readonly dragPlane = new THREE.Plane();
  private readonly intersection = new THREE.Vector3();
  private readonly localAnchor = new Vec3();
  private readonly anchorWorld = new Vec3();
  private readonly velocity = new Vec3();
  private readonly force = new Vec3();
  private readonly target = new Vec3();

  private readonly camera: THREE.Camera;
  private readonly element: HTMLElement;
  private readonly stiffness: number;
  private readonly damping: number;
  private readonly maxForce: number;

  private body: RigidBody | null = null;

  public constructor(
    camera: THREE.Camera,
    element: HTMLElement,
    options: MouseSpringOptions = {},
  ) {
    this.camera = camera;
    this.element = element;
    this.stiffness = options.stiffness ?? 900;
    this.damping = options.damping ?? 60;
    this.maxForce = options.maxForce ?? 20000;
  }

  /** The body currently being pulled, if any. */
  public get grabbed(): RigidBody | null {
    return this.body;
  }

  /**
   * Attempts to grab the nearest dynamic body under the pointer.
   * @returns `true` when a body was grabbed.
   */
  public begin(event: PointerEvent, pickables: THREE.Object3D[]): boolean {
    this.readPointer(event);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hit = this.raycaster.intersectObjects(pickables, false)[0];
    if (hit === undefined) {
      return false;
    }
    const body = hit.object.userData.body as RigidBody | undefined;
    if (body === undefined || !body.isDynamic) {
      return false;
    }

    this.body = body;
    body.wake();
    const worldPoint = new Vec3(hit.point.x, hit.point.y, hit.point.z);
    body.transform.inverseTransformPoint(this.localAnchor, worldPoint);
    this.target.copy(worldPoint);

    const normal = this.camera.getWorldDirection(new THREE.Vector3());
    this.dragPlane.setFromNormalAndCoplanarPoint(normal, hit.point);
    return true;
  }

  /** Moves the spring target to the pointer's projection on the drag plane. */
  public move(event: PointerEvent): void {
    if (this.body === null) {
      return;
    }
    this.readPointer(event);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    if (this.raycaster.ray.intersectPlane(this.dragPlane, this.intersection) !== null) {
      this.target.set(this.intersection.x, this.intersection.y, this.intersection.z);
    }
  }

  /** Releases the grabbed body. */
  public release(): void {
    this.body = null;
  }

  /** Applies the spring force; call once per simulation step. */
  public applyForces(world: World): void {
    const body = this.body;
    if (body === null) {
      return;
    }
    if (!world.bodies.includes(body) || !body.isDynamic) {
      this.release();
      return;
    }

    body.transform.transformPoint(this.anchorWorld, this.localAnchor);
    body.getVelocityAtPoint(this.anchorWorld, this.velocity);
    this.force
      .copy(this.target)
      .sub(this.anchorWorld)
      .scale(this.stiffness)
      .addScaled(this.velocity, -this.damping);
    const magnitude = this.force.length();
    if (magnitude > this.maxForce) {
      this.force.scale(this.maxForce / magnitude);
    }
    body.applyForce(this.force, this.anchorWorld);
  }

  private readPointer(event: PointerEvent): void {
    const rect = this.element.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  }
}
