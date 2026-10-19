import * as THREE from "three";
import { Vec3 } from "impulse3d";
import type { RigidBody } from "impulse3d";
import { addBox, addGround } from "./helpers.js";
import type { DemoContext, DemoDefinition } from "./types.js";

const WHEEL_RADIUS = 0.3;
const WHEEL_HALF_WIDTH = 0.11;
const SUSPENSION_REST = 0.36;
const SPRING_STIFFNESS = 9000;
const SPRING_DAMPING = 900;
const DRIVE_FORCE = 260;
const ROLLING_RESISTANCE = 12;
const LATERAL_STIFFNESS = 1200;
const GRIP = 1.2;
const YAW_TORQUE = 1000;
const YAW_DAMPING = 700;

interface Wheel {
  readonly local: Vec3;
  readonly mesh: THREE.Mesh;
}

interface VehicleState {
  readonly chassis: RigidBody;
  readonly wheels: Wheel[];
  spin: number;
}

const pressed = new Set<string>();
let state: VehicleState | null = null;

function anyPressed(codes: readonly string[]): boolean {
  return codes.some((code) => pressed.has(code));
}

function createWheelMeshes(context: DemoContext): THREE.Mesh[] {
  const geometry = new THREE.CylinderGeometry(WHEEL_RADIUS, WHEEL_RADIUS, WHEEL_HALF_WIDTH * 2, 22);
  // Cylinder axis is local +Y; rotate it to +Z so the wheel rolls along X.
  geometry.rotateX(Math.PI / 2);
  const material = new THREE.MeshStandardMaterial({ color: 0x2e3440, roughness: 0.9 });
  const meshes: THREE.Mesh[] = [];
  for (let i = 0; i < 4; i++) {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    context.group.add(mesh);
    meshes.push(mesh);
  }
  return meshes;
}

function build(context: DemoContext): void {
  pressed.clear();
  const { world } = context;
  addGround(world, 0.9);

  const chassis = addBox(world, new Vec3(0.9, 0.18, 0.45), new Vec3(0, 0.72, 0), {
    mass: 140,
    friction: 0.2,
    linearDamping: 0.06,
    angularDamping: 0.6,
    color: 0xbf616a,
  });

  const meshes = createWheelMeshes(context);
  const wheels: Wheel[] = [];
  let index = 0;
  for (const x of [0.62, -0.62]) {
    for (const z of [-0.42, 0.42]) {
      wheels.push({
        local: new Vec3(x, -0.08, z),
        mesh: meshes[index++],
      });
    }
  }
  state = { chassis, wheels, spin: 0 };
}

const scratchAnchor = new Vec3();
const scratchVelocity = new Vec3();
const scratchForward = new Vec3();
const scratchRight = new Vec3();
const scratchForce = new Vec3();
const LOCAL_FORWARD = new Vec3(1, 0, 0);
const LOCAL_RIGHT = new Vec3(0, 0, 1);
const Z_AXIS = new THREE.Vector3(0, 0, 1);

/**
 * Raycast suspension vehicle.
 *
 * Each wheel casts an analytic ray straight down to the ground plane. The
 * spring/damper pair produces the normal force; longitudinal drive, rolling
 * resistance and lateral grip are applied at the contact patch. Yaw torque
 * provides steering. Wheels are visual-only: this demo is a vehicle *model*
 * built purely from the engine's body, force and torque primitives.
 */
function update(dt: number, _context: DemoContext): void {
  if (state === null) {
    return;
  }
  const { chassis, wheels } = state;

  const steer = (anyPressed(["KeyA", "ArrowLeft"]) ? 1 : 0) -
    (anyPressed(["KeyD", "ArrowRight"]) ? 1 : 0);
  const throttle = (anyPressed(["KeyW", "ArrowUp"]) ? 1 : 0) -
    (anyPressed(["KeyS", "ArrowDown"]) ? 1 : 0);
  const braking = anyPressed(["Space"]);

  chassis.transform.transformDirection(scratchForward, LOCAL_FORWARD);
  chassis.transform.transformDirection(scratchRight, LOCAL_RIGHT);
  const forwardSpeed =
    chassis.linearVelocity.x * scratchForward.x +
    chassis.linearVelocity.y * scratchForward.y +
    chassis.linearVelocity.z * scratchForward.z;

  state.spin += (forwardSpeed / WHEEL_RADIUS) * dt;
  const spinRotation = new THREE.Quaternion().setFromAxisAngle(Z_AXIS, state.spin);

  for (const wheel of wheels) {
    chassis.transform.transformPoint(scratchAnchor, wheel.local);
    // Analytic raycast against the demo's flat ground plane at y = 0.
    const strutLength = scratchAnchor.y - WHEEL_RADIUS;
    const compression = Math.max(0, SUSPENSION_REST - strutLength);

    // Suspension normal force along world +Y.
    chassis.getVelocityAtPoint(scratchAnchor, scratchVelocity);
    const normalForce = Math.max(
      0,
      compression * SPRING_STIFFNESS - scratchVelocity.y * SPRING_DAMPING,
    );

    if (normalForce > 0) {
      scratchForce.set(0, normalForce, 0);
      chassis.applyForce(scratchForce, scratchAnchor);

      // Lateral grip, clamped by the friction cone.
      const lateralSpeed =
        scratchVelocity.x * scratchRight.x +
        scratchVelocity.y * scratchRight.y +
        scratchVelocity.z * scratchRight.z;
      const lateralLimit = GRIP * normalForce;
      const lateralForce = Math.max(
        -lateralLimit,
        Math.min(lateralLimit, -lateralSpeed * LATERAL_STIFFNESS),
      );
      scratchForce.copy(scratchRight).scale(lateralForce);
      chassis.applyForce(scratchForce, scratchAnchor);

      // Drive / brake / rolling resistance along the chassis forward axis.
      let longitudinalForce = 0;
      if (braking) {
        longitudinalForce = -forwardSpeed * 220;
      } else if (throttle !== 0) {
        longitudinalForce = throttle * DRIVE_FORCE;
      }
      longitudinalForce -= forwardSpeed * ROLLING_RESISTANCE;
      const longitudinalLimit = GRIP * normalForce;
      longitudinalForce = Math.max(
        -longitudinalLimit,
        Math.min(longitudinalLimit, longitudinalForce),
      );
      scratchForce.copy(scratchForward).scale(longitudinalForce);
      chassis.applyForce(scratchForce, scratchAnchor);
    }

    // Place the visual wheel at the end of the strut.
    const travel = Math.min(strutLength, SUSPENSION_REST);
    wheel.mesh.position.copy(scratchAnchor);
    wheel.mesh.position.y -= travel;
    wheel.mesh.quaternion.set(
      chassis.rotation.x,
      chassis.rotation.y,
      chassis.rotation.z,
      chassis.rotation.w,
    );
    wheel.mesh.quaternion.multiply(spinRotation);
  }

  // Steering: yaw torque, scaled by speed, plus a mild self-aligning torque.
  const speedFactor = Math.min(1, Math.abs(forwardSpeed) / 3);
  chassis.applyTorque(new Vec3(0, steer * YAW_TORQUE * speedFactor, 0));
  chassis.applyTorque(new Vec3(0, -chassis.angularVelocity.y * YAW_DAMPING, 0));
}

export const vehicleDemo: DemoDefinition = {
  id: "vehicle",
  name: "Raycast Vehicle",
  description:
    "A 140 kg chassis on four raycast spring/damper struts. Drive with W/A/S/D or the arrow keys; Space brakes. Normal, drive, rolling and lateral grip forces all act at the suspension anchor.",
  camera: {
    position: [9, 5.5, 9],
    target: [0, 0.8, 0],
  },
  build,
  update,
  onKey(code: string, down: boolean): void {
    if (down) {
      pressed.add(code);
    } else {
      pressed.delete(code);
    }
  },
};
