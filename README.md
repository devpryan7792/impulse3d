# Impulse3D

[![CI](https://github.com/devpryan7792/impulse3d/actions/workflows/ci.yml/badge.svg)](https://github.com/devpryan7792/impulse3d/actions/workflows/ci.yml)

A zero-dependency 3D rigid-body physics engine in strict TypeScript, with an
interactive Three.js playground.

Impulse3D implements the full simulation stack from scratch: vector/quaternion
math, convex shapes, a dynamic AABB-tree broadphase, analytic and GJK/EPA
narrowphase collision detection, persistent manifolds, and a sequential-impulse
solver with split-impulse position correction. Nothing in `src/` imports a
runtime dependency — Three.js is only used by the `examples/` app.

## Features

- **Math** — `Vec3`, `Quat`, `Mat3`, `Mat4`, `Transform` with alias-safe static
  operators and allocation-free instance methods.
- **Shapes** — sphere, box, capsule, plane, convex polytope, each with support
  mappings and local inertia tensors.
- **Broadphase** — dynamic AABB tree with velocity-expanded bounds, stable
  insert/update/remove and overlap queries.
- **Narrowphase** — analytic sphere/box/capsule/plane paths, clipped box-box
  manifolds, and GJK + EPA for general convex pairs.
- **Contact persistence** — contact cache with feature IDs and warm-started
  normal impulses.
- **Solver** — sequential impulses over a normal + two tangent rows per point,
  Coulomb friction, restitution, Baumgarte-free split-impulse position
  correction, and configurable substeps/iterations.
- **Dynamics** — static/dynamic bodies, force and torque accumulators, damping,
  island detection and automatic sleeping, and `BallSocket`, `Hinge` and
  `Distance` joints.
- **Visualizer** — Three.js mesh sync, shadow support, and a merged wireframe
  overlay for AABBs and contact normals.
- **Playground** — mouse raycast spring grabber and six interactive scenes,
  including a 1000-body benchmark.

## Quick start

Requires Node.js 22+.

```bash
git clone https://github.com/devpryan7792/impulse3d.git
cd impulse3d
npm install

npm run build          # compile the library to dist/
npm test               # run the test suite
npm run demo           # open the playground at http://localhost:5173
```

The package is not published to npm. To use it locally, build it and depend on
the repository (or import from `src/` through your bundler):

```ts
import { BodyType, Box, Plane, RigidBody, Vec3, World } from "impulse3d";

const world = new World();

world.addBody(new RigidBody({ shape: new Plane(), type: BodyType.Static }));

const box = world.addBody(
  new RigidBody({
    shape: new Box(new Vec3(0.5, 0.5, 0.5)),
    mass: 1,
    position: new Vec3(0, 3, 0),
  }),
);

for (let tick = 0; tick < 600; tick++) {
  world.step(1 / 60);
}

console.log(box.position.toString(), box.isSleeping);
```

## Demos

| Scene | What it shows |
| --- | --- |
| **Jenga 100** | 100 alternating blocks in a 10 m tower, pre-settled and asleep; grab a block to wake the island. |
| **Newton's Cradle** | Five equal spheres on distance joints with restitution 1. |
| **Ragdoll** | An 11-body humanoid with ten ball-socket joints tumbling down stairs. |
| **Raycast Vehicle** | A 140 kg chassis on four analytic spring/damper struts with drive, grip, braking and steering forces. Drive with `W`/`A`/`S`/`D`; `Space` brakes. |
| **Demolition** | An 80-brick running-bond wall versus a 220 kg wrecking ball. |
| **1000-Body Benchmark** | 1000 dynamic spheres in a walled arena, tuned for throughput; the HUD reports live fps and step time. |

Panels/keys: `P` pause, `R` reset, `B` debug overlay (AABBs + contact normals),
`H` hide the HUD. Drag any dynamic body to pull it with the spring grabber.
Scenes are deep-linkable: `?demo=jenga`, `?demo=cradle`, `?demo=ragdoll`,
`?demo=vehicle`, `?demo=demolition`, `?demo=benchmark`.

## Project layout

```text
src/
  math/         Vec3, Quat, Mat3, Mat4, Transform
  collision/    shapes, AABB tree, narrowphase, manifolds
  dynamics/     RigidBody, ConstraintRow, ContactSolver, Joint, World
  index.ts      public API
tests/          Vitest suites (math, shapes, collision, dynamics)
examples/       Vite + Three.js playground (dev-only dependency)
```

## Testing

```bash
npm run typecheck           # strict TypeScript, no emit
npm run typecheck:examples  # demo app typecheck
npm test                    # 205 tests
npm run test:coverage       # enforces 100% statements/branches/functions/lines
npm run demo:smoke          # headless physics checks for every demo
npm run demo:build          # production demo bundle
```

The solver's deliberate deviations from a naive sequential-impulse
implementation are documented in the source: split-impulse penetration
correction instead of velocity Baumgarte, friction impulses are not
warm-started, and joint-connected pairs skip contacts by default.

## Performance

The default world uses 8 substeps and 10 solver iterations for stack stability.
The benchmark scene trades margin for throughput (1 substep, 4 iterations,
shadows off) so its HUD can measure the real 60 Hz budget on the host machine.
Physics cost is reported live; it is not a synthetic claim.

## License

MIT — see [LICENSE](LICENSE).
