import { benchmarkDemo } from "./benchmark.js";
import { demolitionDemo } from "./demolition.js";
import { jengaDemo } from "./jenga.js";
import { newtonCradleDemo } from "./newtonCradle.js";
import { ragdollDemo } from "./ragdoll.js";
import type { DemoDefinition } from "./types.js";
import { vehicleDemo } from "./vehicle.js";

/** Ordered list shown in the demo picker. */
export const demos: readonly DemoDefinition[] = [
  jengaDemo,
  newtonCradleDemo,
  ragdollDemo,
  vehicleDemo,
  demolitionDemo,
  benchmarkDemo,
];
