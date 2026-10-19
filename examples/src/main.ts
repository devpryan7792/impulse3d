import { PhysicsScene } from "./PhysicsScene.js";
import { demos } from "./demos/index.js";
import type { DemoDefinition } from "./demos/types.js";

function required<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (element === null) {
    throw new Error(`Missing required element: ${selector}`);
  }
  return element;
}

const canvas = required<HTMLCanvasElement>("#scene");
const hud = required<HTMLElement>("#hud");
const nav = required<HTMLElement>("#demos");
const description = required<HTMLElement>("#description");
const fpsLabel = required<HTMLElement>("#stat-fps");
const bodiesLabel = required<HTMLElement>("#stat-bodies");
const contactsLabel = required<HTMLElement>("#stat-contacts");
const stepLabel = required<HTMLElement>("#stat-step");

const scene = new PhysicsScene(canvas, demos[0], {
  onStats(stats) {
    fpsLabel.textContent = stats.fps.toFixed(0);
    bodiesLabel.textContent = stats.awake > 0 ? `${stats.awake}/${stats.bodies}` : String(stats.bodies);
    contactsLabel.textContent = String(stats.contacts);
    stepLabel.textContent = `${stats.stepMs.toFixed(2)} ms`;
  },
});

const buttons: HTMLButtonElement[] = [];
for (const demo of demos) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = demo.name;
  button.addEventListener("click", () => select(demo));
  nav.append(button);
  buttons.push(button);
}

function select(demo: DemoDefinition): void {
  scene.loadDemo(demo);
  description.textContent = demo.description;
  for (let i = 0; i < demos.length; i++) {
    buttons[i].setAttribute("aria-pressed", String(demos[i] === demo));
  }
  const url = new URL(window.location.href);
  url.searchParams.set("demo", demo.id);
  window.history.replaceState(null, "", url);
}

window.addEventListener("keydown", (event) => {
  if (event.code === "KeyH") {
    hud.hidden = !hud.hidden;
  }
});

const requested = new URLSearchParams(window.location.search).get("demo");
select(demos.find((demo) => demo.id === requested) ?? demos[0]);
scene.start();
