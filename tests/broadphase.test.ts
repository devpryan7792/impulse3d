import { describe, expect, test } from "vitest";
import { AABB } from "../src/collision/AABB.js";
import { AABBTree } from "../src/collision/broadphase/AABBTree.js";
import type { AABBTreeLeaf } from "../src/collision/broadphase/AABBTree.js";
import { Box } from "../src/collision/shapes/Box.js";
import { Capsule } from "../src/collision/shapes/Capsule.js";
import { ConvexPolytope } from "../src/collision/shapes/ConvexPolytope.js";
import { Plane } from "../src/collision/shapes/Plane.js";
import { Sphere } from "../src/collision/shapes/Sphere.js";
import type { Shape } from "../src/collision/shapes/Shape.js";
import { Transform } from "../src/math/Transform.js";
import { Vec3 } from "../src/math/Vec3.js";
import {
  expectVecClose,
  makeRng,
  randomQuat,
  randomUnitVec3,
  randomVec3,
} from "./testUtils.js";
import type { SeededRng } from "./testUtils.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Axis-aligned box from explicit corners. */
function makeBox(
  minX: number,
  minY: number,
  minZ: number,
  maxX: number,
  maxY: number,
  maxZ: number,
): AABB {
  return new AABB(new Vec3(minX, minY, minZ), new Vec3(maxX, maxY, maxZ));
}

/** Random non-empty box centered in `[-centerRange, centerRange]`. */
function randomBox(rng: SeededRng, centerRange: number, minHalf: number, maxHalf: number): AABB {
  const center = randomVec3(rng, -centerRange, centerRange);
  const half = new Vec3(
    rng.range(minHalf, maxHalf),
    rng.range(minHalf, maxHalf),
    rng.range(minHalf, maxHalf),
  );
  return new AABB().setFromCenterAndHalfExtents(center, half);
}

function pairKey(a: number, b: number): string {
  return a < b ? `${a}:${b}` : `${b}:${a}`;
}

/** Collects the tree's reported overlap pairs, asserting pair sanity on the fly. */
function collectTreePairs<T>(tree: AABBTree<T>): { pairs: Set<string>; count: number } {
  const pairs = new Set<string>();
  let count = 0;
  tree.queryOverlaps((a, b) => {
    count++;
    expect(a.id).not.toBe(b.id);
    // Live handles must agree with the traversal's view.
    expect(a.aabb.overlaps(b.aabb)).toBe(true);
    pairs.add(pairKey(a.id, b.id));
  });
  return { pairs, count };
}

/** O(N^2) reference pair set over the given boxes. */
function bruteForcePairs(boxes: Map<number, AABB>): Set<string> {
  const ids = [...boxes.keys()];
  const pairs = new Set<string>();
  for (let i = 0; i < ids.length; i++) {
    for (let j = i + 1; j < ids.length; j++) {
      const a = boxes.get(ids[i]) as AABB;
      const b = boxes.get(ids[j]) as AABB;
      if (a.overlaps(b)) {
        pairs.add(pairKey(ids[i], ids[j]));
      }
    }
  }
  return pairs;
}

/** One simulated body driven across frames. */
interface Body {
  readonly id: number;
  readonly shape: Shape;
  readonly unbounded: boolean;
  readonly transform: Transform;
  readonly velocity: Vec3;
  readonly angularVelocity: Vec3;
  readonly tight: AABB;
}

function makeBody(rng: SeededRng, id: number): Body {
  let shape: Shape;
  switch (id % 4) {
    case 0:
      shape = new Sphere(rng.range(0.2, 1.0));
      break;
    case 1:
      shape = new Box(
        new Vec3(rng.range(0.2, 0.9), rng.range(0.2, 0.9), rng.range(0.2, 0.9)),
      );
      break;
    case 2:
      shape = new Capsule(rng.range(0.15, 0.5), rng.range(0.2, 0.8));
      break;
    default:
      shape = new ConvexPolytope([
        new Vec3(0, 0, 0),
        new Vec3(0.6, 0, 0),
        new Vec3(0, 0.5, 0),
        new Vec3(0, 0, 0.7),
        new Vec3(-0.3, -0.3, -0.3),
      ]);
      break;
  }
  const transform = new Transform(randomVec3(rng, -20, 20), randomQuat(rng));
  return {
    id,
    shape,
    unbounded: false,
    transform,
    velocity: randomVec3(rng, -4, 4),
    angularVelocity: randomVec3(rng, -2, 2),
    tight: shape.computeAABB(transform),
  };
}

function makePlaneBody(rng: SeededRng, id: number): Body {
  const shape = new Plane(randomUnitVec3(rng), rng.range(-4, 4));
  const transform = new Transform();
  return {
    id,
    shape,
    unbounded: true,
    transform,
    velocity: new Vec3(),
    angularVelocity: new Vec3(),
    tight: shape.computeAABB(transform),
  };
}

/** Integrates one body by `dt` and recomputes its tight box. */
function advanceBody(body: Body, dt: number): void {
  if (body.unbounded) {
    return;
  }
  body.transform.position.addScaled(body.velocity, dt);
  body.transform.rotation.integrateAngularVelocity(body.angularVelocity, dt);
  body.shape.computeAABB(body.transform, body.tight);
}

/** A batch of randomized bodies mixing every shape type plus two planes. */
function makeBodies(rng: SeededRng, count: number): Body[] {
  const bodies: Body[] = [];
  for (let i = 0; i < count - 2; i++) {
    bodies.push(makeBody(rng, i));
  }
  bodies.push(makePlaneBody(rng, count - 2));
  bodies.push(makePlaneBody(rng, count - 1));
  return bodies;
}

interface RecordedHit {
  id: number;
  distance: number;
  point: Vec3;
}

function recordHits<T>(
  tree: AABBTree<T>,
  origin: Vec3,
  direction: Vec3,
  maxDistance: number = Infinity,
): RecordedHit[] {
  const hits: RecordedHit[] = [];
  tree.raycast(
    origin,
    direction,
    (hit) => hits.push({ id: hit.leaf.id, distance: hit.distance, point: hit.point }),
    maxDistance,
  );
  return hits;
}

/** Independent slab test used as the raycast oracle. Requires non-zero direction components. */
function referenceSlabDistance(box: AABB, origin: Vec3, direction: Vec3): number {
  const o = [origin.x, origin.y, origin.z];
  const d = [direction.x, direction.y, direction.z];
  const lo = [box.min.x, box.min.y, box.min.z];
  const hi = [box.max.x, box.max.y, box.max.z];
  let tMin = 0;
  let tMax = Infinity;
  for (let axis = 0; axis < 3; axis++) {
    const t1 = (lo[axis] - o[axis]) / d[axis];
    const t2 = (hi[axis] - o[axis]) / d[axis];
    tMin = Math.max(tMin, Math.min(t1, t2));
    tMax = Math.min(tMax, Math.max(t1, t2));
    if (tMin > tMax) {
      return -1;
    }
  }
  return tMin;
}

/** White-box view of the private tree state for invariant-corruption tests. */
interface PoolNode<T> {
  aabb: AABB;
  parent: number;
  left: number;
  right: number;
  height: number;
  leaf: AABBTreeLeaf<T> | null;
}

interface TreeInternals<T> {
  pool: PoolNode<T>[];
  freeList: number[];
  idToNode: Map<number, number>;
  rootIndex: number;
}

function internalsOf<T>(tree: AABBTree<T>): TreeInternals<T> {
  return tree as unknown as TreeInternals<T>;
}

function singleLeafTree(): AABBTree<number> {
  const tree = new AABBTree<number>();
  tree.insertLeaf(1, makeBox(0, 0, 0, 1, 1, 1), 1);
  return tree;
}

function twoLeafTree(): AABBTree<number> {
  const tree = new AABBTree<number>();
  tree.insertLeaf(1, makeBox(0, 0, 0, 1, 1, 1), 1);
  tree.insertLeaf(2, makeBox(5, 5, 5, 6, 6, 6), 2);
  return tree;
}

// ---------------------------------------------------------------------------
// Storage bookkeeping
// ---------------------------------------------------------------------------

describe("AABBTree storage", () => {
  test("empty tree reports neutral stats and tolerates every query", () => {
    const tree = new AABBTree<number>();
    tree.validate();
    expect(tree.size).toBe(0);
    expect(tree.nodeCount).toBe(0);
    expect(tree.stats()).toEqual({ leafCount: 0, nodeCount: 0, treeHeight: 0, treeUpdates: 0 });

    let leaves = 0;
    tree.forEachLeaf(() => {
      leaves++;
    });
    expect(leaves).toBe(0);

    let visits = 0;
    tree.queryAABB(makeBox(-1e9, -1e9, -1e9, 1e9, 1e9, 1e9), () => {
      visits++;
    });
    expect(visits).toBe(0);

    let pairs = 0;
    const reported = tree.queryOverlaps(() => {
      pairs++;
    });
    expect(pairs).toBe(0);
    expect(reported).toBe(0);

    expect(tree.removeLeaf(1)).toBe(false);
    expect(tree.getLeaf(1)).toBeUndefined();
    tree.clear();
    tree.validate();
  });

  test("insert, getLeaf, stats and removal keep the pool balanced", () => {
    const tree = new AABBTree<string>();
    const boxA = makeBox(0, 0, 0, 1, 1, 1);
    const boxB = makeBox(10, 10, 10, 11, 11, 11);
    const boxC = makeBox(-5, -5, -5, -4, -4, -4);
    const leafA = tree.insertLeaf(10, boxA, "a");
    tree.insertLeaf(20, boxB, "b");
    tree.insertLeaf(30, boxC, "c");

    expect(tree.size).toBe(3);
    // Two nodes per leaf minus one root: 2 * 3 - 1.
    expect(tree.nodeCount).toBe(5);
    expect(tree.stats()).toEqual({ leafCount: 3, nodeCount: 5, treeHeight: 2, treeUpdates: 0 });

    expect(tree.getLeaf(10)).toBe(leafA);
    expect(leafA.payload).toBe("a");
    expectVecClose(leafA.aabb.min, boxA.min, 0);
    expectVecClose(leafA.aabb.max, boxA.max, 0);
    expect(tree.getLeaf(999)).toBeUndefined();

    expect(tree.removeLeaf(999)).toBe(false);
    expect(tree.removeLeaf(20)).toBe(true);
    expect(tree.getLeaf(20)).toBeUndefined();
    expect(tree.size).toBe(2);
    expect(tree.nodeCount).toBe(3);
    tree.validate();

    expect(tree.removeLeaf(10)).toBe(true);
    expect(tree.removeLeaf(30)).toBe(true);
    expect(tree.size).toBe(0);
    expect(tree.nodeCount).toBe(0);
    tree.validate();
  });

  test("duplicate ids and NaN bounds are rejected", () => {
    const tree = new AABBTree<number>();
    tree.insertLeaf(1, makeBox(0, 0, 0, 1, 1, 1), 1);
    expect(() => tree.insertLeaf(1, makeBox(2, 2, 2, 3, 3, 3), 2)).toThrow(RangeError);
    expect(() => tree.insertLeaf(1, makeBox(2, 2, 2, 3, 3, 3), 2)).toThrow(/duplicate id 1/);

    const infinite = makeBox(-Infinity, -Infinity, -Infinity, Infinity, Infinity, Infinity);
    tree.insertLeaf(2, infinite, 2);
    expect(() => tree.insertLeaf(2, makeBox(2, 2, 2, 3, 3, 3), 3)).toThrow(/duplicate id 2/);

    const nanBox = new AABB(new Vec3(Number.NaN, 0, 0), new Vec3(1, 1, 1));
    expect(() => tree.insertLeaf(3, nanBox, 3)).toThrow(/NaN/);
    expect(tree.size).toBe(2);
  });

  test("getLeaf returns undefined when a mapped node no longer holds a leaf", () => {
    const tree = singleLeafTree();
    expect(tree.getLeaf(1)).toBeDefined();
    internalsOf(tree).pool[0].leaf = null;
    expect(tree.getLeaf(1)).toBeUndefined();
  });

  test("unbounded leaves live in a side registry", () => {
    const tree = new AABBTree<string>();
    const finite = makeBox(0, 0, 0, 1, 1, 1);
    tree.insertLeaf(1, finite, "finite");
    const planeLeaf = tree.insertLeaf(
      2,
      makeBox(-Infinity, -Infinity, -Infinity, Infinity, Infinity, Infinity),
      "plane",
    );
    const halfSpace = tree.insertLeaf(
      3,
      makeBox(0, -Infinity, -Infinity, 1, Infinity, Infinity),
      "half-space",
    );

    expect(tree.size).toBe(3);
    // Neither unbounded leaf allocates a tree node.
    expect(tree.nodeCount).toBe(1);
    expect(tree.stats().treeHeight).toBe(0);
    tree.validate();

    expect(tree.getLeaf(2)).toBe(planeLeaf);
    expect(tree.getLeaf(3)).toBe(halfSpace);

    const visited = new Set<number>();
    tree.forEachLeaf((leaf) => visited.add(leaf.id));
    expect(visited).toEqual(new Set([1, 2, 3]));

    const queried = new Set<number>();
    tree.queryAABB(makeBox(0.5, 0.5, 0.5, 0.75, 0.75, 0.75), (leaf) => queried.add(leaf.id));
    expect(queried).toEqual(new Set([1, 2, 3]));

    // An empty query box overlaps nothing, not even unbounded leaves.
    const emptyQuery = makeBox(1, 1, 1, -1, -1, -1);
    const emptyVisits = new Set<number>();
    tree.queryAABB(emptyQuery, (leaf) => emptyVisits.add(leaf.id));
    expect(emptyVisits.size).toBe(0);

    expect(tree.updateLeaf(2, finite, new Vec3(1, 0, 0), 0.1)).toBe(false);
    expect(tree.stats().treeUpdates).toBe(0);

    expect(tree.removeLeaf(2)).toBe(true);
    expect(tree.removeLeaf(3)).toBe(true);
    expect(tree.size).toBe(1);
    expect(tree.removeLeaf(1)).toBe(true);
    expect(tree.size).toBe(0);
  });

  test("queryOverlaps reports unbounded leaves exactly once per pair", () => {
    const tree = new AABBTree<number>();
    tree.insertLeaf(0, makeBox(0, 0, 0, 1, 1, 1), 0);
    tree.insertLeaf(1, makeBox(10, 10, 10, 11, 11, 11), 1);
    tree.insertLeaf(2, makeBox(-Infinity, -Infinity, -Infinity, Infinity, Infinity, Infinity), 2);
    tree.insertLeaf(3, makeBox(-Infinity, -Infinity, -Infinity, 5, Infinity, Infinity), 3);

    const { pairs, count } = collectTreePairs(tree);
    expect(count).toBe(pairs.size);
    // Leaf 3 is the half-space x <= 5, which does not reach leaf 1 at x >= 10.
    expect(pairs).toEqual(
      new Set([pairKey(0, 2), pairKey(1, 2), pairKey(0, 3), pairKey(2, 3)]),
    );
    expect(tree.queryOverlaps(() => {})).toBe(4);
  });

  test("clear drops every leaf and resets diagnostics", () => {
    const tree = new AABBTree<number>();
    tree.insertLeaf(1, makeBox(0, 0, 0, 1, 1, 1), 1);
    tree.insertLeaf(2, makeBox(0.5, 0.5, 0.5, 1.5, 1.5, 1.5), 2);
    tree.removeLeaf(1);
    tree.insertLeaf(3, makeBox(9, 9, 9, 10, 10, 10), 3);
    tree.updateLeaf(3, makeBox(50, 50, 50, 51, 51, 51), new Vec3(1, 0, 0), 1);
    expect(tree.stats().treeUpdates).toBe(1);

    tree.clear();
    expect(tree.size).toBe(0);
    expect(tree.nodeCount).toBe(0);
    expect(tree.stats()).toEqual({ leafCount: 0, nodeCount: 0, treeHeight: 0, treeUpdates: 0 });
    tree.validate();
  });
});

// ---------------------------------------------------------------------------
// Insertion structure
// ---------------------------------------------------------------------------

describe("SAH insertion invariants", () => {
  test("500 random inserts satisfy validation and the AVL height bound", () => {
    const rng = makeRng(0x5eed01);
    const tree = new AABBTree<number>();
    const count = 500;
    const boxes: AABB[] = [];
    for (let i = 0; i < count; i++) {
      // Mix moderate boxes with degenerate slivers and extreme aspect ratios.
      const kind = i % 8;
      if (kind === 0) {
        const p = randomVec3(rng, -50, 50);
        boxes.push(new AABB(p, p.clone()));
      } else if (kind === 1) {
        boxes.push(randomBox(rng, 40, 5, 25));
      } else {
        boxes.push(randomBox(rng, 40, 0.02, 3));
      }
    }

    const order = [...Array(count).keys()];
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rng.next() * (i + 1));
      const swap = order[i];
      order[i] = order[j];
      order[j] = swap;
    }
    for (const index of order) {
      tree.insertLeaf(index, boxes[index], index);
    }

    tree.validate();
    expect(tree.size).toBe(count);
    expect(tree.nodeCount).toBe(2 * count - 1);
    expect(tree.stats().treeHeight).toBeLessThanOrEqual(1.44 * Math.log2(count + 2));

    // Remove half and check the tree shrinks while staying balanced.
    for (let i = 0; i < count; i += 2) {
      expect(tree.removeLeaf(i)).toBe(true);
    }
    tree.validate();
    expect(tree.size).toBe(count / 2);
    expect(tree.nodeCount).toBe(count - 1);
    expect(tree.stats().treeHeight).toBeLessThanOrEqual(1.44 * Math.log2(count / 2 + 2));
  });

  test("repeated insert/remove cycles reuse pool nodes", () => {
    const rng = makeRng(0x5eed02);
    const tree = new AABBTree<number>();
    for (let i = 0; i < 64; i++) {
      tree.insertLeaf(i, randomBox(rng, 30, 0.1, 2), i);
    }
    const poolSize = internalsOf(tree).pool.length;
    expect(poolSize).toBe(2 * 64 - 1);

    for (let i = 0; i < 64; i++) {
      expect(tree.removeLeaf(i)).toBe(true);
    }
    expect(tree.nodeCount).toBe(0);
    expect(internalsOf(tree).pool.length).toBe(poolSize);
    expect(internalsOf(tree).freeList.length).toBe(poolSize);

    for (let i = 100; i < 164; i++) {
      tree.insertLeaf(i, randomBox(rng, 30, 0.1, 2), i);
    }
    expect(internalsOf(tree).pool.length).toBe(poolSize);
    expect(tree.nodeCount).toBe(2 * 64 - 1);
    tree.validate();
  });

  test("mixed insert/remove/update fuzz preserves validity and overlap completeness", () => {
    const rng = makeRng(0xf022);
    const tree = new AABBTree<number>();
    const envelopes = new Map<number, AABB>();
    let nextId = 0;

    for (let step = 0; step < 2000; step++) {
      const roll = rng.next();
      if (roll < 0.45 || envelopes.size === 0) {
        const leaf = tree.insertLeaf(nextId, randomBox(rng, 30, 0.05, 4), nextId);
        envelopes.set(nextId, leaf.aabb);
        nextId++;
      } else if (roll < 0.7) {
        const ids = [...envelopes.keys()];
        const id = ids[Math.floor(rng.next() * ids.length)];
        expect(tree.removeLeaf(id)).toBe(true);
        envelopes.delete(id);
      } else {
        const ids = [...envelopes.keys()];
        const id = ids[Math.floor(rng.next() * ids.length)];
        const stored = envelopes.get(id) as AABB;
        const center = stored.getCenter().add(randomVec3(rng, -3, 3));
        const tight = new AABB().setFromCenterAndHalfExtents(
          center,
          stored.getHalfExtents().scale(0.5),
        );
        tree.updateLeaf(id, tight, randomVec3(rng, -2, 2), 1 / 60);
        envelopes.set(id, tree.getLeaf(id)?.aabb as AABB);
      }
      if (step % 200 === 0) {
        tree.validate();
      }
    }

    tree.validate();
    const { pairs, count } = collectTreePairs(tree);
    expect(count).toBe(pairs.size);
    expect(pairs).toEqual(bruteForcePairs(envelopes));
    if (tree.size > 0) {
      expect(tree.nodeCount).toBe(2 * tree.size - 1);
    }
  });

  test("touching envelopes overlap while empty envelopes never do", () => {
    const tree = new AABBTree<number>();
    tree.insertLeaf(1, makeBox(0, 0, 0, 1, 1, 1), 1);
    tree.insertLeaf(2, makeBox(1, 0, 0, 2, 1, 1), 2);
    tree.insertLeaf(3, makeBox(2.001, 0, 0, 3, 1, 1), 3);

    expect(collectTreePairs(tree).pairs).toEqual(new Set([pairKey(1, 2)]));

    tree.removeLeaf(2);
    expect(collectTreePairs(tree).pairs.size).toBe(0);
  });

  test("coincident point leaves exercise the cheap-stop SAH branch", () => {
    const tree = new AABBTree<number>();
    const point = new Vec3(1, 2, 3);
    for (let i = 0; i < 4; i++) {
      tree.insertLeaf(i, new AABB(point.clone(), point.clone()), i);
    }
    tree.validate();
    expect(tree.nodeCount).toBe(2 * 4 - 1);
    expect(collectTreePairs(tree).pairs.size).toBe(6);
  });
});

// ---------------------------------------------------------------------------
// Moving shapes vs O(N^2) brute force
// ---------------------------------------------------------------------------

describe("queryOverlaps vs brute force", () => {
  test("500 randomized moving shapes: exact match on a freshly rebuilt tree", () => {
    const rng = makeRng(0x0f0f0f);
    const count = 500;
    const bodies = makeBodies(rng, count);
    const tree = new AABBTree<number>({ velocityScale: 0, margin: 0 });
    const dt = 1 / 60;

    for (let frame = 0; frame < 3; frame++) {
      for (const body of bodies) {
        advanceBody(body, dt);
      }
      tree.clear();
      for (const body of bodies) {
        const leaf = tree.insertLeaf(body.id, body.tight, body.id);
        expectVecClose(leaf.aabb.min, body.tight.min, 0);
        expectVecClose(leaf.aabb.max, body.tight.max, 0);
      }
      tree.validate();

      const tightBoxes = new Map<number, AABB>();
      for (const body of bodies) {
        tightBoxes.set(body.id, body.tight);
      }
      const { pairs, count: reported } = collectTreePairs(tree);
      expect(reported).toBe(pairs.size);
      expect(pairs).toEqual(bruteForcePairs(tightBoxes));
    }
  });

  test("500 randomized moving shapes: fattened tree is a superset of brute force", () => {
    const rng = makeRng(0x500fa7);
    const count = 500;
    const bodies = makeBodies(rng, count);
    const tree = new AABBTree<string>();
    const dt = 1 / 60;

    for (const body of bodies) {
      tree.insertLeaf(body.id, body.tight, `body-${body.id}`);
    }
    expect(tree.size).toBe(count);

    let escapes = 0;
    for (let frame = 0; frame < 20; frame++) {
      for (const body of bodies) {
        advanceBody(body, dt);
        if (tree.updateLeaf(body.id, body.tight, body.velocity, dt)) {
          escapes++;
        }
      }

      const tightBoxes = new Map<number, AABB>();
      for (const body of bodies) {
        tightBoxes.set(body.id, body.tight);
      }
      const { pairs, count: reported } = collectTreePairs(tree);
      expect(reported).toBe(pairs.size);
      for (const key of bruteForcePairs(tightBoxes)) {
        expect(pairs.has(key)).toBe(true);
      }
      if (frame % 5 === 0) {
        tree.validate();
      }
    }

    expect(escapes).toBeGreaterThan(0);
    tree.validate();
    expect(tree.stats().leafCount).toBe(count);
  });

  test("queryAABB matches brute force across stored envelopes", () => {
    const rng = makeRng(0x9a1b);
    const tree = new AABBTree<number>();
    const envelopes = new Map<number, AABB>();
    for (let i = 0; i < 300; i++) {
      const leaf = tree.insertLeaf(i, randomBox(rng, 25, 0.05, 3), i);
      envelopes.set(i, leaf.aabb);
    }
    const unboundedLeaf = tree.insertLeaf(
      999,
      makeBox(-Infinity, -Infinity, -Infinity, Infinity, Infinity, Infinity),
      999,
    );
    envelopes.set(999, unboundedLeaf.aabb);

    for (let query = 0; query < 60; query++) {
      const queryBox =
        query % 10 === 0 ? makeBox(1, 1, 1, -1, -1, -1) : randomBox(rng, 28, 0.05, 12);
      const visited = new Set<number>();
      let calls = 0;
      tree.queryAABB(queryBox, (leaf) => {
        calls++;
        visited.add(leaf.id);
      });
      expect(calls).toBe(visited.size);

      const expected = new Set<number>();
      for (const [id, box] of envelopes) {
        if (box.overlaps(queryBox)) {
          expected.add(id);
        }
      }
      expect(visited).toEqual(expected);
    }
  });
});

// ---------------------------------------------------------------------------
// Raycast
// ---------------------------------------------------------------------------

describe("raycast", () => {
  test("axis-parallel hits, slab swaps and maxDistance cuts", () => {
    const tree = new AABBTree<number>();
    tree.insertLeaf(7, makeBox(0, 0, 0, 1, 1, 1), 7);

    const hit = (origin: Vec3, direction: Vec3, maxDistance?: number): RecordedHit[] =>
      recordHits(tree, origin, direction, maxDistance);

    // Positive direction along +x, with the direction deliberately unnormalized.
    const straight = hit(new Vec3(-2, 0.5, 0.5), new Vec3(7, 0, 0));
    expect(straight.length).toBe(1);
    expect(straight[0].id).toBe(7);
    expect(straight[0].distance).toBeCloseTo(2, 12);
    expectVecClose(straight[0].point, new Vec3(0, 0.5, 0.5), 1e-12);

    // Axis-parallel misses on the y and z zero-slabs.
    expect(hit(new Vec3(-2, 2, 0.5), new Vec3(1, 0, 0))).toHaveLength(0);
    expect(hit(new Vec3(-2, 0.5, 2), new Vec3(1, 0, 0))).toHaveLength(0);
    // Axis-parallel miss on the x zero-slab.
    expect(hit(new Vec3(2, 0.5, 0.5), new Vec3(0, 1, 0))).toHaveLength(0);

    // Negative directions exercise the t1/t2 swap on each axis.
    expect(hit(new Vec3(3, 0.5, 0.5), new Vec3(-1, 0, 0))[0].distance).toBeCloseTo(2, 12);
    expect(hit(new Vec3(0.5, 3, 0.5), new Vec3(0, -1, 0))[0].distance).toBeCloseTo(2, 12);
    expect(hit(new Vec3(0.5, 0.5, 3), new Vec3(0, 0, -1))[0].distance).toBeCloseTo(2, 12);

    // Origin inside the box.
    const inside = hit(new Vec3(0.5, 0.5, 0.5), new Vec3(1, 0, 0));
    expect(inside[0].distance).toBe(0);
    expectVecClose(inside[0].point, new Vec3(0.5, 0.5, 0.5), 0);

    // maxDistance is inclusive and can cut an otherwise valid hit.
    expect(hit(new Vec3(-2, 0.5, 0.5), new Vec3(1, 0, 0), 1.5)).toHaveLength(0);
    expect(hit(new Vec3(-2, 0.5, 0.5), new Vec3(1, 0, 0), 2)[0].distance).toBeCloseTo(2, 12);

    // All three slabs clip the traversal interval on a diagonal.
    expect(hit(new Vec3(-2, -2, -2), new Vec3(1, 1, 1), 10)[0].distance).toBeCloseTo(
      Math.sqrt(12),
      12,
    );

    // Interval exits on the y and z slabs.
    expect(hit(new Vec3(-2, 100.5, 0.5), new Vec3(1, 1, 0))).toHaveLength(0);
    expect(hit(new Vec3(-2, 0.5, 100.5), new Vec3(1, 0, 1))).toHaveLength(0);
    // Exit inside the y slab when the box is behind the ray.
    expect(hit(new Vec3(0.5, 1.5, 0.5), new Vec3(0, 1, 0))).toHaveLength(0);
    // Exit inside the z slab via maxDistance.
    expect(hit(new Vec3(0.5, 0.5, -2), new Vec3(0, 0, 1), 1.5)).toHaveLength(0);
  });

  test("rejects zero and near-zero directions", () => {
    const tree = singleLeafTree();
    expect(() => tree.raycast(new Vec3(), new Vec3(0, 0, 0), () => {})).toThrow(RangeError);
    expect(() => tree.raycast(new Vec3(), new Vec3(0, 0, 0), () => {})).toThrow(
      /direction must be non-zero/,
    );
    expect(() => tree.raycast(new Vec3(), new Vec3(1e-13, 0, 0), () => {})).toThrow(RangeError);
    expect(() => tree.raycastClosest(new Vec3(), new Vec3(0, 0, 0))).toThrow(RangeError);
  });

  test("matches an independent slab reference on random boxes and rays", () => {
    const rng = makeRng(0x1a7e);
    const tree = new AABBTree<number>();
    const boxes = new Map<number, AABB>();
    for (let i = 0; i < 60; i++) {
      const box = randomBox(rng, 30, 0.1, 6);
      boxes.set(i, box);
      tree.insertLeaf(i, box, i);
    }

    for (let shot = 0; shot < 200; shot++) {
      const origin = randomVec3(rng, -60, 60);
      const raw = new Vec3(
        rng.range(0.15, 1) * (rng.next() < 0.5 ? -1 : 1),
        rng.range(0.15, 1) * (rng.next() < 0.5 ? -1 : 1),
        rng.range(0.15, 1) * (rng.next() < 0.5 ? -1 : 1),
      );
      const direction = raw.normalize();

      const hits = recordHits(tree, origin, direction);
      const expected = new Map<number, number>();
      for (const [id, box] of boxes) {
        const distance = referenceSlabDistance(box, origin, direction);
        if (distance >= 0) {
          expected.set(id, distance);
        }
      }

      expect(new Set(hits.map((h) => h.id))).toEqual(new Set(expected.keys()));
      for (const h of hits) {
        expect(h.distance).toBeCloseTo(expected.get(h.id) as number, 9);
        expectVecClose(
          h.point,
          origin.clone().addScaled(direction, h.distance),
          1e-9,
          `raycast point for leaf ${h.id}`,
        );
      }
    }
  });

  test("raycastClosest picks the nearest hit and respects maxDistance", () => {
    const tree = new AABBTree<number>();
    tree.insertLeaf(1, makeBox(5, 0, 0, 6, 1, 1), 1);
    tree.insertLeaf(2, makeBox(0, 0, 0, 1, 1, 1), 2);

    const closest = tree.raycastClosest(new Vec3(-2, 0.5, 0.5), new Vec3(1, 0, 0));
    expect(closest?.leaf.id).toBe(2);
    expect(closest?.distance).toBeCloseTo(2, 12);

    const clipped = tree.raycastClosest(new Vec3(-2, 0.5, 0.5), new Vec3(1, 0, 0), 3);
    expect(clipped?.leaf.id).toBe(2);

    expect(tree.raycastClosest(new Vec3(-2, 0.5, 0.5), new Vec3(1, 0, 0), 1.5)).toBeNull();
    expect(new AABBTree<number>().raycastClosest(new Vec3(), new Vec3(1, 0, 0))).toBeNull();
  });

  test("unbounded leaves are always hit, one-sided boxes can be missed", () => {
    const tree = new AABBTree<number>();
    tree.insertLeaf(1, makeBox(-Infinity, -Infinity, -Infinity, Infinity, Infinity, Infinity), 1);
    tree.insertLeaf(2, makeBox(-Infinity, -Infinity, -Infinity, -1, Infinity, Infinity), 2);

    const forward = recordHits(tree, new Vec3(), new Vec3(1, 0, 0));
    expect(forward.map((h) => h.id)).toEqual([1]);
    expect(forward[0].distance).toBe(0);

    const backward = recordHits(tree, new Vec3(), new Vec3(-1, 0, 0));
    expect(backward.map((h) => h.id).sort()).toEqual([1, 2]);
    expect(backward.find((h) => h.id === 2)?.distance).toBeCloseTo(1, 12);
  });

  test("root-level misses and internal traversal agree with direct leaf tests", () => {
    const tree = new AABBTree<number>();
    for (let i = 0; i < 8; i++) {
      tree.insertLeaf(i, makeBox(i * 3, 0, 0, i * 3 + 1, 1, 1), i);
    }
    const miss = recordHits(tree, new Vec3(0, 100, 0), new Vec3(1, 0, 0));
    expect(miss).toHaveLength(0);

    const hit = recordHits(tree, new Vec3(-1, 0.5, 0.5), new Vec3(1, 0, 0));
    expect(hit.map((h) => h.id).sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    for (const recorded of hit) {
      expect(recorded.distance).toBeCloseTo(recorded.id * 3 + 1, 12);
    }
  });
});

// ---------------------------------------------------------------------------
// updateLeaf
// ---------------------------------------------------------------------------

describe("updateLeaf", () => {
  test("contained motion is free, escaping motion refits the envelope", () => {
    const tree = new AABBTree<number>();
    const initial = makeBox(0, 0, 0, 1, 1, 1);
    tree.insertLeaf(1, initial.clone().expand(0.05), 1);

    expect(tree.updateLeaf(1, initial, new Vec3(0.1, 0, 0), 1 / 60)).toBe(false);
    expect(tree.stats().treeUpdates).toBe(0);

    const escaped = makeBox(5, 5, 5, 6, 6, 6);
    const velocity = new Vec3(3, -2, 0);
    const dt = 1 / 60;
    expect(tree.updateLeaf(1, escaped, velocity, dt)).toBe(true);
    expect(tree.stats().treeUpdates).toBe(1);

    // The public handle exposes the live fattened envelope.
    const leaf = tree.getLeaf(1);
    expect(leaf).toBeDefined();
    const expected = AABB.fatten(new AABB(), escaped, velocity, dt, 1, 0.05);
    expectVecClose((leaf as AABBTreeLeaf<number>).aabb.min, expected.min, 0);
    expectVecClose((leaf as AABBTreeLeaf<number>).aabb.max, expected.max, 0);
    tree.validate();

    // The envelope now contains the tight box, so the next update is free again.
    expect(tree.updateLeaf(1, escaped, velocity, dt)).toBe(false);
    expect(tree.stats().treeUpdates).toBe(1);

    expect(() => tree.updateLeaf(42, escaped, velocity, dt)).toThrow(RangeError);
    expect(() => tree.updateLeaf(42, escaped, velocity, dt)).toThrow(/unknown id 42/);
  });

  test("velocity-free escapes still grow by the static margin", () => {
    const tree = new AABBTree<number>();
    tree.insertLeaf(1, makeBox(0, 0, 0, 1, 1, 1), 1);
    const moved = makeBox(0.2, 0, 0, 1.2, 1, 1);
    expect(tree.updateLeaf(1, moved, new Vec3(0, 0, 0), 1 / 60)).toBe(true);
    const envelope = tree.getLeaf(1)?.aabb as AABB;
    expectVecClose(envelope.min, new Vec3(0.15, -0.05, -0.05), 1e-12);
    expectVecClose(envelope.max, new Vec3(1.25, 1.05, 1.05), 1e-12);
  });

  test("jitter within the static margin never touches the tree", () => {
    const rng = makeRng(0x1177e2);
    const count = 120;
    const tree = new AABBTree<string>();
    const bodies: Body[] = [];
    const dt = 0.01;
    const base = new Map<number, Vec3>();

    for (let i = 0; i < count; i++) {
      const body = makeBody(rng, i);
      // Collapse each body onto a modest starting position so the batch is dense.
      body.transform.position.set(rng.range(-8, 8), rng.range(-8, 8), rng.range(-8, 8));
      body.shape.computeAABB(body.transform, body.tight);
      base.set(body.id, body.transform.position.clone());
      bodies.push(body);
      tree.insertLeaf(body.id, body.tight.clone().expand(0.05), `body-${body.id}`);
    }

    for (let frame = 0; frame < 60; frame++) {
      for (const body of bodies) {
        const jitter = randomVec3(rng, -0.04, 0.04);
        body.transform.position.copy(base.get(body.id) as Vec3).add(jitter);
        body.velocity.copy(jitter).scale(1 / dt);
        body.shape.computeAABB(body.transform, body.tight);
        expect(tree.updateLeaf(body.id, body.tight, body.velocity, dt)).toBe(false);
      }
    }

    expect(tree.stats().treeUpdates).toBe(0);
    tree.validate();

    const tightBoxes = new Map<number, AABB>();
    for (const body of bodies) {
      tightBoxes.set(body.id, body.tight);
    }
    const { pairs } = collectTreePairs(tree);
    for (const key of bruteForcePairs(tightBoxes)) {
      expect(pairs.has(key)).toBe(true);
    }
  });

  test("options default to alpha = 1 and beta = 0.05", () => {
    const defaults = new AABBTree<number>();
    const explicitEmpty = new AABBTree<number>({});
    for (const tree of [defaults, explicitEmpty]) {
      tree.insertLeaf(1, makeBox(0, 0, 0, 1, 1, 1), 1);
      const tight = makeBox(2, 2, 2, 3, 3, 3);
      const velocity = new Vec3(1, 1, 1);
      const dt = 1 / 30;
      expect(tree.updateLeaf(1, tight, velocity, dt)).toBe(true);
      const expected = AABB.fatten(new AABB(), tight, velocity, dt, 1, 0.05);
      const leaf = tree.getLeaf(1) as AABBTreeLeaf<number>;
      expectVecClose(leaf.aabb.min, expected.min, 0);
      expectVecClose(leaf.aabb.max, expected.max, 0);
    }

    const configured = new AABBTree<number>({ velocityScale: 2, margin: 0 });
    configured.insertLeaf(1, makeBox(0, 0, 0, 1, 1, 1), 1);
    const tight = makeBox(2, 0, 0, 3, 1, 1);
    const velocity = new Vec3(1, 0, 0);
    const dt = 1 / 30;
    expect(configured.updateLeaf(1, tight, velocity, dt)).toBe(true);
    const expected = AABB.fatten(new AABB(), tight, velocity, dt, 2, 0);
    const leaf = configured.getLeaf(1) as AABBTreeLeaf<number>;
    expectVecClose(leaf.aabb.min, expected.min, 0);
    expectVecClose(leaf.aabb.max, expected.max, 0);
  });
});

// ---------------------------------------------------------------------------
// validate() corruption detection
// ---------------------------------------------------------------------------

describe("validate corruption detection", () => {
  test("detects leaves mapped without a root", () => {
    const tree = singleLeafTree();
    internalsOf(tree).rootIndex = -1;
    expect(() => tree.validate()).toThrow(/without a root/);
  });

  test("detects a root with a parent", () => {
    const tree = twoLeafTree();
    const state = internalsOf(tree);
    state.pool[state.rootIndex].parent = state.pool[state.rootIndex].left;
    expect(() => tree.validate()).toThrow(/root has a parent/);
  });

  test("detects a node reachable twice", () => {
    const tree = twoLeafTree();
    const state = internalsOf(tree);
    const root = state.pool[state.rootIndex];
    root.right = root.left;
    expect(() => tree.validate()).toThrow(/cycle/);
  });

  test("detects a leaf with a non-zero height", () => {
    const tree = singleLeafTree();
    internalsOf(tree).pool[0].height = 1;
    expect(() => tree.validate()).toThrow(/has height 1/);
  });

  test("detects a leaf with children", () => {
    const tree = singleLeafTree();
    internalsOf(tree).pool[0].left = 0;
    expect(() => tree.validate()).toThrow(/has children/);
  });

  test("detects an unmapped leaf id", () => {
    const tree = singleLeafTree();
    internalsOf(tree).idToNode.delete(1);
    expect(() => tree.validate()).toThrow(/not mapped to its node/);
  });

  test("detects an internal node missing a child", () => {
    const tree = twoLeafTree();
    internalsOf(tree).pool[internalsOf(tree).rootIndex].left = -1;
    expect(() => tree.validate()).toThrow(/missing a child/);
  });

  test("detects a stale parent link", () => {
    const tree = twoLeafTree();
    const state = internalsOf(tree);
    const root = state.pool[state.rootIndex];
    state.pool[root.left].parent = root.right;
    expect(() => tree.validate()).toThrow(/stale parent link/);
  });

  test("detects an inconsistent height label", () => {
    const tree = twoLeafTree();
    internalsOf(tree).pool[internalsOf(tree).rootIndex].height = 9;
    expect(() => tree.validate()).toThrow(/inconsistent height/);
  });

  test("detects an AVL balance violation", () => {
    const tree = twoLeafTree();
    const state = internalsOf(tree);
    const root = state.pool[state.rootIndex];
    state.pool[root.left].height = 2;
    root.height = 3;
    expect(() => tree.validate()).toThrow(/AVL balance/);
  });

  test("detects node bounds that do not contain their children", () => {
    const tree = twoLeafTree();
    const state = internalsOf(tree);
    state.pool[state.rootIndex].aabb.set(new Vec3(0, 0, 0), new Vec3(0.5, 0.5, 0.5));
    expect(() => tree.validate()).toThrow(/bounds do not contain/);
  });

  test("detects a registry larger than the reachable leaf set", () => {
    const tree = singleLeafTree();
    internalsOf(tree).idToNode.set(2, 0);
    expect(() => tree.validate()).toThrow(/visited 1 leaves but 2 are registered/);
  });
});
