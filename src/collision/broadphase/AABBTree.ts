import { EPSILON } from "../../math/MathUtils.js";
import { Vec3 } from "../../math/Vec3.js";
import { AABB } from "../AABB.js";

/**
 * Public handle for a leaf stored in the tree. The `aabb` reference is the
 * *live* fattened broadphase box owned by the tree: it is mutated in place by
 * {@link AABBTree.updateLeaf}.
 */
export interface AABBTreeLeaf<T> {
  readonly id: number;
  readonly payload: T;
  readonly aabb: AABB;
}

/** Tuning parameters of the velocity-fattened envelope (Section 4.1). */
export interface AABBTreeOptions {
  /**
   * Velocity multiplier `α` of the fattening term `v · Δt · α`; `1` projects
   * exactly the distance travelled during the step.
   */
  velocityScale?: number;
  /**
   * Static margin `β` (in world units) applied symmetrically on every side so
   * that slow or stationary bodies never trigger a tree update. Defaults to
   * `0.05` m per the specification.
   */
  margin?: number;
}

/** Diagnostics snapshot used by benchmarks and invariant tests. */
export interface AABBTreeStats {
  /** Number of stored leaves, including unbounded ones. */
  leafCount: number;
  /** Number of currently allocated tree nodes (internal nodes plus finite leaves). */
  nodeCount: number;
  /** Height of the finite tree; `0` for an empty tree or a lone leaf. */
  treeHeight: number;
  /** Number of `updateLeaf` calls that had to refit and reinsert a node. */
  treeUpdates: number;
}

/** A ray/leaf intersection reported by the tree raycast queries. */
export interface RaycastHit<T> {
  readonly leaf: AABBTreeLeaf<T>;
  /** Distance along the query direction from the ray origin (direction is normalized). */
  readonly distance: number;
  /** World-space intersection point with the *fattened* leaf bounds. */
  readonly point: Vec3;
}

/** Internal pool node: either an internal branch or a leaf, addressed by index. */
interface TreeNode<T> {
  aabb: AABB;
  parent: number;
  left: number;
  right: number;
  /** `0` for leaves, `1 + max(child heights)` for internal nodes. */
  height: number;
  /** Non-null exactly for leaves. */
  leaf: AABBTreeLeaf<T> | null;
}

/** Scratch box for SAH cost evaluation (single-threaded use). */
const scratchAABB = new AABB();

/** `AABB.merge` into the scratch box, returning its surface area. */
function unionSurfaceArea(a: AABB, b: AABB): number {
  return AABB.merge(scratchAABB, a, b).surfaceArea();
}

/**
 * Incremental, self-balancing binary AABB tree with a Surface Area Heuristic
 * (Section 4 of the specification) and AVL rotations.
 *
 * ## Insertion cost metric
 * Descending from a node `C` with children `L`/`R`, the greedy SAH comparison
 * implemented here is exactly the specification's recurrence evaluated one
 * level deep:
 *
 * ```
 * cost_here  = Area(C ∪ N)
 * cost_left  = Area(C.left  ∪ N) - Area(C.left)   (+ recursive cost, implicit in the descent)
 * cost_right = Area(C.right ∪ N) - Area(C.right)  (+ recursive cost, implicit in the descent)
 * ```
 *
 * The descent stops when the new parent is cheaper than either child branch,
 * otherwise it continues into the cheaper subtree, and the new leaf is spliced
 * in as a sibling. Splice points are rebalanced with AVL rotations so the tree
 * height stays `≤ 1.44 log2(n + 2)` — see {@link AABBTree.validate}.
 *
 * ## Velocity fattening
 * {@link AABBTree.updateLeaf} only refits the tree when the tight shape box
 * escapes the stored fat envelope
 * `AABB_tight + v · Δt · α + β` (Section 4.1), which keeps moving bodies from
 * forcing rotations every frame.
 *
 * ## Unbounded leaves
 * Bounds with `±Infinity` components (infinite planes) are kept in a separate
 * registry: they overlap every query by definition and never participate in
 * SAH cost math (which would be `NaN`/`Infinity`). Every query still reports
 * them, so broadphase output remains a superset of the brute-force result.
 */
export class AABBTree<T> {
  private readonly pool: TreeNode<T>[] = [];
  private readonly freeList: number[] = [];
  private readonly idToNode = new Map<number, number>();
  private readonly unbounded = new Map<number, AABBTreeLeaf<T>>();
  private rootIndex = -1;
  private readonly velocityScale: number;
  private readonly margin: number;
  private updateCount = 0;

  constructor(options: AABBTreeOptions = {}) {
    this.velocityScale = options.velocityScale ?? 1;
    this.margin = options.margin ?? 0.05;
  }

  /** Number of stored leaves (finite and unbounded). */
  get size(): number {
    return this.idToNode.size + this.unbounded.size;
  }

  /** Number of allocated nodes; useful for pool-reuse assertions. */
  get nodeCount(): number {
    return this.pool.length - this.freeList.length;
  }

  /** Diagnostics snapshot. */
  stats(): AABBTreeStats {
    return {
      leafCount: this.size,
      nodeCount: this.nodeCount,
      treeHeight: this.rootIndex === -1 ? 0 : this.pool[this.rootIndex].height,
      treeUpdates: this.updateCount,
    };
  }

  /**
   * Inserts a leaf with a unique `id` and its fattened broadphase bounds.
   * Infinite bounds register the leaf as unbounded instead of descending into
   * the tree.
   *
   * @throws RangeError on a duplicate `id` or when `aabb` contains `NaN`.
   */
  insertLeaf(id: number, aabb: AABB, payload: T): AABBTreeLeaf<T> {
    if (aabb.hasNaN()) {
      throw new RangeError(`AABBTree.insertLeaf: AABB contains NaN for id ${id}`);
    }
    if (this.idToNode.has(id) || this.unbounded.has(id)) {
      throw new RangeError(`AABBTree.insertLeaf: duplicate id ${id}`);
    }
    if (aabb.isFinite()) {
      const index = this.allocateNode();
      const node = this.pool[index];
      node.aabb.copy(aabb);
      // Share the pool node's box so `updateLeaf` fattening is visible through
      // the public handle.
      const leaf: AABBTreeLeaf<T> = { id, payload, aabb: node.aabb };
      node.leaf = leaf;
      this.idToNode.set(id, index);
      if (this.rootIndex === -1) {
        this.rootIndex = index;
        node.parent = -1;
      } else {
        this.insertNode(index);
      }
      return leaf;
    }
    const leaf: AABBTreeLeaf<T> = { id, payload, aabb: aabb.clone() };
    this.unbounded.set(id, leaf);
    return leaf;
  }

  /** Removes a leaf. Returns `false` when `id` is unknown. */
  removeLeaf(id: number): boolean {
    const unboundedLeaf = this.unbounded.get(id);
    if (unboundedLeaf !== undefined) {
      this.unbounded.delete(id);
      return true;
    }
    const index = this.idToNode.get(id);
    if (index === undefined) {
      return false;
    }
    this.detachNode(index);
    this.freeNode(index);
    this.idToNode.delete(id);
    return true;
  }

  /**
   * Refreshes the fattened bounds of a leaf from its current tight box.
   *
   * When `tight` is still contained in the stored envelope the tree is left
   * untouched and the call is nearly free. Otherwise the envelope is rebuilt
   * from the motion term `velocity · dt · α` plus the static margin `β` and the
   * node is reinserted through the SAH descent.
   *
   * @returns `true` when the tree was refit, `false` when the envelope still
   * covers `tight` (or when the leaf is unbounded).
   * @throws RangeError when `id` is unknown.
   */
  updateLeaf(id: number, tight: AABB, velocity: Vec3, dt: number): boolean {
    const index = this.idToNode.get(id);
    if (index === undefined) {
      if (this.unbounded.has(id)) {
        return false;
      }
      throw new RangeError(`AABBTree.updateLeaf: unknown id ${id}`);
    }
    const node = this.pool[index];
    if (node.aabb.contains(tight)) {
      return false;
    }
    AABB.fatten(node.aabb, tight, velocity, dt, this.velocityScale, this.margin);
    this.detachNode(index);
    if (this.rootIndex === -1) {
      // The leaf was the whole tree; it becomes the root again.
      this.rootIndex = index;
      node.parent = -1;
    } else {
      this.insertNode(index);
    }
    this.updateCount++;
    return true;
  }

  /** Returns the live leaf handle for `id`, or `undefined` when absent. */
  getLeaf(id: number): AABBTreeLeaf<T> | undefined {
    const index = this.idToNode.get(id);
    if (index !== undefined) {
      const leaf = this.pool[index].leaf;
      if (leaf !== null) {
        return leaf;
      }
    }
    return this.unbounded.get(id);
  }

  /** Removes every leaf. */
  clear(): void {
    this.pool.length = 0;
    this.freeList.length = 0;
    this.idToNode.clear();
    this.unbounded.clear();
    this.rootIndex = -1;
    this.updateCount = 0;
  }

  // ------------------------------------------------------------------
  // Queries
  // ------------------------------------------------------------------

  /** Visits every stored leaf (finite and unbounded). */
  forEachLeaf(visit: (leaf: AABBTreeLeaf<T>) => void): void {
    if (this.rootIndex !== -1) {
      this.visitTreeLeaves(this.pool[this.rootIndex].aabb, visit);
    }
    for (const leaf of this.unbounded.values()) {
      visit(leaf);
    }
  }

  /** Visits every leaf whose fat envelope overlaps `aabb`. */
  queryAABB(aabb: AABB, visit: (leaf: AABBTreeLeaf<T>) => void): void {
    this.visitTreeLeaves(aabb, visit);
    for (const leaf of this.unbounded.values()) {
      if (leaf.aabb.overlaps(aabb)) {
        visit(leaf);
      }
    }
  }

  /**
   * Visits every unordered pair of leaves whose fat envelopes overlap, exactly
   * once per pair. Returns the number of reported pairs.
   *
   * This is a self-collision query: it is the broadphase primitive that feeds
   * the narrowphase. Pair ordering within `visit(a, b)` is unspecified but
   * stable for a given tree state.
   */
  queryOverlaps(visit: (a: AABBTreeLeaf<T>, b: AABBTreeLeaf<T>) => void): number {
    let pairCount = 0;
    const report = (a: AABBTreeLeaf<T>, b: AABBTreeLeaf<T>): void => {
      pairCount++;
      visit(a, b);
    };

    if (this.rootIndex !== -1) {
      const stack: number[] = [this.rootIndex, this.rootIndex];
      while (stack.length > 0) {
        const j = stack.pop()!;
        const i = stack.pop()!;
        const nodeI = this.pool[i];
        const nodeJ = this.pool[j];
        if (i === j) {
          if (nodeI.leaf !== null) {
            continue;
          }
          stack.push(nodeI.left, nodeI.left);
          stack.push(nodeI.right, nodeI.right);
          stack.push(nodeI.left, nodeI.right);
          continue;
        }
        if (!nodeI.aabb.overlaps(nodeJ.aabb)) {
          continue;
        }
        const leafI = nodeI.leaf;
        const leafJ = nodeJ.leaf;
        if (leafI !== null && leafJ !== null) {
          report(leafI, leafJ);
          continue;
        }
        if (leafI === null) {
          stack.push(nodeI.left, j);
          stack.push(nodeI.right, j);
          continue;
        }
        stack.push(i, nodeJ.left);
        stack.push(i, nodeJ.right);
      }
    }

    const unboundedLeaves = [...this.unbounded.values()];
    for (let i = 0; i < unboundedLeaves.length; i++) {
      for (let j = i + 1; j < unboundedLeaves.length; j++) {
        report(unboundedLeaves[i], unboundedLeaves[j]);
      }
    }
    for (const unboundedLeaf of unboundedLeaves) {
      this.visitTreeLeaves(unboundedLeaf.aabb, (leaf) => {
        report(unboundedLeaf, leaf);
      });
    }
    return pairCount;
  }

  /**
   * Casts a ray and visits every leaf whose fat envelope it hits, in tree
   * traversal order (not sorted by distance). Use
   * {@link AABBTree.raycastClosest} for nearest-hit queries.
   *
   * `direction` is normalized internally, so `distance` in the reported hits
   * is world-space length.
   *
   * @throws RangeError when `direction` is the zero vector.
   */
  raycast(
    origin: Vec3,
    direction: Vec3,
    visit: (hit: RaycastHit<T>) => void,
    maxDistance: number = Infinity,
  ): void {
    const lengthSq = direction.lengthSq();
    if (lengthSq < EPSILON * EPSILON) {
      throw new RangeError("AABBTree.raycast: direction must be non-zero");
    }
    const inverseLength = 1 / Math.sqrt(lengthSq);
    const dx = direction.x * inverseLength;
    const dy = direction.y * inverseLength;
    const dz = direction.z * inverseLength;

    const makeHit = (leaf: AABBTreeLeaf<T>, distance: number): RaycastHit<T> => ({
      leaf,
      distance,
      point: new Vec3(
        origin.x + dx * distance,
        origin.y + dy * distance,
        origin.z + dz * distance,
      ),
    });

    if (this.rootIndex !== -1) {
      const stack: number[] = [this.rootIndex];
      while (stack.length > 0) {
        const index = stack.pop()!;
        const node = this.pool[index];
        const distance = raySlabDistance(node.aabb, origin, dx, dy, dz, maxDistance);
        if (distance < 0) {
          continue;
        }
        if (node.leaf !== null) {
          visit(makeHit(node.leaf, distance));
        } else {
          stack.push(node.left, node.right);
        }
      }
    }
    for (const leaf of this.unbounded.values()) {
      const distance = raySlabDistance(leaf.aabb, origin, dx, dy, dz, maxDistance);
      if (distance >= 0) {
        visit(makeHit(leaf, distance));
      }
    }
  }

  /** Nearest leaf hit by the ray, or `null`. */
  raycastClosest(
    origin: Vec3,
    direction: Vec3,
    maxDistance: number = Infinity,
  ): RaycastHit<T> | null {
    let closest: RaycastHit<T> | null = null;
    this.raycast(
      origin,
      direction,
      (hit) => {
        if (closest === null || hit.distance < closest.distance) {
          closest = hit;
        }
      },
      maxDistance,
    );
    return closest;
  }

  // ------------------------------------------------------------------
  // Invariants
  // ------------------------------------------------------------------

  /**
   * Verifies structural invariants: parent/child linkage, AVL balance
   * (`|height(L) - height(R)| <= 1`), recomputed node bounds, height labels,
   * and one-to-one correspondence with the leaf registry.
   *
   * @throws Error describing the first violated invariant.
   */
  validate(): void {
    if (this.rootIndex === -1) {
      if (this.idToNode.size !== 0) {
        throw new Error(`validate: ${this.idToNode.size} leaves mapped without a root`);
      }
      return;
    }
    if (this.pool[this.rootIndex].parent !== -1) {
      throw new Error("validate: root has a parent");
    }
    const visited = new Set<number>();
    const stack: number[] = [this.rootIndex];
    let leafCount = 0;
    while (stack.length > 0) {
      const index = stack.pop() as number;
      if (visited.has(index)) {
        throw new Error(`validate: node ${index} reachable twice (cycle)`);
      }
      visited.add(index);
      const node = this.pool[index];
      if (node.leaf !== null) {
        leafCount++;
        if (node.height !== 0) {
          throw new Error(`validate: leaf ${node.leaf.id} has height ${node.height}`);
        }
        if (node.left !== -1 || node.right !== -1) {
          throw new Error(`validate: leaf ${node.leaf.id} has children`);
        }
        if (this.idToNode.get(node.leaf.id) !== index) {
          throw new Error(`validate: id ${node.leaf.id} not mapped to its node`);
        }
        continue;
      }
      if (node.left === -1 || node.right === -1) {
        throw new Error(`validate: internal node ${index} is missing a child`);
      }
      const left = this.pool[node.left];
      const right = this.pool[node.right];
      if (left.parent !== index || right.parent !== index) {
        throw new Error(`validate: child of node ${index} has a stale parent link`);
      }
      if (node.height !== 1 + Math.max(left.height, right.height)) {
        throw new Error(`validate: node ${index} has an inconsistent height`);
      }
      if (Math.abs(left.height - right.height) > 1) {
        throw new Error(`validate: node ${index} violates AVL balance`);
      }
      if (!node.aabb.contains(left.aabb) || !node.aabb.contains(right.aabb)) {
        throw new Error(`validate: node ${index} bounds do not contain its children`);
      }
      stack.push(node.left, node.right);
    }
    if (leafCount !== this.idToNode.size) {
      throw new Error(
        `validate: visited ${leafCount} leaves but ${this.idToNode.size} are registered`,
      );
    }
  }

  // ------------------------------------------------------------------
  // Internals
  // ------------------------------------------------------------------

  private allocateNode(): number {
    const recycled = this.freeList.pop();
    if (recycled !== undefined) {
      const node = this.pool[recycled];
      node.parent = -1;
      node.left = -1;
      node.right = -1;
      node.height = 0;
      node.leaf = null;
      return recycled;
    }
    this.pool.push({
      aabb: new AABB(),
      parent: -1,
      left: -1,
      right: -1,
      height: 0,
      leaf: null,
    });
    return this.pool.length - 1;
  }

  private freeNode(index: number): void {
    const node = this.pool[index];
    node.parent = -1;
    node.left = -1;
    node.right = -1;
    node.height = 0;
    node.leaf = null;
    this.freeList.push(index);
  }

  /** Splices leaf node `index` into the tree through the greedy SAH descent. */
  private insertNode(index: number): void {
    const leafNode = this.pool[index];
    let current = this.rootIndex;
    while (true) {
      const node = this.pool[current];
      if (node.leaf !== null) {
        break;
      }
      const costHere = unionSurfaceArea(node.aabb, leafNode.aabb);
      const leftNode = this.pool[node.left];
      const rightNode = this.pool[node.right];
      const costLeft = unionSurfaceArea(leftNode.aabb, leafNode.aabb) - leftNode.aabb.surfaceArea();
      const costRight =
        unionSurfaceArea(rightNode.aabb, leafNode.aabb) - rightNode.aabb.surfaceArea();
      if (costHere <= costLeft && costHere <= costRight) {
        break;
      }
      current = costLeft < costRight ? node.left : node.right;
    }

    const siblingIndex = current;
    const sibling = this.pool[siblingIndex];
    const parentIndex = this.allocateNode();
    const parent = this.pool[parentIndex];
    AABB.merge(parent.aabb, sibling.aabb, leafNode.aabb);
    parent.parent = sibling.parent;
    parent.height = sibling.height + 1;
    parent.left = siblingIndex;
    parent.right = index;
    sibling.parent = parentIndex;
    leafNode.parent = parentIndex;

    if (parent.parent === -1) {
      this.rootIndex = parentIndex;
    } else {
      const grandparent = this.pool[parent.parent];
      if (grandparent.left === siblingIndex) {
        grandparent.left = parentIndex;
      } else {
        grandparent.right = parentIndex;
      }
    }
    // Refit from the new parent: stopping the SAH descent above a leaf keeps it
    // balanced, but an internal sibling can make the new parent itself
    // unbalanced, so it must participate in the rotation walk.
    this.refitUp(parentIndex);
  }

  /** Detaches leaf node `index`, replacing its parent with the sibling. */
  private detachNode(index: number): void {
    const node = this.pool[index];
    if (index === this.rootIndex) {
      this.rootIndex = -1;
      node.parent = -1;
      return;
    }
    const parentIndex = node.parent;
    const parent = this.pool[parentIndex];
    const siblingIndex = parent.left === index ? parent.right : parent.left;
    const sibling = this.pool[siblingIndex];
    node.parent = -1;

    if (parent.parent === -1) {
      this.rootIndex = siblingIndex;
      sibling.parent = -1;
    } else {
      const grandparent = this.pool[parent.parent];
      if (grandparent.left === parentIndex) {
        grandparent.left = siblingIndex;
      } else {
        grandparent.right = siblingIndex;
      }
      sibling.parent = parent.parent;
    }
    this.freeNode(parentIndex);
    this.refitUp(sibling.parent);
  }

  /**
   * Recomputes bounds and heights from `index` up to the root, applying AVL
   * rotations wherever a subtree became unbalanced.
   */
  private refitUp(index: number): void {
    let current = index;
    while (current !== -1) {
      const node = this.pool[current];
      const left = this.pool[node.left];
      const right = this.pool[node.right];
      node.height = 1 + Math.max(left.height, right.height);
      AABB.merge(node.aabb, left.aabb, right.aabb);

      const balance = right.height - left.height;
      let subtreeRoot = current;
      if (balance > 1) {
        if (this.pool[right.left].height > this.pool[right.right].height) {
          this.rotateRight(node.right);
        }
        subtreeRoot = this.rotateLeft(current);
      } else if (balance < -1) {
        if (this.pool[left.right].height > this.pool[left.left].height) {
          this.rotateLeft(node.left);
        }
        subtreeRoot = this.rotateRight(current);
      }
      current = this.pool[subtreeRoot].parent;
    }
  }

  /**
   * Rotates `index` down to the left: its right child becomes the subtree root
   * and returns that new root. Bounds and heights of both moved nodes are
   * recomputed.
   */
  private rotateLeft(index: number): number {
    const node = this.pool[index];
    const pivotIndex = node.right;
    const pivot = this.pool[pivotIndex];
    const parentIndex = node.parent;

    node.right = pivot.left;
    this.pool[pivot.left].parent = index;
    pivot.left = index;
    node.parent = pivotIndex;
    pivot.parent = parentIndex;
    if (parentIndex === -1) {
      this.rootIndex = pivotIndex;
    } else {
      const parent = this.pool[parentIndex];
      if (parent.left === index) {
        parent.left = pivotIndex;
      } else {
        parent.right = pivotIndex;
      }
    }
    this.refitNode(index);
    this.refitNode(pivotIndex);
    return pivotIndex;
  }

  /** Mirror of {@link AABBTree.rotateLeft}. */
  private rotateRight(index: number): number {
    const node = this.pool[index];
    const pivotIndex = node.left;
    const pivot = this.pool[pivotIndex];
    const parentIndex = node.parent;

    node.left = pivot.right;
    this.pool[pivot.right].parent = index;
    pivot.right = index;
    node.parent = pivotIndex;
    pivot.parent = parentIndex;
    if (parentIndex === -1) {
      this.rootIndex = pivotIndex;
    } else {
      const parent = this.pool[parentIndex];
      if (parent.left === index) {
        parent.left = pivotIndex;
      } else {
        parent.right = pivotIndex;
      }
    }
    this.refitNode(index);
    this.refitNode(pivotIndex);
    return pivotIndex;
  }

  /** Recomputes height and bounds of an internal node from its children. */
  private refitNode(index: number): void {
    const node = this.pool[index];
    const left = this.pool[node.left];
    const right = this.pool[node.right];
    node.height = 1 + Math.max(left.height, right.height);
    AABB.merge(node.aabb, left.aabb, right.aabb);
  }

  /** Traverses the finite tree, visiting leaves that overlap `aabb`. */
  private visitTreeLeaves(aabb: AABB, visit: (leaf: AABBTreeLeaf<T>) => void): void {
    if (this.rootIndex === -1) {
      return;
    }
    const stack: number[] = [this.rootIndex];
    while (stack.length > 0) {
      const index = stack.pop()!;
      const node = this.pool[index];
      if (!node.aabb.overlaps(aabb)) {
        continue;
      }
      if (node.leaf !== null) {
        visit(node.leaf);
      } else {
        stack.push(node.left, node.right);
      }
    }
  }
}

/**
 * Slab test between a ray and an AABB (inclusive), returning the entry
 * distance along the unit direction, or `-1` when there is no hit within
 * `[0, maxDistance]`. Axis-parallel rays are handled without `0 * Infinity`.
 */
function raySlabDistance(
  aabb: AABB,
  origin: Vec3,
  dx: number,
  dy: number,
  dz: number,
  maxDistance: number,
): number {
  let tMin = 0;
  let tMax = maxDistance;

  if (Math.abs(dx) < EPSILON) {
    if (origin.x < aabb.min.x || origin.x > aabb.max.x) {
      return -1;
    }
  } else {
    const inverse = 1 / dx;
    let t1 = (aabb.min.x - origin.x) * inverse;
    let t2 = (aabb.max.x - origin.x) * inverse;
    if (t1 > t2) {
      const swap = t1;
      t1 = t2;
      t2 = swap;
    }
    if (t1 > tMin) {
      tMin = t1;
    }
    if (t2 < tMax) {
      tMax = t2;
    }
    if (tMin > tMax) {
      return -1;
    }
  }

  if (Math.abs(dy) < EPSILON) {
    if (origin.y < aabb.min.y || origin.y > aabb.max.y) {
      return -1;
    }
  } else {
    const inverse = 1 / dy;
    let t1 = (aabb.min.y - origin.y) * inverse;
    let t2 = (aabb.max.y - origin.y) * inverse;
    if (t1 > t2) {
      const swap = t1;
      t1 = t2;
      t2 = swap;
    }
    if (t1 > tMin) {
      tMin = t1;
    }
    if (t2 < tMax) {
      tMax = t2;
    }
    if (tMin > tMax) {
      return -1;
    }
  }

  if (Math.abs(dz) < EPSILON) {
    if (origin.z < aabb.min.z || origin.z > aabb.max.z) {
      return -1;
    }
  } else {
    const inverse = 1 / dz;
    let t1 = (aabb.min.z - origin.z) * inverse;
    let t2 = (aabb.max.z - origin.z) * inverse;
    if (t1 > t2) {
      const swap = t1;
      t1 = t2;
      t2 = swap;
    }
    if (t1 > tMin) {
      tMin = t1;
    }
    if (t2 < tMax) {
      tMax = t2;
    }
    if (tMin > tMax) {
      return -1;
    }
  }

  return tMin;
}
