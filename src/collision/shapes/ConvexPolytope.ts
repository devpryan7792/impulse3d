import { Mat3 } from "../../math/Mat3.js";
import { Quat } from "../../math/Quat.js";
import type { Transform } from "../../math/Transform.js";
import { Vec3 } from "../../math/Vec3.js";
import { AABB } from "../AABB.js";
import { Shape, ShapeType } from "./Shape.js";

/** Scratch storage for the local direction / transformed vertex (single-threaded use). */
const scratchMat3 = new Mat3();
const scratchVec3 = new Vec3();

/**
 * Convex polytope given by its vertex set, e.g. the result of a convex hull
 * computation. The vertex set is validated for finiteness but **not** for
 * convexity: feeding a non-convex point cloud yields the support mapping of its
 * convex hull, which silently disagrees with the input geometry.
 *
 * Support mapping (Section 3.1): `S(d) = argmax_{v_i} (v_i · d)`. The maximum is
 * found in the local frame (a dot product is invariant under rotation), then
 * the winning vertex is transformed to world space.
 */
export class ConvexPolytope extends Shape {
  public readonly type: ShapeType = ShapeType.ConvexPolytope;
  public readonly vertices: readonly Vec3[];

  constructor(vertices: readonly Vec3[]) {
    super();
    if (vertices.length < 4) {
      throw new RangeError(
        `ConvexPolytope: at least 4 vertices are required, got ${vertices.length}`,
      );
    }
    for (const vertex of vertices) {
      if (!vertex.isFinite()) {
        throw new RangeError(`ConvexPolytope: vertex must be finite, got ${vertex.toString()}`);
      }
    }
    this.vertices = vertices.map((vertex) => vertex.clone());
  }

  override computeAABB(transform: Transform, out: AABB = new AABB()): AABB {
    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    for (const vertex of this.vertices) {
      transform.transformPoint(scratchVec3, vertex);
      if (scratchVec3.x < minX) minX = scratchVec3.x;
      if (scratchVec3.y < minY) minY = scratchVec3.y;
      if (scratchVec3.z < minZ) minZ = scratchVec3.z;
      if (scratchVec3.x > maxX) maxX = scratchVec3.x;
      if (scratchVec3.y > maxY) maxY = scratchVec3.y;
      if (scratchVec3.z > maxZ) maxZ = scratchVec3.z;
    }
    out.min.set(minX, minY, minZ);
    out.max.set(maxX, maxY, maxZ);
    return out;
  }

  override getSupport(direction: Vec3, transform: Transform, out: Vec3 = new Vec3()): Vec3 {
    const rotation = Quat.toMat3(scratchMat3, transform.rotation);
    Mat3.transformVectorTranspose(scratchVec3, rotation, direction);
    const dx = scratchVec3.x;
    const dy = scratchVec3.y;
    const dz = scratchVec3.z;

    let bestIndex = 0;
    let bestDot = -Infinity;
    for (let i = 0; i < this.vertices.length; i++) {
      const vertex = this.vertices[i];
      const dot = vertex.x * dx + vertex.y * dy + vertex.z * dz;
      if (dot > bestDot) {
        bestDot = dot;
        bestIndex = i;
      }
    }
    return transform.transformPoint(out, this.vertices[bestIndex]);
  }
}
