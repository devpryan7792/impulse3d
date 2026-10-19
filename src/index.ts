/**
 * Impulse3D public API: math, shapes, broadphase, narrowphase/manifolds and
 * the rigid-body dynamics layer.
 *
 * The package has zero runtime dependencies; every export is a self-contained
 * numerical or geometric primitive usable from Node, browsers, and WebAssembly
 * glue code.
 */

// --- Mathematics -----------------------------------------------------------
export { Vec3 } from "./math/Vec3.js";
export { Mat3 } from "./math/Mat3.js";
export { Mat4 } from "./math/Mat4.js";
export { Quat } from "./math/Quat.js";
export { Transform } from "./math/Transform.js";
export { EPSILON, EPSILON_SQ, clamp, approximatelyEqual } from "./math/MathUtils.js";

// --- Collision geometry ----------------------------------------------------
export { AABB } from "./collision/AABB.js";
export { Shape, ShapeType } from "./collision/shapes/Shape.js";
export { Sphere } from "./collision/shapes/Sphere.js";
export { Box } from "./collision/shapes/Box.js";
export { Capsule } from "./collision/shapes/Capsule.js";
export { Plane } from "./collision/shapes/Plane.js";
export { ConvexPolytope } from "./collision/shapes/ConvexPolytope.js";

// --- Broadphase ------------------------------------------------------------
export { AABBTree } from "./collision/broadphase/AABBTree.js";
export type {
  AABBTreeLeaf,
  AABBTreeOptions,
  AABBTreeStats,
  RaycastHit,
} from "./collision/broadphase/AABBTree.js";

// --- Narrowphase -----------------------------------------------------------
export { collideShapes, planeManifold } from "./collision/narrowphase/Narrowphase.js";
export { doSimplex, gjkIntersects, minkowskiSupport } from "./collision/narrowphase/GJK.js";
export type { GJKResult, SupportVertex } from "./collision/narrowphase/GJK.js";
export { epa } from "./collision/narrowphase/EPA.js";
export type { EPAResult } from "./collision/narrowphase/EPA.js";
export { flipManifold, hashContactId } from "./collision/narrowphase/Contact.js";
export type { ContactManifold, ContactPoint } from "./collision/narrowphase/Contact.js";
export {
  boxBoxManifold,
  boxWorldVertices,
  clipPolygon,
  ContactCache,
  reduceManifold,
} from "./collision/manifold/ContactManifold.js";
export type { CachedImpulse } from "./collision/manifold/ContactManifold.js";


// --- Dynamics --------------------------------------------------------------
export { BodyType, RigidBody, computeLocalInertia } from "./dynamics/RigidBody.js";
export type { RigidBodyOptions } from "./dynamics/RigidBody.js";
export { ConstraintRow } from "./dynamics/ConstraintRow.js";
export { ContactConstraint, ContactSolver } from "./dynamics/ContactSolver.js";
export type { ContactPair, ContactSolverOptions } from "./dynamics/ContactSolver.js";
export {
  BallSocketJoint,
  DistanceJoint,
  HingeJoint,
  Joint,
} from "./dynamics/Joint.js";
export type { HingeJointOptions, JointOptions } from "./dynamics/Joint.js";
export { World } from "./dynamics/World.js";
export type { WorldOptions, WorldStats } from "./dynamics/World.js";
