# Impulse3D: Full-Fledged 3D Rigid Body Physics Engine
## Architectural & Mathematical Engineering Specification (v1.0.0-PROD)
**Author:** Antigravity Physics & Systems Architecture Group  
**Target Environment:** Node.js / Modern Browsers (ES2023+ / WebAssembly / WebGPU)  
**Primary Language Target:** TypeScript (Strict Mode) / Rust (Wasm target option)  
**Visualization Layer:** Three.js / WebGL / WebGPU Render Pipeline  
**Status:** Approved for Autonomous Implementation  

---

## 1. Executive Summary & Scope

### 1.1 The Engineering Objective
`Impulse3D` is a zero-dependency, mathematically rigorous 3D rigid body physics engine featuring:
1. **Full 6-DOF Rigid Body Kinematics:** Position, linear velocity, orientation (unit quaternions), angular velocity, and dynamic world-space inertia tensor rotation.
2. **Dynamic Broadphase:** Self-balancing Dynamic Bounding Volume Hierarchy (AABB Tree) with Surface Area Heuristic (SAH) and raycast traversal.
3. **Robust Narrowphase:**
   * Analytic closed-form collision detectors for basic primitives (Sphere, Capsule, Plane, Box).
   * **GJK (Gilbert-Johnson-Keerthi):** Convex polytope distance and intersection detection using support mapping.
   * **EPA (Expanding Polytope Algorithm):** Accurate penetration depth, contact normal, and feature extraction on overlapping convex hulls.
   * **Persistent Contact Manifold Generator:** Temporal contact caching with Sutherland-Hodgman polygon clipping, retaining up to 4 points per face-face collision.
4. **Sequential Impulse Solver (Erin Catto / Box2D / Bullet model):**
   * Pre-calculated effective mass matrices ($K = J M^{-1} J^T$).
   * Projected Gauss-Seidel velocity-level constraint solving with Baumgarte stabilization.
   * Full 3D Coulomb friction cones (two orthogonal tangent impulses per contact point).
   * **Warm Starting:** Re-projection of accumulated normal and friction impulses across frames for stable stacking.
5. **Joints & Articulations:** Ball-and-Socket (Spherical), Revolute (Hinge), Prismatic (Slider), and Distance Spring constraints.
6. **Simulation Islands & Sleeping:** Union-Find / BFS graph clustering to transition stationary body clusters into a zero-CPU sleep state.
7. **Interactive 3D Harness:** Modular Three.js / WebGPU renderer with debug draw calls (contact points, impulse arrows, AABB wireframes) and mouse ray spring interaction.

```
+---------------------------------------------------------------------------------------+
|                                    WORLD STEP LOOP                                    |
+---------------------------------------------------------------------------------------+
|  1. Integration (Predictive): v* = v + dt * M^-1 * F_ext ; w* = w + dt * I^-1 * T_ext |
|  2. Broadphase: Update Dynamic AABB Tree -> Compute Potential Overlap Pairs           |
|  3. Narrowphase: Analytic & GJK/EPA -> Generate Persistent Contact Manifolds          |
|  4. Island Partitioning: Build Contact Graph -> Identify Active vs Sleeping Islands   |
|  5. Solver Warm-Start: Apply 0.85 * cached impulses from previous step               |
|  6. Velocity Solver (8-16 iterations): Solve Contacts, Friction & Bilateral Joints   |
|  7. Position Integration: x = x + dt * v ; q = Normalize(q + 0.5 * dt * w (x) q)      |
|  8. State Sleep Evaluation: Compute Kinetic Energy -> Mark Inactive Islands           |
+---------------------------------------------------------------------------------------+
```

---

## 2. Mathematical Foundation & Kinematics

### 2.1 State Representation
Each rigid body $i$ possesses a 13-dimensional kinematic state vector:

$$\mathbf{Y}_i(t) = \begin{bmatrix} \mathbf{x}_i(t) \\ \mathbf{q}_i(t) \\ \mathbf{P}_i(t) \\ \mathbf{L}_i(t) \end{bmatrix} \in \mathbb{R}^3 \times \mathbb{H} \times \mathbb{R}^3 \times \mathbb{R}^3$$

Where:
* $\mathbf{x}_i \in \mathbb{R}^3$: Position of the center of mass.
* $\mathbf{q}_i = [s, v_x, v_y, v_z]^T \in \mathbb{H}$: Orientation quaternion ($\|\mathbf{q}_i\| = 1$).
* $\mathbf{P}_i = m_i \mathbf{v}_i \in \mathbb{R}^3$: Linear momentum.
* $\mathbf{L}_i = \mathbf{I}_i(t) \boldsymbol{\omega}_i \in \mathbb{R}^3$: Angular momentum.

Auxiliary derived quantities:
* Linear velocity: $\mathbf{v}_i = m_i^{-1} \mathbf{P}_i$ (or inverse mass $w_i = \frac{1}{m_i}$; for static bodies $w_i = 0$).
* Rotation matrix: $\mathbf{R}_i \in \mathrm{SO}(3)$ derived from $\mathbf{q}_i$:
  $$\mathbf{R}(\mathbf{q}) = \begin{bmatrix} 
  1 - 2(y^2 + z^2) & 2(xy - sz) & 2(xz + sy) \\ 
  2(xy + sz) & 1 - 2(x^2 + z^2) & 2(yz - sx) \\ 
  2(xz - sy) & 2(yz + sx) & 1 - 2(x^2 + y^2) 
  \end{bmatrix}$$
* World-space inverse inertia tensor:
  $$\mathbf{I}_i^{-1}(t) = \mathbf{R}_i(t) \, \mathbf{I}_{\text{body}, i}^{-1} \, \mathbf{R}_i(t)^T$$
* Angular velocity: $\boldsymbol{\omega}_i = \mathbf{I}_i^{-1}(t) \mathbf{L}_i$.

### 2.2 Numerical Integration (Symplectic Semi-Implicit Euler)
Given timestep $\Delta t$:

1. **Velocity Update:**
   $$\mathbf{v}(t + \Delta t) = \mathbf{v}(t) + \Delta t \left( w \, \mathbf{F}_{\text{ext}} + w \, \mathbf{F}_{\text{gravity}} \right)$$
   $$\boldsymbol{\omega}(t + \Delta t) = \boldsymbol{\omega}(t) + \Delta t \, \mathbf{I}^{-1} \left( \mathbf{T}_{\text{ext}} - \boldsymbol{\omega}(t) \times (\mathbf{I} \boldsymbol{\omega}(t)) \right)$$

2. **Position Update (post-constraint velocity resolution):**
   $$\mathbf{x}(t + \Delta t) = \mathbf{x}(t) + \Delta t \, \mathbf{v}(t + \Delta t)$$

3. **Orientation Quaternion Integration:**
   Let pure imaginary quaternion $\boldsymbol{\Omega} = [0, \boldsymbol{\omega}_x, \boldsymbol{\omega}_y, \boldsymbol{\omega}_z]^T$.
   $$\mathbf{q}(t + \Delta t) = \mathbf{q}(t) + \frac{\Delta t}{2} \left( \boldsymbol{\Omega} \otimes \mathbf{q}(t) \right)$$
   $$\mathbf{q}(t + \Delta t) \leftarrow \frac{\mathbf{q}(t + \Delta t)}{\|\mathbf{q}(t + \Delta t)\|}$$

### 2.3 Closed-Form Local Inertia Tensors ($\mathbf{I}_{\text{body}}$)

* **Solid Sphere (radius $r$, mass $m$):**
  $$\mathbf{I} = \frac{2}{5} m r^2 \, \mathbf{I}_{3\times 3}$$

* **Solid Box (half-extents $\mathbf{e} = [h_x, h_y, h_z]$):**
  $$\mathbf{I} = \frac{1}{12} m \begin{bmatrix} (2h_y)^2 + (2h_z)^2 & 0 & 0 \\ 0 & (2h_x)^2 + (2h_z)^2 & 0 \\ 0 & 0 & (2h_x)^2 + (2h_y)^2 \end{bmatrix}$$

* **Solid Capsule (radius $r$, cylinder height $h$, total mass $m$):**
  Derived as composite of cylinder ($m_{\text{cyl}}$) + two hemispheres ($m_{\text{hem}}$) using Parallel Axis Theorem.

---

## 3. Collision Geometries & Support Mapping

Every convex shape implements the **Support Function** $S_A(\mathbf{d}) = \arg\max_{\mathbf{p} \in A} (\mathbf{p} \cdot \mathbf{d})$:

### 3.1 Primitive Support Implementations
```typescript
interface Shape {
  type: ShapeType;
  computeAABB(transform: Transform): AABB;
  getSupport(direction: Vec3, transform: Transform): Vec3;
}
```

* **Sphere:**
  $$S(\mathbf{d}) = \mathbf{x} + r \frac{\mathbf{d}}{\|\mathbf{d}\|}$$

* **Box (half-extents $\mathbf{h} = [h_x, h_y, h_z]$, rotation $\mathbf{R}$, center $\mathbf{c}$):**
  Compute direction in local frame $\mathbf{d}_{\text{local}} = \mathbf{R}^T \mathbf{d}$.
  $$\mathbf{s}_{\text{local}} = \begin{bmatrix} \operatorname{sgn}(d_x) h_x \\ \operatorname{sgn}(d_y) h_y \\ \operatorname{sgn}(d_z) h_z \end{bmatrix}$$
  $$S(\mathbf{d}) = \mathbf{c} + \mathbf{R} \, \mathbf{s}_{\text{local}}$$

* **Capsule (radius $r$, axis vector from $\mathbf{a}$ to $\mathbf{b}$):**
  $$S(\mathbf{d}) = (\mathbf{d} \cdot \mathbf{a} > \mathbf{d} \cdot \mathbf{b} \,?\, \mathbf{a} : \mathbf{b}) + r \frac{\mathbf{d}}{\|\mathbf{d}\|}$$

* **Convex Polytope ($N$ vertices $\{\mathbf{v}_1, \dots, \mathbf{v}_N\}$):**
  $$S(\mathbf{d}) = \arg\max_{\mathbf{v}_i} (\mathbf{v}_i \cdot \mathbf{d})$$

---

## 4. Broadphase: Dynamic AABB Bounding Tree

To avoid $O(N^2)$ checks, `Impulse3D` uses an incremental, self-balancing binary AABB tree with a Surface Area Heuristic (SAH).

### 4.1 AABB Envelope Expansion & Velocity Fattening
To avoid rebuilding tree nodes every frame for moving bodies:
$$\mathbf{AABB}_{\text{fat}} = \mathbf{AABB}_{\text{tight}} + \mathbf{v} \cdot \Delta t \times \alpha + \beta$$
* Velocity projection: expands box along motion vector.
* Margin $\beta$: static margin ($0.05\text{ m}$) ensuring stationary or slow bodies never trigger tree updates.

### 4.2 Tree Insertion with SAH Cost Metric
When inserting leaf node $L$ into tree root $N$:
At node $C$ with children $C_{\text{left}}, C_{\text{right}}$:
* Cost of creating new parent with $L$: $\text{Area}(C \cup L)$
* Cost of descending left: $(\text{Area}(C_{\text{left}} \cup L) - \text{Area}(C_{\text{left}})) + \text{Cost}(C_{\text{left}}, L)$
* Cost of descending right: $(\text{Area}(C_{\text{right}} \cup L) - \text{Area}(C_{\text{right}})) + \text{Cost}(C_{\text{right}}, L)$
Descend along minimum cost branch; rebalance via AVL-style tree rotations.

---

## 5. Narrowphase: GJK, EPA, & Contact Generation

For any two convex shapes $A$ and $B$, the Minkowski difference is defined as:
$$A \ominus B = \{ \mathbf{a} - \mathbf{b} \mid \mathbf{a} \in A, \, \mathbf{b} \in B \}$$
Shapes $A$ and $B$ intersect if and only if the origin $\mathbf{0} \in A \ominus B$.
The support mapping of the Minkowski difference is:
$$S_{A \ominus B}(\mathbf{d}) = S_A(\mathbf{d}) - S_B(-\mathbf{d})$$

### 5.1 GJK Intersection & Distance Algorithm
1. Initialize simplex with initial search direction $\mathbf{d} = \mathbf{c}_B - \mathbf{c}_A$.
2. Compute new support point: $\mathbf{w} = S_{A \ominus B}(\mathbf{d})$.
3. If $\mathbf{w} \cdot \mathbf{d} < 0$, the origin lies outside the Minkowski difference $\implies$ **No intersection** (Return False).
4. Add $\mathbf{w}$ to simplex $\mathcal{S}$.
5. Run `DoSimplex(&mut \mathcal{S}, &mut \mathbf{d})`:
   * **1-Simplex (Line segment $AB$):** Check Voronoi regions; pick direction orthogonal to $AB$ pointing towards $\mathbf{0}$.
   * **2-Simplex (Triangle $ABC$):** Compute face normals; determine which edge or face region contains $\mathbf{0}$.
   * **3-Simplex (Tetrahedron $ABCD$):** Check 4 triangular faces. If $\mathbf{0}$ is strictly inside all 4 faces $\implies$ **Intersection detected** (Proceed to EPA). Else, drop redundant vertex and update direction $\mathbf{d}$.

### 5.2 Expanding Polytope Algorithm (EPA)
When GJK terminates with a 4-vertex tetrahedron enclosing the origin:
1. Initialize a triangular mesh polytope $\mathcal{P}$ from the 4 vertices of the enclosing GJK simplex.
2. For each triangular face $F_k$ of $\mathcal{P}$, compute its outward normal $\mathbf{n}_k$ and distance to origin $d_k = \mathbf{n}_k \cdot \mathbf{v}_{k,0}$.
3. Find face $F_{\text{closest}}$ with minimal $d_{\text{min}}$.
4. Query support point in that normal direction: $\mathbf{w} = S_{A \ominus B}(\mathbf{n}_{\text{closest}})$.
5. If $\mathbf{w} \cdot \mathbf{n}_{\text{closest}} - d_{\text{min}} < \epsilon$ (tolerance $10^{-4}\text{ m}$):
   * **Terminated:** Penetration depth is $d_{\text{min}}$.
   * Contact normal is $\mathbf{n}_{\text{closest}}$.
   * Extract contact point using barycentric coordinates of projection on $F_{\text{closest}}$.
6. Else:
   * Perform Silhouette/Horizon search on $\mathcal{P}$ from viewpoint $\mathbf{w}$.
   * Remove visible faces; stitch new triangular faces connecting the horizon edges to $\mathbf{w}$.
   * Loop to step 2 (max 32 iterations).

### 5.3 Persistent Contact Manifold Assembly
A face-face collision in 3D produces a planar contact area, not a single point. To avoid micro-jitter:
1. Identify the reference face (most aligned with contact normal) and incident face.
2. Clip incident polygon against reference face side planes using **Sutherland-Hodgman Polygon Clipping**.
3. Keep only points with penetration depth $\ge 0$.
4. **Manifold Reduction to 4 Points:**
   * Point 1: Deepest penetration.
   * Point 2: Furthest from Point 1.
   * Point 3: Forms largest triangle area with Points 1 & 2.
   * Point 4: Forms largest quad area with Points 1, 2, & 3.
5. Store contact ID (composed of vertex feature hashes) to enable **Warm Starting** in the solver.

---

## 6. Constraint & Sequential Impulse Dynamics Solver

### 6.1 Non-Penetration Constraint Formulation
For contact point $k$ between Body 1 and Body 2:
Let $\mathbf{r}_1 = \mathbf{p}_k - \mathbf{x}_1$ and $\mathbf{r}_2 = \mathbf{p}_k - \mathbf{x}_2$.
The contact separation velocity along normal $\mathbf{n}$ is:
$$v_n = \left( \mathbf{v}_2 + \boldsymbol{\omega}_2 \times \mathbf{r}_2 - \mathbf{v}_1 - \boldsymbol{\omega}_1 \times \mathbf{r}_1 \right) \cdot \mathbf{n} = \mathbf{J}_n \mathbf{V}$$

Where velocity vector $\mathbf{V} = [\mathbf{v}_1, \boldsymbol{\omega}_1, \mathbf{v}_2, \boldsymbol{\omega}_2]^T$, and the Jacobian row is:
$$\mathbf{J}_n = \begin{bmatrix} -\mathbf{n}^T & -(\mathbf{r}_1 \times \mathbf{n})^T & \mathbf{n}^T & (\mathbf{r}_2 \times \mathbf{n})^T \end{bmatrix}$$

### 6.2 Effective Mass Matrix ($M_{\text{eff}}$)
$$K_n = \mathbf{J}_n \mathbf{M}^{-1} \mathbf{J}_n^T = w_1 + w_2 + (\mathbf{r}_1 \times \mathbf{n})^T \mathbf{I}_1^{-1} (\mathbf{r}_1 \times \mathbf{n}) + (\mathbf{r}_2 \times \mathbf{n})^T \mathbf{I}_2^{-1} (\mathbf{r}_2 \times \mathbf{n})$$
$$m_{\text{eff}, n} = \frac{1}{K_n}$$

### 6.3 Position Stabilization (Baumgarte Correction)
To eliminate visual sinking without adding energetic explosions:
$$b = -\frac{\beta}{\Delta t} \max(0, d_k - d_{\text{slop}}) + e \cdot \min(0, v_n + v_{\text{bounce\_threshold}})$$
* Baumgarte factor: $\beta = 0.2$
* Penetration slop: $d_{\text{slop}} = 0.005\text{ m}$ (avoids micro-jitter on resting planes)
* Restitution coefficient: $e = \sqrt{e_1 \cdot e_2}$ (only applies if closing velocity $> 1.0\text{ m/s}$).

### 6.4 Normal Impulse Update Loop
1. Compute relative velocity: $v_n = \mathbf{J}_n \mathbf{V}$.
2. Compute unconstrained delta impulse:
   $$\Delta \lambda_n = -m_{\text{eff}, n} (v_n - b)$$
3. Clamp accumulated impulse (Signorini condition: contacts only push, never pull):
   $$\lambda_{n, \text{new}} = \max(0, \lambda_{n, \text{old}} + \Delta \lambda_n)$$
   $$\Delta \lambda_n^* = \lambda_{n, \text{new}} - \lambda_{n, \text{old}}$$
   $$\lambda_{n, \text{old}} \leftarrow \lambda_{n, \text{new}}$$
4. Apply impulse to body velocities:
   $$\mathbf{v}_1 \leftarrow \mathbf{v}_1 - w_1 \Delta \lambda_n^* \mathbf{n}$$
   $$\boldsymbol{\omega}_1 \leftarrow \boldsymbol{\omega}_1 - \mathbf{I}_1^{-1} (\mathbf{r}_1 \times (\Delta \lambda_n^* \mathbf{n}))$$
   $$\mathbf{v}_2 \leftarrow \mathbf{v}_2 + w_2 \Delta \lambda_n^* \mathbf{n}$$
   $$\boldsymbol{\omega}_2 \leftarrow \boldsymbol{\omega}_2 + \mathbf{I}_2^{-1} (\mathbf{r}_2 \times (\Delta \lambda_n^* \mathbf{n}))$$

### 6.5 Coulomb Friction Solver (Two Orthogonal Tangents)
Form orthonormal tangent vectors $\mathbf{t}_1, \mathbf{t}_2$ perpendicular to $\mathbf{n}$:
For each tangent $i \in \{1, 2\}$:
1. Form friction Jacobian $\mathbf{J}_{t_i} = [ -\mathbf{t}_i^T, -(\mathbf{r}_1 \times \mathbf{t}_i)^T, \mathbf{t}_i^T, (\mathbf{r}_2 \times \mathbf{t}_i)^T ]$.
2. Compute tangent effective mass $K_{t_i} = \mathbf{J}_{t_i} \mathbf{M}^{-1} \mathbf{J}_{t_i}^T$.
3. Compute $\Delta \lambda_{t_i} = -m_{\text{eff}, t_i} (\mathbf{J}_{t_i} \mathbf{V})$.
4. **Friction Cone Clamping (Box Friction Approximation):**
   $$\lambda_{\text{max}} = \mu \cdot \lambda_{n, \text{accumulated}}$$
   $$\lambda_{t_i, \text{new}} = \operatorname{clamp}(\lambda_{t_i, \text{old}} + \Delta \lambda_{t_i}, -\lambda_{\text{max}}, \lambda_{\text{max}})$$
   $$\Delta \lambda_{t_i}^* = \lambda_{t_i, \text{new}} - \lambda_{t_i, \text{old}}$$
   $$\lambda_{t_i, \text{old}} \leftarrow \lambda_{t_i, \text{new}}$$
5. Apply impulse to velocities along $\mathbf{t}_i$.

---

## 7. Joint & Articulation Constraints

Joints remove relative degrees of freedom between two bodies:

| Joint Type | Constrained DOFs | Free DOFs | Physical Equivalent |
| :--- | :--- | :--- | :--- |
| **Spherical (Ball-and-Socket)** | 3 Linear | 3 Angular | Shoulder, ragdoll joint |
| **Revolute (Hinge)** | 3 Linear + 2 Angular | 1 Angular | Door hinge, wheel axle |
| **Prismatic (Slider)** | 2 Linear + 3 Angular | 1 Linear | Elevator, hydraulic piston |
| **Distance Joint** | 1 Linear (scalar distance) | 5 Mixed | Rigid rod or spring suspension |

### 7.1 Spherical Joint Constraint Equation
Constrains anchor $\mathbf{p}_1$ on Body 1 to coincide with anchor $\mathbf{p}_2$ on Body 2 in world space:
$$\mathbf{C}(\mathbf{x}_1, \mathbf{x}_2) = (\mathbf{x}_1 + \mathbf{R}_1 \mathbf{a}_{1, \text{local}}) - (\mathbf{x}_2 + \mathbf{R}_2 \mathbf{a}_{2, \text{local}}) = \mathbf{0}$$
Velocity Jacobian has 3 rows (rank 3):
$$\mathbf{J}_{\text{ball}} = \begin{bmatrix} -\mathbf{I}_{3\times 3} & [\mathbf{r}_1]_\times & \mathbf{I}_{3\times 3} & -[\mathbf{r}_2]_\times \end{bmatrix}$$
Solved simultaneously using $3 \times 3$ block matrix inversion.

---

## 8. Simulation Islands & Sleeping Architecture

To achieve $>10,000$ bodies in a scene without CPU degradation:
1. **Island Graph Construction:**
   * Graph nodes = Rigid bodies.
   * Graph edges = Active contact manifolds and joints.
   * Static/kinematic bodies act as infinite-mass boundary condition sinks (they do not connect distinct islands).
2. **Connected Components:** Run Disjoint Set Union (DSU) or BFS every frame.
3. **Sleep Trigger Condition:**
   For all bodies $i$ in island $\mathcal{I}$:
   $$\text{Kinetic Energy } E_i = \frac{1}{2} m_i \|\mathbf{v}_i\|^2 + \frac{1}{2} \boldsymbol{\omega}_i^T \mathbf{I}_i \boldsymbol{\omega}_i < E_{\text{threshold}} \quad (10^{-4}\text{ J})$$
   If condition holds for continuous duration $> 0.5\text{ seconds}$:
   Mark entire island as `SLEEPING`. Set $\mathbf{v} = \mathbf{0}, \boldsymbol{\omega} = \mathbf{0}$.
4. **Wakeup Trigger:**
   If any sleeping body is contacted by an active (awake) body or receives user force, the entire island wakes up immediately.

---

## 9. Interactive Three.js / WebGPU Rendering Harness

The engine separates physics simulation from rendering via clean decoupled updates:

```typescript
export class PhysicsScene {
  private world: World;
  private visualizer: ThreeVisualizer;
  private mouseJoint: MouseConstraintHelper;

  public init(canvas: HTMLCanvasElement): void {
    // 1. Setup Three.js Scene, Camera, OrbitControls, ShadowMap
    // 2. Initialize Debug Lines (AABB bounding boxes, contact normals)
    // 3. Register Pointer Events (Raycasting to drag bodies with spring joint)
  }

  public tick(dt: number): void {
    this.world.step(1 / 60);
    this.visualizer.syncTransforms(); // copies body.position/quaternion to Mesh
    this.visualizer.render();
  }
}
```

### 9.1 Included Presets
* **Demo 1: The 100-Box Jenga Tower:** Stacking stability test for sequential impulse warm starting.
* **Demo 2: Newton's Cradle:** Elastic collision and momentum conservation test.
* **Demo 3: Articulated Ragdoll:** Multi-link humanoid dropping down a staircase using spherical and revolute joints.
* **Demo 4: Raycast Suspension Vehicle:** 4-wheel car with spring dampers, friction steering, and torque acceleration.
* **Demo 5: Demolition Playground:** Wrecking ball suspended by distance constraint destroying a brick wall.

---

## 10. Phased Implementation Roadmap

### Phase 1: Mathematics & Linear Algebra Core (Days 1–3)
- [ ] Implement `src/math/Vec3.ts`: SIMD-friendly vector operations, dot, cross, triple product.
- [ ] Implement `src/math/Mat3.ts` & `Mat4.ts`: Inertia tensor transforms, skew-symmetric cross-product matrix $[\mathbf{r}]_\times$.
- [ ] Implement `src/math/Quat.ts`: Slerp, conjugate, quaternion multiplication, quaternion integration from angular velocity.
- [ ] Comprehensive unit tests (`tests/math.test.ts`) validating 100% matrix identities and numerical precision.

### Phase 2: Shapes & Broadphase AABB Tree (Days 4–7)
- [ ] Implement shape classes: `Sphere`, `Box`, `Capsule`, `Plane`, `ConvexPolytope`.
- [ ] Implement support mapping functions $S(\mathbf{d})$ for all shapes.
- [ ] Build `src/collision/broadphase/AABBTree.ts`: SAH tree cost insertion, AABB fattening with motion velocity, node raycasting.
- [ ] Broadphase test suite: overlapping pairs correctness against brute-force $O(N^2)$ ground truth.

### Phase 3: Narrowphase: Analytic, GJK & EPA (Days 8–12)
- [ ] Implement analytic fast-paths: Sphere-Sphere, Sphere-Box, Sphere-Capsule.
- [ ] Implement `src/collision/narrowphase/GJK.ts`: Line, triangle, tetrahedron simplex solvers.
- [ ] Implement `src/collision/narrowphase/EPA.ts`: Horizon edge search, face subdivision, termination thresholds.
- [ ] Implement `src/collision/manifold/ContactManifold.ts`: Sutherland-Hodgman clipping, 4-point reduction, contact caching.

### Phase 4: Sequential Impulse Dynamics Solver & Joints (Days 13–17)
- [ ] Implement `src/dynamics/RigidBody.ts`: Mass, velocity, inertia tensor calculation, sleep state.
- [ ] Implement `src/dynamics/Solver.ts`: Effective mass setup, Baumgarte stabilization, warm starting.
- [ ] Implement Coulomb friction cone with dual orthogonal tangents.
- [ ] Implement Joints: `BallSocketJoint`, `HingeJoint`, `DistanceJoint`.
- [ ] Implement Island Manager and sleeping state machine.

### Phase 5: Three.js Visualizer & Interactive Playground (Days 18–21)
- [ ] Implement Three.js synchronization adapter and wireframe debug overlay renderer.
- [ ] Implement mouse raycast spring grabber (click and fling dynamic bodies).
- [ ] Build the 5 interactive demos (Jenga, Ragdoll, Newton's Cradle, Suspension Vehicle, Demolition).
- [ ] Benchmark suite: Maintain $\ge 60\text{ FPS}$ with 1,000 active colliding bodies in browser.

---

## 11. Verification & Automated Test Matrix

```typescript
describe("Impulse3D Verification Suite", () => {
  test("Momentum Conservation in Elastic Collision");
  test("GJK Correctness: Touching, Separated, and Deeply Intersecting Polytopes");
  test("EPA Accuracy: Normal and Penetration Depth Matches Analytical Box-Box");
  test("Stacking Stability: 20-Box Column Does Not Settle or Drift After 3,000 Ticks");
  test("Zero Energy Gain in Symplectic Euler Orbit");
  test("Island Sleeping: Stationary Pile of 100 Spheres Enters Sleep State Within 1.0s");
  test("Joint Constraint Error Drift Remains < 0.001m Under Continuous Load");
});
```
