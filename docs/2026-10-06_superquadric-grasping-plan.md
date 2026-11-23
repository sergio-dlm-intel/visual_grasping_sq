# Plan: Visual Grasping via Superquadric Abstraction

Created 2026-10-06. Synthesises the sessions of 2026-10-01 to 2026-10-06.

## Goal
A Franka policy that picks any rigid object and carries it to a target, by splitting the problem in two:
- **Perception:** cameras → SuperDec superquadrics.
- **Learning:** PPO on procedurally generated objects that are represented as superquadrics.

## Decisions
- **Robot:** Franka Panda + Franka Hand (80 mm stroke), with a wrist camera.
- **Actions:** same as rl_lab today (ManiSkill `pd_joint_delta_pos`):
  - a normalised action in [-1, 1] per joint, scaled to a joint-position delta added to the *current* joint position (not the previous target), then tracked by a PD controller;
  - 20 Hz control (`control_timestep` = 0.05 s);
  - SO100 today: 6-D, ±0.05 rad for the 5 arm joints and ±0.2 rad for the gripper joint.
  - Franka: 8-D, 7 arm-joint deltas + 1 gripper-opening delta (both fingers mirror each other). On the real robot, send the joint targets through libfranka's joint impedance control.
  - ManiSkill's Panda mode with the same name differs: its gripper takes an absolute position, so it must be overridden to keep the delta gripper.
- **Policy observation:** proprioception, τ_ext, gripper width, goal, an object block, and 16 object tokens.
- **Object block (13-D, as in today's 36-D state):** object pose (7), TCP→object (3), object→goal (3), all referred to the object reference point. It carries the tracked motion explicitly and keeps the relational features that the cube baseline already exploits.
- **Object tokens:** raw SuperDec output, all 16 slots, no filtering, padding or symmetry handling. Per slot: scale (3), ε (2), rotation (9), translation (3), existence probability (1) = 18 numbers. A linear embedding gives one token per slot, followed by a small attention encoder. The network learns the symmetries (bitter-lesson bet: compute and data over built-in knowledge).
  - Pose is kept, refreshed every step: the object is rigid, so each slot's cached object-frame pose is moved by the single tracked object pose and fed in the base frame. Scale, ε and existence stay fixed for the episode. Freezing the poses at their reset values would show the policy where the object *was*.
  - The only requirement is that the variability is in the data: cache several SuperDec decompositions per asset (different poses/views) and sample one per episode.
  - Fallback if generalization is poor: points sampled on the superquadric union.
- **Not observed:** `is_grasped` (it is privileged).
- **Object frame (O):** origin at the volume centroid of the union of superquadrics; axes aligned with the world at reset. Defined the same way in sim and on the real robot, because a mesh's body axes are arbitrary and don't exist outside the simulator.
  - Cache: each slot's pose in O, (R_i^O, t_i^O), stored once from the reset decomposition.
  - Object pose (7) = (p, q): the centroid position, and the rotation since reset (q = identity at reset).
  - Per step, with R = R(q): R_i = R · R_i^O and t_i = R · t_i^O + p. In matrix form, T_i = T_obj · T_i^O.
  - Reward and success use the same centroid. In sim the pose comes from the body pose; at deployment, from the tracker.
- **One language for training and deployment:** each training asset's tokens come from SuperDec itself, run once on dense mesh samples of the asset (a decomposition cache), then moved by the object pose. A perception noise model is applied on top. At deployment the same tokens come from multi-camera SuperDec + pose tracking.
- **Objects:**
  - *Train:* procedural superquadric unions and EGAD-style shapes.
  - *Test (held out):* GSO / YCB.
  - Filter for graspability by the Franka Hand. Randomize mass, friction and centre of mass.
- **Adaptation:** RMA, not Transformer-XL.
  - Phase 1: PPO with privileged physics latent z.
  - Phase 2: estimate z from history (proprioception, τ_ext, actions).
  - Episodes are long enough to regrasp.
- **Reward:** object-agnostic: gripper to nearest surface, contact on both fingers, lift, object to goal, sparse success.
- **Simulator:** Genesis 1.4.3 (passed the step 1 spike, 2026-10-08). Code behind a thin backend interface (batched reset/step → proprio, cameras, ground-truth poses).
- **Compute:** about 8 GPUs (RTX 2080 Ti) on the current server.

## Evidence so far (PickCubeSO100, ManiSkill)
- **Privileged PPO baseline:** 0.961 success (deterministic, 128 evaluation episodes).
- **Perception wrapper:** observations are bit-identical to state mode. On the GPU sim, partial auto-resets perturb the other envs, so offline rollouts must use the same environment setup as evaluation.
- **Cube points per camera frame:** about 20 at 128², about 80 at 256².
- **Direct superquadric fit:** 1.9 mm / 1.6° at 128²; 0.8 mm / 0.5° at 256².
- **SuperDec:**
  - Works only on dense, complete clouds: 0.2 mm, 0.7°, 1 primitive.
  - Fails with fewer than ~1,000 distinct points (10–16 primitives) and on partial views even when dense (3 primitives, IoU 0.10).
  - Not rotation-equivariant: random 3D rotations give 2–7 primitives.

## Step 1 results: Genesis spike (closed 2026-10-08)
**Verdict: pass → migrate to Genesis.** Per-step camera rendering is deferred to step 5.

Setup: Genesis 1.4.3, torch 2.5.1, one RTX 2080 Ti, 64 envs, Franka, 16 procedural objects (8 single superquadrics, 8 unions of 2–3). Script: `rl_lab/scripts/genesis_spike.py`.

| Check | Result |
|---|---|
| Different mesh per env | Pass: 16/16 variants, bounding boxes within 0.3 mm |
| Free-object lift, no drift | Pass with the elliptic friction cone: 100% lifted, rest drift ≤ 0.2 mm, in-hand slip 0.21 mm median / 0.56 mm p95 |
| Per-env physics randomization | Pass: mass, friction (slide distance within ~3% of Coulomb), centre of mass |
| Reproducible resets | Pass: replays bit-identical; partial resets leave the other envs bit-identical (unlike ManiSkill) |
| Depth + segmentation throughput | Fail for per-step use: the rasterizer renders envs one at a time, ~900 frames/s at any env count. Physics: 37k sim steps/s at 64 envs, 111k at 256 |

Findings to carry forward:
- **Contacts:** the default pyramidal friction cone drops 78% of grasps; use `RigidOptions(friction_cone=elliptic)` (~1.5× slower).
- **Gripper:** finger PD with kp 2000, kv 100, force capped at ±20 N. Pure force control drives the fingers through thin objects.
- **Objects:** load meshes with `align=False`, so the centre of mass can be randomized and the origin stays at the volume centroid (frame O). Genesis computes inertia from the convex decomposition (unions up to +13% mass), so set mass explicitly.
- **Object assignment:** each env's object is fixed when the scene is built. Sampling a new object per episode needs envs ≥ objects or scene rebuilds.
- **Rendering budget:** every control step ≈ 50× slower than physics; at reset only ≈ +50%. Only step 5's image-based pose tracking needs per-step rendering, i.e. the batch renderer (`gs-madrona`). It needs CUDA 12.8 libraries: blocked by torch 2.5.1 and driver 560 (CUDA 12.6). Ask for a driver ≥ 570, or test a separate torch ≥ 2.8 venv.
- **Environment:** set `QD_MANYLINUX2014_OK=1` (otherwise importing Genesis breaks `torch._dynamo`). Genesis `main` needs torch ≥ 2.6.

## Step 2 results: Franka cube baseline (closed 2026-10-08)
**Status: done.** Env and protocol frozen, noise floor 0, 3-seed baseline measured at a 500-iteration budget (25.6 M env-steps, ~1.8 h per seed on an RTX 2080 Ti). Full protocol and reference table: `rl_lab/PROTOCOL.md`, section "Step 2".

Setup: ManiSkill `PickCube-v1` (Panda config) ported to Genesis in `rl_lab/envs/genesis_pick_cube.py`, behind the `ManiSkillVectorEnv` contract, so `PPOAgent`, the training loop and the frozen `rl/evaluation.py` run unchanged. Scripts: `train_ppo_genesis.py`, `eval_ppo_genesis.py` (frozen eval, noise-floor repeats, mosaic video), `check_genesis_pick_cube.py`.

- **Actions (concretized):** arm ±0.1 rad per step (ManiSkill's Panda bound), fingers ±0.01 m of half-width per step (matches the finger PD's 20 N cap and 0.2 m/s speed); both added to the current positions.
- **Observation (45-D):** arm q, q̇ (7+7), gripper width, τ_ext / torque limit (7), TCP pose, goal, then the 13-D object block. No `is_grasped`. The cube's rotation is absolute for now; the frame-O convention arrives with the tokens in step 4.
- **Checks:** scripted IK expert through the action space 0.97–0.98 success; `evaluate` bit-identical over 5 repeats and across processes.
- **Baseline (3 seeds, eval at iteration 500):** det success **1.000 ± 0.000** (stoch 0.813 ± 0.023); det steps-to-success **18.10 ± 0.34** (stoch 28.0 ± 0.6). Det success is ≥ 0.99 from 5 M env-steps, so it only flags breakage in later steps; det steps-to-success and the stochastic column are the discriminating metrics.
- **Det/stoch gap = the static condition:** with σ = 0.301 on a 0.1 rad scale, the arm is static (all |q̇| ≤ 0.2 rad/s) on 9.5% of at-goal steps in stochastic mode vs 87% in deterministic mode.
- **Throughput:** ~3.5–4 k env-steps/s per RTX 2080 Ti once the policy grasps (≈ 2 h per seed at the 500-iteration budget, cut from 2 500 once det success saturated); grasp contacts dominate. Multi-contact off would be 2.1× faster but makes a resting cube rock (3.5 mm / 3.2°), so it stays on.
- **Genesis 1.4.3 + torch 2.5.1:** `get_contacts` fails on the zero-copy path (int32 gather index); the env routes it through the kernel path (`gs.use_zerocopy=False` around the call). Net per-link contact forces are not a substitute: they also count finger–table and finger–finger contact (35–135 N with the closed gripper pressed into the table), which can fake a grasp.

## Drill pick-and-place: robust single-object task (planned 2026-10-19)
One real asset before procedural objects. Covers step 3 (token stability on one object), step 4 (decomposition cache) and step 6 (RMA and its ablation). Actions, PPO and the evaluation code are the same as step 2.

### Asset
- **Object:** HANDAL `obj_000024` (cordless drill), 78 × 143 × 219 mm, from `~/data/HANDAL/models` (curated 40-object subset).
- **Held out:** the other drills, `obj_000020–023` and `obj_000025`.
- **Graspable part:** only the handle; the body (78 mm) is as wide as the hand's stroke (80 mm).
- **Mesh:** not watertight, so repair it (e.g. voxel remesh) before computing volume, centroid and inertia. Set the mass explicitly.
- **Origin:** move it to frame O (volume centroid of the superquadric union); load with `align=False`.
- **SuperDec convention:** world z-up, no LM, `normalized` checkpoint, 4096 surface points.
  - Rest pose: 4 primitives, Chamfer-L1 5.9 mm.
  - Orientation, not LM, drives the primitive count: as-is vs z-up gives 3–5 vs 4–9 primitives on the six drills, and z-up with or without LM gives identical counts.
  - Script: `scripts/superdec_handal.py --z_up`.

### Task: place on the table and release
- **Goal:** a target xy on the table, sampled in the spawn area at least 15 cm from the start xy, so the drill must be carried. Observed as (x, y, table z). No orientation goal: any stable resting pose counts. A yaw goal is a later extension.
- **Success:** all of the following, held for 5 consecutive control steps (0.25 s):
  - object centroid within 3 cm of the target in xy;
  - object at rest: |v| ≤ 2 cm/s, |ω| ≤ 0.2 rad/s;
  - no finger–object contact (released);
  - robot static, as in step 2.
- **Episode:** 150 control steps (7.5 s), enough to regrasp. Same termination convention as step 2.
- **Reward:** dense, staged, normalised as in step 2:
  - *reach:* 1 − tanh(5·d), where d is the TCP's radial distance to the nearest superquadric surface, computed from the tokens;
  - *grasp:* `is_grasped` (both fingers);
  - *carry:* xy distance to the target, while grasped, with the object's lowest point ≥ 2 cm above the table;
  - *lower:* once within 3 cm of the target in xy, height towards the resting height;
  - *release:* opening and retreat while the object is on target and at rest;
  - *success:* sparse maximum.
  
  `is_grasped` and contacts are used for the reward but are not observed.
- **Gripper:** finger force cap raised from 20 N to the real hand's 70 N. Without that, 1.5 kg at μ = 0.5 leaves no margin against the torque from a centre of mass 5–10 cm from the handle.

### Reset distribution
- **Resting poses** (`trimesh.poses.compute_stable_poses`, each one settled in Genesis):

  | Pose | Height | Probability |
  |---|---|---|
  | Lying on either side | 78 mm | 0.37 + 0.37 |
  | Front / back | 143 mm | 0.06 + 0.06 + 0.03 |
  | Standing on the battery | 219 mm | 0.045 |

  - Start with the side poses only (top-down handle grasp fits the stroke). Then add every pose that passes a grasp-feasibility check, sampled uniformly over poses (not by probability), so the rare ones are trained.
- **Placement:** yaw uniform, snapped to a 5° grid to match the token cache; xy ±15 cm around (0.615, 0), after checking reach.
- **Physics (privileged z):**

  | Parameter | Range |
  |---|---|
  | Mass | 0.6–2.0 kg, log-uniform |
  | Friction μ | 0.4–1.2 |
  | Centre of mass offset | ±2 cm per axis (uniform density misplaces it; battery and motor concentrate the mass) |
  | Finger force cap | 50–70 N |
- **Object family:** N anisotropic scale variants of the drill (0.9–1.1 per axis), each with its own decomposition. Each env's object is fixed when the scene is built, so N ≤ number of envs.
- **Token noise (perception model, a separate knob):** per episode, scale ±5%, translation ±5 mm, rotation ±5°, ε ±0.1, occasional existence flips. The simulated body stays unchanged.
- **Token cache:** one decomposition per (resting pose, yaw bin, shape variant), computed on the world-frame cloud, so SuperDec's lack of rotation equivariance stays in the data. It doubles as step 3's stability study for this drill (spread of slot count and tokens across yaw).

### Disturbances within an episode
These replace mid-episode mass and friction changes, which don't happen to a rigid object.
- **Pushes:** a random force at the object centroid, 5–20 N for 0.1–0.5 s, plus slow-rising pulls; in about half of the episodes.
- **Gripper force:** piecewise constant within 50–70 N, at most 2 switches per episode.
- **Latency and noise:** action delay of 0–1 control steps (per episode); object-pose noise; dropouts that hold the last value for 1–3 steps.
- **Teacher's privileged z:** mass, μ, centre-of-mass offset, current finger force cap, latency. Pushes are not in z: only history reveals them.

### Stages (each with a go/no-go check)
| Stage | Adds | Go if |
|---|---|---|
| S0 Asset + gripper | Repaired mesh, frame O, convex decomposition of the handle, 70 N fingers, settled resting poses | Scripted handle grasp holds 2 kg at μ = 0.4 with < 5 mm slip over a 3 s carry; no fingers through the object; resting poses drift < 2 mm |
| S1 | Side poses, random yaw and xy, nominal physics, ground-truth tokens + 13-D object block, place-and-release reward | Det success ≥ 0.9 |
| S2 | All feasible resting poses | No pose below 0.7 success |
| S3 | Physics randomization at reset + object family; teacher with z (RMA phase 1) | Full-range success ≥ 0.85 × nominal |
| S4 | Token noise | Report success on held-out drills |
| S5 | Disturbances within the episode | Report drop and recovery rates |
| S6 | RMA student (phase 2): adaptation module from history (proprioception, τ_ext, actions) | Gap to the teacher |
| S7 | Ablation: no physics encoder (below) | Decides whether RMA is kept |

### S7 ablation: what the RMA physics encoder is worth
All arms are trained on the S5 distribution with the same budget, 3 seeds each:

| Arm | Policy input | Deployable |
|---|---|---|
| A Teacher | obs + encoder(z) | No (upper bound) |
| B RMA student | obs + adaptation module(history) | Yes |
| C No encoder (randomization only) | obs | Yes |
| D No encoder + history | obs + the same history window as B, fed straight to the policy | Yes |

- **Why D:** it separates "access to history" from "the RMA structure". If B ≈ D > C, the gain comes from the history, not from the encoder.
- **Metrics:** det/stoch success, steps-to-success, drop rate, recovery after pushes, in-hand slip. Report them per physics bin (mass × μ terciles) and under disturbances.
- **Decision:**
  - Keep RMA only if B − C exceeds 2× the seed std in the hard bins (high mass, low μ) or under disturbances.
  - If B ≈ D, prefer D (simpler).

### Frozen evaluation sets (fixed seeds)
- **Nominal:** side pose, mean physics, no disturbances.
- **Full training distribution.**
- **Held-out drills:** `obj_000020–023`, `obj_000025`.
- **Stress test:** high mass and low μ, with pushes and latency.

### S0 results (closed 2026-10-20)
**Verdict: pass** at a 70 N finger force cap. Torsional friction is adopted for the drill env.

Code (all in `rl_lab`):
- `envs/handal_asset.py`: asset preparation and grasp search.
- `scripts/prepare_handal_asset.py`: writes `assets/handal/obj_000024/` (`mesh.obj`, `asset.json`, `preview.png`).
- `scripts/check_drill_s0.py`: the Genesis checks; results in `logs/drill_s0/`, with mosaic images and video.

| Check | Result |
|---|---|
| Mesh | The HANDAL scan is a polygon soup (1,443 disconnected pieces). A 1.5 mm voxel remesh plus Taubin smoothing gives a watertight 916 cm³ solid with 90k faces. Decimation breaks watertightness, so the full resolution is kept. |
| Frame O | SuperDec on the remesh gives 3 primitives (4 on the raw scan). The union's IoU with the mesh is 0.71. Frame O lies 7.2 mm from the mesh's volume centroid. |
| Collision | CoACD at threshold 0.05 gives 26 hulls: IoU 0.945, hull volume 0.97 × mesh. The grasp width on the hulls is within 0.9 mm of the mesh on the side poses (≤ 1.3 mm on all poses). |
| Resting poses | 8 poses with p ≥ 0.01. 7 stay within 0.4° and 0.2 mm over 2 s. Pose 5 (p = 0.02) rocks 5°, so it is excluded. |
| Grasp, 20 N (step 2's cap) | 2 kg at μ = 0.4: 0/32 lifted. Random physics: 75% lifted. |
| Grasp, 70 N | 64/64 lifted, including 2 kg at μ = 0.4 and its +2 cm centre-of-mass shift along the pads. Carry slip p95 ≤ 1.6 mm; finger penetration ≤ 2.6 mm. One random env (1.86 kg, centre of mass 26 mm off) pivots 42° in the hand without dropping. |
| Grasp, 70 N + torsional friction | 64/64 lifted, worst carry slip 1.6 mm / 1.9°. |

Findings to carry forward:
- **Friction pairing:** Genesis uses the *max* of the two geoms' μ for a contact. The Franka finger geoms have μ = 1, so any object μ below 1 is silently ignored. Give the finger and table geoms ~0 friction (`friction_ratio` 0.01) so the object's μ governs. Step 2's cube never randomized μ, so it is unaffected.
- **Grasps must not wedge:** the first grasp on side pose 1 sat on upward-facing surfaces (contact normal z ≈ +0.5) and squeezed the drill out at any force. The grasp search now requires antipodal pad contacts (within 35° of the closing axis) and summed normal z ≤ 0.1. Reuse this criterion for S2's feasibility check.
- **Finger PD:** kp = F/0.01 and kv = F/0.2, i.e. 7000 / 350 at 70 N. The ±1 cm finger delta then still saturates the cap, and closing speed stays at 0.2 m/s. In step 2's delta action, squeezing was limited to kp·0.01 = 20 N whatever the force range.
- **Torsional friction:** set `RigidOptions(enable_torsional_friction=True)` in the drill env (default 5 mm patch radius). It resists the twist that the centre-of-mass offset along the pads causes. Measure its throughput cost in S1.
- **Nominal centre of mass** (uniform density): (−0.5, −1.8, 7.0) mm in frame O. Scripted grasps reach within 3 mm of the target TCP.
- **License:** the asset is derived from HANDAL (CC BY-NC-SA). Regenerate it with the prepare script rather than redistributing it.

### S1: env and pilot (in progress, 2026-10-22)
Code (all in `rl_lab`):
- `envs/genesis_place_drill.py`: `FrankaPlaceDrillEnv`, which subclasses the cube env (same contract, actions, control and `step`).
- `scripts/check_genesis_place_drill.py`: scripted expert, determinism and throughput checks.
- `train_ppo_genesis.py` / `eval_ppo_genesis.py --task drill --reach {surface,grasp}`; config `PPO_GENESIS_DRILL_PARAMETERS`.

**Env:**
- **Reset:** side pose 0 or 1, yaw on the 5° token-cache grid, xy ±15 cm around (0.615, 0). The goal xy is ≥ 15 cm away, in the same square.
- **Physics:** nominal (1.2 kg, μ 0.8, uniform-density centre of mass), with the S0 settings (70 N cap, torsional friction, ~0 finger and table friction).
- **Episodes:** 150 steps.
- **Observation (333-D):**
  - the 45-D state of step 2, with object pose = frame O position + rotation since reset, and goal = (x, y, frame O's resting height);
  - 16 × 18 tokens: the cached SuperDec decomposition at reset (`tokens.npz`: 8 poses × 72 yaw bins), moved each step by the object's motion since reset.
  - The side poses get 2–4 primitives across yaw (mean ≈ 3).
- **Policy:** a linear embedding per slot, a 2-layer self-attention encoder (d = 64, no slot positions), mean pooling, then the step 2 MLP (`rl/networks.py: TokenMLP`).
- **Success:** frame O within 3 cm of the goal in xy and at its resting height (±1 cm), at rest, no finger contact, arm static, held for 5 steps. Two latches must also be set: the object was lifted ≥ 2 cm while grasped, then set down within 1.5 cm while grasped and on target. Drops and drags don't count.
- **Reward:** staged, normalised by 6.
  - reach (≤ 1) + grasped (1) + carry (≤ 1, scaled by lift height up to 2 cm) + lower (≤ 1, on target);
  - once set down and on target: 4 + released (0.5) + released × arm-still (0.5);
  - success = 6.

**Checks:**
- **Scripted expert:** 0.961 (123/128) through the 8-D action space. The 5 failures are expert timing at the far edge (goal x ≈ 0.75 m): it opened 1.6–2.3 cm above the table, so set-down never latched, as intended. One grasp missed.
- **Determinism:** `evaluate` bit-identical twice. `gu.transform_quat_by_quat` is not bit-deterministic (≈1e-8 run to run on identical inputs), so the env uses its own quaternion product.
- **Throughput:** 16k env-steps/s at 1024 envs under random actions.

**Reach reward, a deviation to test:**
- The plan's object-agnostic reach (TCP to the nearest superquadric surface) also pays for the body. But only ~0.13% of top-down grasp candidates on the side poses are feasible.
- The `grasp` variant instead pays for the distance to the nearest of 512 geometrically feasible grasps, plus closing-axis alignment, all from S0's grasp search. It is automatic for any mesh, but uses privileged grasp geometry.

**Pilot:** reach {surface, grasp} × γ {0.95, 0.98}, seed 42, 1000 iterations (51 M env-steps), GPUs 1–4. Logs in `logs/drill_s1/pilot_*.log`. Then 3 seeds of the best arm.

**Pilot v1 (stopped at iteration 400–500): local optimum, never lets go.**
- All arms learned reach → grasp → lift → carry → set-down: 70–100% lifted, up to 70–100% set down. Set-downs were accurate: 8–11 mm from the target in xy, height within 5 mm, drill still.
- But they never released, so det success was 0–0.03. They held the drill ~5 mm above the target at 75 N, gripper action pinned at "close", for ~115 steps.
- **Cause:** a reward cliff. The placed branch required the drill to be at rest, and opening the fingers lets it settle and wobble for a few steps. During those steps the reward fell back to the non-placed branch, with no grasp term (≈1/6), while hovering paid 3.87/6.
- **Fix (v2):** the reward's placed branch drops the velocity requirement. It now only needs set-down + on target + height within 1 cm. A dense gripper-opening term was added: 4 + 0.5 × opening + 0.5 × released × arm-still. Success stays strict.
- Along the expert's trajectory the reward now rises monotonically through the release: 4.29 → 4.84 → 5.0 → 6 (success). Expert success is unchanged at 0.961.
- The v2 pilot was relaunched on 2026-10-22; the v1 logs are in `logs/drill_s1/pilot_v1/`.

**Pilot v2 (1000 iterations, 2026-10-23): success still ~0, but now because of the success definition.**
- Final checkpoints set the drill down in 62–90% of episodes, 8–11 mm from the target.
- **Grasp, γ 0.95:** lets go, but the arm then keeps drifting. Success needed arm-static (all joints ≤ 0.2 rad/s) ∧ placed ∧ released for 5 consecutive steps; arm-static held only ~12 steps per episode. Without it, the same checkpoint scores **det 0.328 / stoch 0.453** on the frozen protocol (steps-to-success 75 / 82).
- **γ 0.98 arms:** never open. The release bonus was gated on arm stillness, so it was ≈ 0. The gripper action mean saturated at −1.0 to −1.3 (std 0.3), so opening was never explored.
- **Crash:** surface/γ 0.95 crashed with NaN at iteration 337. It was launched before the guard. The guarded grasp/γ 0.98 run (same seed) hit a sim divergence at iteration 65, where its unguarded twin had crashed, logged it and finished. Divergences are rare (1 in 51 M steps).

**v3 (2026-10-26):**
- **Success:** placed (on target, resting height, drill at rest) ∧ released, for 5 steps, after the lift and set-down latches. The arm may move after letting go: it is a place task, and the drill-at-rest condition already rules out pushing.
- **Reward when set on target:** 4 + 0.5 × opening + 1.0 × released; success 6. Expert trace through the release: 4.29 → 5.37 → 5.5 → 6. Expert success 0.961.
- **Guard:** also flags observations with |x| > 1e3.
- **Runs:** 4 pilot arms (seed 42) plus grasp/γ 0.95 seeds 43 and 44, on GPUs 1–6. Logs in `logs/drill_s1/v3_*.log`; v2 logs and videos in `logs/drill_s1/pilot_v2/`.

**v3 results (1000 iterations, 51 M env-steps, 2026-10-27).** No crashes; 1 sim divergence caught by the guard. Analysis, figures and demos are in `logs/drill_s1/analysis/` (`eval_curves.png`, `grasp_locations.png`, `demo_*.mp4`).

| Run | Det / stoch at 1000 | Det, mean of last 3 evals | Steps-to-success (det) | First det ≥ 0.5 |
|---|---|---|---|---|
| surface γ 0.95 s42 | 0.758 / 0.680 | 0.698 | 48.9 | 43.5 M |
| **surface γ 0.98 s42** | **0.867 / 0.758** | **0.849** | 44.9 | 28.2 M |
| **grasp γ 0.95 s42** | **0.867 / 0.758** | 0.807 | 40.8 | 23.0 M |
| grasp γ 0.95 s43 | 0.781 / 0.664 | 0.781 | 42.0 | 33.3 M |
| grasp γ 0.95 s44 | 0.758 / 0.641 | 0.745 | 42.1 | 43.5 M |
| grasp γ 0.98 s42 | 0.016 / 0.055 | 0.026 | – | never |

- **Seeds:** grasp/γ 0.95 over 3 seeds gives det **0.802 ± 0.058**, stoch 0.688 ± 0.062. The binomial SE of one 128-episode eval at p ≈ 0.85 is 0.03, and consecutive evals of one run move by ±0.05–0.1.
- **Learning onset** varies by ~20 M steps across seeds of the same config (23 M vs 43.5 M). Single-seed comparisons of speed or final score are therefore not conclusive.
- **Reach reward: no evidence that the privileged `grasp` reach helps.**
  - At γ 0.95: grasp 0.80 ± 0.06 (3 seeds) vs surface 0.76 (1 seed), indistinguishable.
  - At γ 0.98: surface 0.87 vs grasp 0.02.
  - Every policy grasps within a median 5–8 mm (axis 12–18°) of S0's feasible set, in the same three families: handle, battery edge and the top edge of the motor housing. None converges on S0's single "best" grasp (27–82 mm away).
  - Reach is 1 of the 6 reward points; grasp, carry and success are judged by physics in both arms. The surface arm, guided only by the observed tokens, finds the same feasible grasps.
- **grasp/γ 0.98** fails as in v2: it sets down (56/128) but does not release, or loses the grasp before lifting (41/128). It has the most contact flicker (5.7 grasp onsets per lifted episode, vs 2–3). It is one seed, so this can't be attributed to the config.
- **Failure modes of the good runs** (128 frozen-protocol placements):
  - grasped but not lifted: 4–13;
  - lifted but never set down (drop, or not placed accurately): 7–26 (26 for surface γ 0.95);
  - release failures: ≤ 5.
  - Both side poses succeed about equally (0.72–0.93).
- **Timing (median step):** grasp 10–14, lift 17–19, set down 25–29, success 36–42, so the carry takes ~0.5 s. Motions are fast (the arm delta saturates at 2 rad/s), and the drill is carried low: the reward only asks for 2 cm of clearance.
- **S1 go criterion (det ≥ 0.9): not met yet.** Best is 0.867, and most curves are still rising at 51 M steps.
- **Next:**
  1. Seeds 43 and 44 for surface/γ 0.98 (and optionally grasp/γ 0.98), to decide the reach × γ question.
  2. Continue the best runs to 2000 iterations.
  3. If surface holds up, adopt it (object-agnostic, as the plan intended) and drop the privileged grasp set.
- **Seed runs launched 2026-10-30:** surface/γ 0.98 and grasp/γ 0.98, seeds 43 and 44 (same v3 env and config), on GPUs 1–4. Logs in `logs/drill_s1/v3_{surface,grasp}_g0.98_s{43,44}.log`.

## Main risks
1. **SuperDec's decomposition is unstable** across poses, views and coverage → the policy's input flickers.
2. **Partial clouds at deployment:** the face on the table and the gripper during the grasp are never fully seen.
3. **Genesis:** per-step rendering throughput (see step 1 results). Heterogeneous meshes and grasp stability were verified in step 1.
4. **Adaptation may buy little** with a stiff arm. Measure the gap before building it.
5. **Sample cost** grows a lot with object diversity.

## Steps (each with a go/no-go check)
1. **Genesis spike, 1–2 weeks.** Check:
   - a different mesh per environment;
   - a free-object lift with no drift;
   - batched multi-camera depth + segmentation throughput;
   - per-environment physics randomization;
   - reproducible resets.
   
   Pass → migrate. Fail → ManiSkill (native Panda, `PickSingleYCB`). **Done: pass (2026-10-08).**
2. **Franka cube baseline** with ground-truth state; re-freeze the evaluation protocol. **Done (2026-10-08): det success 1.000, det steps-to-success 18.1 ± 0.3.**

   **Next: drill pick-and-place** (see its section: stages S0–S7), a single-asset pass through steps 3, 4 and 6.
3. **SuperDec stability study:** same object × poses × camera layouts (fixed + wrist). If unstable → fine-tune SuperDec with rotation/partiality augmentation (sparse visible clouds as input, full mesh as target), or canonicalize the cloud first.
4. **Procedural objects + decomposition cache,** ground-truth pose. Measure generalization to held-out procedural shapes and to GSO/YCB.
5. **Perception in the loop:** noise model, then SuperDec on simulated cameras, then pose tracking. Image-based tracking in the loop needs the batch renderer (or tracking on a few eval envs only).
6. **RMA:** compare oracle-z vs domain-randomization-only. Build the adaptation module only if the gap is significant.
7. **Real Franka.**
