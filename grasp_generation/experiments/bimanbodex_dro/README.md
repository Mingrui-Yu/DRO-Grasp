# BimanBODex DGN2k adapter

This experiment adapter runs the released complete-point-cloud DRO-Grasp model
on the same ShadowHand DGN2k scene contract used by the project baselines. It
does not train or modify the network or invoke the official Isaac Gym evaluator.
It supports a strict initialization ablation between the released random root
orientation and deterministic tabletop-oriented proposals. The default
`unfiltered_baseline` production mode retains the original no-rejection
denominator. The separate `tabletop_filtered` production mode performs an
explicit final-grasp table-plane rejection after generation; it does not add a
collision loss or execution-path check.

## Fixed contracts

- Source baseline: upstream commit `07590bd8aeb074671e0d133fb372027f46fbd5f3`.
- Checkpoint: release `v1.0` `ckpt/model/model_3robots.pth`; the partial model is
  rejected by config validation.
- Input: 512 deterministic complete surface points sampled from the exact
  scaled `mesh/simplified.obj` in object-local metres. The adapter does not
  normalize, recenter, rescale, simplify, or substitute assets.
- The released network's official preprocessing is retained: only the robot
  point cloud is zero-centered inside the network (`network_center_robot_pc`);
  object points stay in the exact object-local frame and are not normalized.
- Scene export: `T_WH = T_WO @ T_OPalm`. For every candidate and controller
  stage, `T_OPalm` is evaluated by the actual released
  `pytorch_kinematics` chain as `FK_DRO(q)["palm"]`. It includes the floating
  forearm pose plus `WRJ2/WRJ1`; these wrist joints are not silently dropped
  into the Bench finger vector. Production export does not use a hand-written
  wrist or palm offset.
- Scene input must explicitly contain the `table` plane. Its pose and local
  normal are resolved in world coordinates and persisted in v2 scene
  provenance; the adapter never infers a table from object bounds or names.
- Initialization modes:
  - `released_random` retains `HandModel.get_initial_q()` exactly.
  - `tabletop_stratified` first calls that same released sampler, then replaces
    only `q[3:6]`. The default ordered allocation is 6 `top_down` proposals at
    60--90 degrees, 8 `oblique` proposals at 20--60 degrees, and 6
    `near_horizontal` proposals at 0--20 degrees. Elevation is measured from
    the table plane; elevation, azimuth, and roll use deterministic midpoint
    strata recorded in the resolved config.
- The numerically validated DRO/Bench palm-local approach axis is `+Y` with
  `object_to_palm` sign semantics. This is a palm-side direction, not the
  execution velocity. Tabletop directions are built in world coordinates,
  transformed by `R_WO^T`, and applied to the actual palm after compensating
  the unchanged `WRJ2`/`WRJ1` wrist rotation.
- Hand mapping: the released URDF and Bench MJCF share right-hand finger joint
  semantics. Export adds the `rh_` namespace and reorders by name. The official
  raw controller stages are retained unchanged; an export-only copy clamps
  finger joints to the intersection of the released DRO URDF and Bench limits,
  with every changed value recorded in raw diagnostics. A numerical validator
  checks joint limits and palm-local FK landmarks.
- Stages: the optimized `predict_q` remains the grasp state. The official
  `controller(predict_q)` output is exported as `q_outer -> pregrasp`,
  `predict_q -> grasp`, and `q_inner -> squeeze`; `isaac_q` is not substituted
  for any of these three stages. Palm FK is evaluated separately for all three.
- Baseline budget: `unfiltered_baseline` generates exactly 20 raw candidates
  per scene. Candidate seeds and ordering are stable and recorded. Both
  initialization modes consume the released sampler before any override, so
  root translation and `q[6:]` match for each paired candidate. The CPU and
  applicable CUDA Torch RNG-state digests are captured immediately before
  network forward to verify identical validation latent state.
- Filtered production budget: `tabletop_filtered` generates exactly one complete
  20-candidate initialization group per scene. The hard generated-candidate cap
  is therefore 20 per scene. Every candidate that passes the final-pose table
  filter is returned, so completed artifacts contain between zero and 20 grasps.
- Final-pose table filter: only the export-clamped `grasp_qpos` corresponding to
  `predict_q` is checked. The model parses the released URDF `collision`
  geometry and selects `palm` plus all kinematic descendants; `forearm`,
  `wrist`, and every other palm ancestor are excluded. Signed height is measured
  against the explicit scene table plane with `margin = 0`, and every included
  geometry must satisfy strict `min_z > 0`. `pregrasp`, `squeeze`, interpolation,
  approach, arm paths, object collision, and self-collision are not checked.
- Filtered selection: because the generation cap equals the 20-candidate target,
  no valid candidate is downsampled. All passing candidates are retained in stable
  generation order. No score, top-k, quality weighting, diversity ranking,
  deduplication, padding, or manual choice is used. Source batch/candidate/proposal
  indices and generation and selection Seeds remain persisted for provenance.
- Partial-success policy: baseline outputs remain scene-atomic under their
  original 20-candidate policy. Filtered production records individual inference
  failures and continues accumulating other candidates. Structural scene,
  inference, filter, FK, or export failures still fail the scene. Reaching the
  20-candidate budget below 20 valid grasps is instead a normal `partial` or
  `empty` result: every valid grasp is returned without relaxing the filter,
  duplicating grasps, or padding the artifact.

The Bench-facing keys, joint order, and stage semantics remain unchanged, while
the candidate axis becomes variable-length for filtered v4 outputs:

```python
{
    "robot_pose": np.ndarray,  # float32 [1, N, 3, 29], 0 <= N <= 20
    "joint_names": [...],      # Bench rh_* order
    "scene_path": ["src/curobo/content/assets/object/DGN_2k/scene_cfg/..."],
}
```

DRO-specific provenance is stored only in `raw/`, `run_manifest.json`,
`failure_manifest.json`, and `resolved_config.json`.

Each successful raw artifact stores `stage_q` as the untouched official
`q_outer/q/q_inner` result and `export_stage_q` as the Bench-facing clamped
copy. `export_clamp_diagnostics` records the candidate, stage, joint, raw and
clamped values, delta, and both source limit intervals for every clamp.
Unfiltered runs use `drograsp.dgn2k.raw.v2` / `drograsp.dgn2k.run.v2` and
additionally
store `released_initial_q`, effective `initial_q`, initialization metadata, the
explicit table contract, pre-network RNG digests, and `palm_fk` provenance
(`backend`, link, joint order, URDF SHA256, dtype, and device). Validators and
the viewer retain read-only support for #24 artifacts without `palm_fk` by
using the old hand-written formula only as an explicit historical compatibility
path; new writes always use actual PK FK. Historical filtered production uses
`drograsp.dgn2k.raw.v3` / `drograsp.dgn2k.run.v3` and remains read-only
compatible. New partial-success production uses `drograsp.dgn2k.raw.v4` /
`drograsp.dgn2k.run.v4`: selected fields use the persisted returned count `N`,
while `generation_*`, `batch_summaries`, filter diagnostics, selection indices,
shortfall accounting, and source provenance retain all generated candidates.

## Assets and environment

Release assets are intentionally ignored by Git. The verified local release
records for Issue #24 are:

| Asset | Size | SHA256 |
| --- | ---: | --- |
| `ckpt-v1.0.zip` | 274728721 | `8f49d15361b50b43fdd030888a4d88b6c1280990f3543a0c7272395ed09e06f9` |
| `data-v1.0.zip` | 994807865 | `0fbdd0fb6aeb8e0dbc0eca4cfff019e0e4317aa51c892b3f29bf5ade9f341e40` |
| `ckpt/model/model_3robots.pth` | 56209306 | `997918656839ebd3e2641b598c999d856dc90035692bacdfada24bdb35333db9` |

Create the isolated environment only when environment installation is
authorized:

```bash
conda env create -f environment_dgn2k.yml
```

## Validation and use

CPU contracts:

```bash
PYTHONPATH=. python -m unittest discover \
  -s grasp_generation/tests -p 'test_bimanbodex_dro*.py' -v
```

Shadow mapping against the unchanged Bench asset:

```bash
python grasp_generation/scripts/validate_bimanbodex_dro_shadow_mapping.py \
  --dro-urdf data/data_urdf/robot/shadowhand/shadow_hand_right_extended.urdf \
  --bench-mjcf ../BimanDexGraspBench/assets/hand/shadow/right_hand_v2.xml \
  --bench-hand-config ../BimanDexGraspBench/config/hand/shadow.yaml \
  --samples 128
```

The default acceptance threshold is `0.5 mm` maximum common-link origin error.
Link rotations are compared after a fixed zero-pose frame alignment because
the URDF link and MJCF body frames can use different constant orientations.

The checked-in `config.json` uses the read-only Heur-Fix reference root
`../BimanBODex/src/curobo/content/assets/output/sim_shadow/tabletop_full/
single_type_DGN2k_1000/graspdata`. It resolves 996 scenes (787 objects) across
the complete scale range 0.02--0.30. The config pins the ordered
scene/mesh/scale/pose/table provenance digest
`1f6ef04c5e2234bd54edd6a0075883d6f16909af3b6e65a38ae8141934621030`;
an incomplete or changed reference set is rejected before inference. A
different approved scene revision must update the config and version evidence.

A read-only contract dry-run validates the entire authoritative set, samples
one representative complete point cloud, and does not create an output
directory:

```bash
python grasp_generation/scripts/generate_bimanbodex_dro.py \
  --config grasp_generation/experiments/bimanbodex_dro/config.json \
  --initialization-mode tabletop_stratified \
  --dry-run
```

The checked-in config defaults to `unfiltered_baseline`. A filtered dry-run must
name the production mode and the run-specific safety cap explicitly. It parses
all palm-scope collision assets but does not start inference or create an output
root:

```bash
python grasp_generation/scripts/generate_bimanbodex_dro.py \
  --config grasp_generation/experiments/bimanbodex_dro/config.json \
  --initialization-mode tabletop_stratified \
  --production-mode tabletop_filtered \
  --max-batches 1 \
  --selection-seed 240826 \
  --dry-run
```

After GPU authorization, use a new output root:

```bash
DRO_PYTHON="${DRO_PYTHON:-../.conda-envs/dro/bin/python}"
"$DRO_PYTHON" grasp_generation/scripts/generate_bimanbodex_dro.py \
  --config grasp_generation/experiments/bimanbodex_dro/config.json \
  --initialization-mode tabletop_stratified \
  --max-scenes 1 \
  --output-root /path/to/new/issue31-tabletop_stratified-model_3robots-seed240825
```

The one-scene command is only the bounded GPU smoke. Remove `--max-scenes 1`
only after separate approval for the formal 996-scene run.

After separate GPU/output authorization, the corresponding bounded filtered
smoke uses a new output root:

```bash
DRO_PYTHON="${DRO_PYTHON:-../.conda-envs/dro/bin/python}"
"$DRO_PYTHON" grasp_generation/scripts/generate_bimanbodex_dro.py \
  --config grasp_generation/experiments/bimanbodex_dro/config.json \
  --initialization-mode tabletop_stratified \
  --production-mode tabletop_filtered \
  --max-batches 1 \
  --selection-seed 240826 \
  --max-scenes 1 \
  --output-root /path/to/new/issue42-tabletop-filtered-seed240826
```

Revalidate the persisted raw/artifact contract independently:

```bash
python grasp_generation/scripts/validate_bimanbodex_dro_outputs.py \
  /path/to/new/dro-model_3robots-seed240825
```

For the formal ablation, run both modes from the same source/config into two new
output roots, then validate the pair. The paired validator requires identical
scenes, point clouds, candidate seeds, released q, root translations, wrist and
finger q, and pre-network RNG digests; all 20 effective root rotations must be
replaced only in the tabletop run:

```bash
python grasp_generation/scripts/validate_bimanbodex_dro_pair.py \
  --released-root /path/to/new/issue31-released_random-model_3robots-seed240825 \
  --tabletop-root /path/to/new/issue31-tabletop_stratified-model_3robots-seed240825
```

## Exported three-pose Viser viewer

The viewer is a read-only CPU inspection tool for an existing synthesis output
root. It does not run DRO inference, the controller, optimization, Isaac Gym, or
Bench evaluation, and it never writes a cache, screenshot, or modified artifact.

Start it in the isolated `dro` environment:

```bash
DRO_PYTHON="${DRO_PYTHON:-../.conda-envs/dro/bin/python}"
"$DRO_PYTHON" grasp_generation/scripts/visualize_bimanbodex_dro.py \
  --output-root /path/to/read-only/dro-model_3robots-seed240825 \
  --scene-root /path/to/BimanBODex/src/curobo/content/assets/object/DGN_2k/scene_cfg \
  --shadow-urdf data/data_urdf/robot/shadowhand/shadow_hand_right_extended.urdf \
  --mode three_poses \
  --host 127.0.0.1 \
  --port 8080
```

If the persisted absolute scene or URDF path is still valid, `--scene-root` and
`--shadow-urdf` may be omitted. An override is accepted only when the exact
scene/mesh provenance and persisted Shadow URDF SHA256 still match; the viewer
does not fall back to release CMapDataset objects or a default hand pose.

For a remote server, keep Viser on loopback and forward the port from the local
machine:

```bash
ssh -N -L 8080:127.0.0.1:8080 user@server
```

Then open `http://127.0.0.1:8080`. Choose the successful scene, isotropic object
scale, candidate, stage, and either one stage or the three-pose overlay. The
fixed colors are orange for `pregrasp`, blue for `grasp`, and pink for
`squeeze`; the selected stage is more opaque in overlay mode. World, object,
and palm axes and the persisted 512-point input cloud can be toggled separately.

To compare the producer URDF with the exact palm-root Bench asset, add the Bench
MJCF and select `dro_bench_overlay`:

```bash
DRO_PYTHON="${DRO_PYTHON:-../.conda-envs/dro/bin/python}"
"$DRO_PYTHON" grasp_generation/scripts/visualize_bimanbodex_dro.py \
  --output-root /path/to/read-only/dro-model_3robots-seed240825 \
  --bench-mjcf ../BimanDexGraspBench/assets/hand/shadow/right_hand_v2.xml \
  --candidate 0 --stage grasp --mode dro_bench_overlay
```

This mode renders the same scaled/world-posed DGN2k object, the complete DRO
URDF, and the Bench group-2 visual meshes rooted at the exported `rh_palm`
pose. DRO forearm, wrist, and palm/fingers can be hidden independently. Palm
and common-link axes can be toggled, and diagnostics report maximum position
and rotation error plus the worst link. The overlay is a frame/asset contract
check, not grasp success, stability, or table-collision evidence.

The default pose source is `exported`, which reconstructs the actual
`export_stage_q`/`robot_pose` Bench artifact. `raw` shows the untouched official
controller state only as a clearly labelled diagnostic. Clamp count and maximum
absolute delta refer to the export-only DRO/Bench joint-limit intersection; they
are not optimization iterations, contact forces, or physical success evidence.

Before opening a port, the same loader/render preparation can be checked without
starting Viser:

```bash
DRO_PYTHON="${DRO_PYTHON:-../.conda-envs/dro/bin/python}"
"$DRO_PYTHON" grasp_generation/scripts/visualize_bimanbodex_dro.py \
  --output-root /path/to/read-only/dro-model_3robots-seed240825 \
  --scene-root /path/to/DGN_2k/scene_cfg \
  --candidate 0 --stage grasp --mode three_poses --prepare-only
```

The loader requires the terminal `run_manifest.json`, matching
`resolved_config.json` and `failure_manifest.json`, every completed
`graspdata/`/`raw/` pair, exact scene/mesh hashes, `wxyz` object pose, isotropic
scale, stage and joint order, finite `float32` arrays, approved clamp result,
and `export_stage_q` to persisted `robot_pose` round-trip within `1e-6`
absolute tolerance for GPU/CPU float32 FK reproducibility. Failed scenes
are reported in diagnostics but are never offered as renderable grasps.

All geometry is constructed on CPU in the saved world frame using
`T_WH = T_WO @ T_OPalm`. The complete released Shadow URDF includes the floating
root, `WRJ2`/`WRJ1`, and all 22 finger joints. The object mesh is loaded from the
same `processed_data/<object_id>/mesh/simplified.obj`, scaled and posed exactly;
it is not recentered, normalized, or resampled. The viewer does not render or
evaluate table contact: the absence of a visible collision must not be
interpreted as tabletop success or collision-free generation.

The current Bench tabletop protocol is the only evaluation target. The network
still does not receive table points and the adapter adds no table-collision
objective. Initialization ablations must continue to use `unfiltered_baseline`
and may only be described as tabletop-oriented/table-conditioned initialization.
`tabletop_filtered` results may be described only as passing the final
palm-and-descendants table-plane filter. They are not evidence of collision-free
approach paths, object/self-collision freedom, simulation success, stability, or
real-robot executability.

## Issue #47 bounded three-scale comparison

The Issue #47 config resolves exactly three bottle-like DGN2k tabletop scenes:

- `scale002`, persisted actual scale `0.02`;
- `scale011`, persisted actual scale `0.106` (the dataset bucket is not exact `0.11`);
- `scale030`, persisted actual scale `0.30`.

It intentionally clears `reference_grasp_roots`, uses the checked-in scene list,
and pins the three-scene manifest hash. Its filtered budget is one complete
20-candidate batch, so each scene can never generate more than 20 candidates.
A scene that remains below 20 returns all valid grasps, including an
explicit zero-length artifact when no candidate passes.
Per-candidate timings are printed during official inference, and
`progress_manifest.json` is atomically updated after every completed batch.
Dry-run the exact bounded contract before starting CUDA inference:

```bash
DRO_PYTHON="${DRO_PYTHON:-../.conda-envs/dro/bin/python}"
"$DRO_PYTHON" grasp_generation/scripts/generate_bimanbodex_dro.py \
  --config grasp_generation/experiments/bimanbodex_dro/config_issue47_three_scale.json \
  --dry-run
```

After confirming a new output path and CUDA device mapping, run the filtered
production without changing the 20-candidate target:

```bash
CUDA_VISIBLE_DEVICES=<PHYSICAL_GPU> "$DRO_PYTHON" \
  grasp_generation/scripts/generate_bimanbodex_dro.py \
  --config grasp_generation/experiments/bimanbodex_dro/config_issue47_three_scale.json \
  --output-root output/issue47-three-scale-tabletop-filtered-<RUN_ID>
```

Validate all terminal manifests and artifacts before visualization:

```bash
"$DRO_PYTHON" grasp_generation/scripts/validate_bimanbodex_dro_outputs.py \
  output/issue47-three-scale-tabletop-filtered-<RUN_ID>
```

The comparison viewer uses one output root and exactly three repeated `--scene`
arguments in the desired left-to-right order. It keeps all persisted geometry in
metres and applies only centered display translations. Candidate sliders are
independent; the default Web view shows exported `grasp` only, with optional
pregrasp, squeeze, point-cloud, table, and coordinate-axis controls.

```bash
"$DRO_PYTHON" grasp_generation/scripts/visualize_bimanbodex_dro_comparison.py \
  --output-root output/issue47-three-scale-tabletop-filtered-<RUN_ID> \
  --scene core_bottle_134c723696216addedee8d59893c8633/tabletop_ur10e/scale002_pose000_0 \
  --scene sem_Bottle_9afea0432f292379dc0e610397fef7f9/tabletop_ur10e/scale011_pose000_0 \
  --scene core_bottle_eded10bf44a2571a911cff0cb398f845/tabletop_ur10e/scale030_pose000_0 \
  --host 127.0.0.1 --port 8080
```

Use `--prepare-only` for a read-only CPU contract check before opening the port.
The finite table patch is only a visualization of the authoritative infinite
plane; it is not a collision boundary. The viewer does not run Bench dynamics,
rank candidates, or establish physical grasp success.
