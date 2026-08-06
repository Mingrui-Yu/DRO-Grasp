# BimanBODex DGN2k adapter

This experiment adapter runs the released complete-point-cloud DRO-Grasp model
on the same ShadowHand DGN2k scene contract used by the project baselines. It
does not train or modify the network, add table-aware proposals, or invoke the
official Isaac Gym evaluator.

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
- Scene export: `T_WH = T_WO @ T_OPalm`. `T_OPalm` includes the released
  floating forearm pose plus `WRJ2/WRJ1`; these wrist joints are not silently
  dropped into the Bench finger vector.
- Hand mapping: the released URDF and Bench MJCF share right-hand finger joint
  semantics. Export adds the `rh_` namespace and reorders by name. A numerical
  validator checks joint limits and palm-local FK landmarks.
- Stages: the official `controller()` output is exported as
  `q_outer -> pregrasp`, optimized `q -> grasp`, and `q_inner -> squeeze`.
- Budget: exactly 20 raw candidates per scene. Candidate seeds and ordering are
  stable and recorded.
- Failure policy: outputs are scene-atomic. If any candidate fails inference or
  export validation, no normal Bench artifact is written for that scene and all
  20 candidates are conservatively counted as failed. Partial raw diagnostics
  remain under `failed_raw/`; candidates are never resampled.

The Bench-facing artifact remains unchanged:

```python
{
    "robot_pose": np.ndarray,  # float32 [1, 20, 3, 29]
    "joint_names": [...],      # Bench rh_* order
    "scene_path": ["src/curobo/content/assets/object/DGN_2k/scene_cfg/..."],
}
```

DRO-specific provenance is stored only in `raw/`, `run_manifest.json`,
`failure_manifest.json`, and `resolved_config.json`.

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
  --samples 128
```

The checked-in `config.json` uses the read-only Heur-Fix reference root
`../BimanBODex/src/curobo/content/assets/output/sim_shadow/tabletop_full/
single_type_DGN2k_1000/graspdata`. It resolves 996 scenes (787 objects) across
the complete scale range 0.02--0.30. The config pins the ordered
scene/mesh/scale/pose provenance digest
`061d9305037b86bffed6732954a47126bbdb01ee476b12b9b8427600723a595e`;
an incomplete or changed reference set is rejected before inference. A
different approved scene revision must update the config and version evidence.

A read-only contract dry-run validates the entire authoritative set, samples
one representative complete point cloud, and does not create an output
directory:

```bash
python grasp_generation/scripts/generate_bimanbodex_dro.py \
  --config grasp_generation/experiments/bimanbodex_dro/config.json \
  --dry-run
```

After GPU authorization, use a new output root:

```bash
conda run -n dro python grasp_generation/scripts/generate_bimanbodex_dro.py \
  --config grasp_generation/experiments/bimanbodex_dro/config.json \
  --max-scenes 1 \
  --output-root /path/to/new/dro-model_3robots-seed240825
```

The one-scene command is only the bounded GPU smoke. Remove `--max-scenes 1`
only after separate approval for the formal 996-scene run.

Revalidate the persisted raw/artifact contract independently:

```bash
python grasp_generation/scripts/validate_bimanbodex_dro_outputs.py \
  /path/to/new/dro-model_3robots-seed240825
```

The current Bench tabletop protocol is the only evaluation target. Because DRO
does not receive the table and this adapter adds no table-collision objective or
post-filter, results may only be described as performance under that protocol,
not as native table-aware or collision-free generation.
