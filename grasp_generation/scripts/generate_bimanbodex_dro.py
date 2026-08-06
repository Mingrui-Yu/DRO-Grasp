#!/usr/bin/env python3
"""Run the approved DRO-Grasp Shadow/DGN2k adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GRASP_GENERATION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = GRASP_GENERATION_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GRASP_GENERATION_ROOT) not in sys.path:
    sys.path.insert(0, str(GRASP_GENERATION_ROOT))

from experiments.bimanbodex_dro.initialization import INITIALIZATION_MODES
from experiments.bimanbodex_dro.runner import dry_run, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--scene-list", type=Path)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--initialization-mode", choices=INITIALIZATION_MODES)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.output_root is not None:
        config["output_root"] = str(args.output_root)
    if args.scene_list is not None:
        config["scene_list"] = str(args.scene_list)
    if args.max_scenes is not None:
        config["max_scenes"] = args.max_scenes
    if args.initialization_mode is not None:
        config.setdefault("initialization", {})["mode"] = args.initialization_mode

    result = dry_run(repo_root, config) if args.dry_run else run(repo_root, config)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
