#!/usr/bin/env python3
"""Revalidate a persisted DRO DGN2k run without running inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GRASP_GENERATION_ROOT = Path(__file__).resolve().parents[1]
if str(GRASP_GENERATION_ROOT) not in sys.path:
    sys.path.insert(0, str(GRASP_GENERATION_ROOT))

from experiments.bimanbodex_dro.runner import validate_run_outputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_run_outputs(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
