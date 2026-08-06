#!/usr/bin/env python3
"""Validate strict pairing between released-random and tabletop DRO runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GRASP_GENERATION_ROOT = Path(__file__).resolve().parents[1]
if str(GRASP_GENERATION_ROOT) not in sys.path:
    sys.path.insert(0, str(GRASP_GENERATION_ROOT))

from experiments.bimanbodex_dro.pairing import validate_paired_outputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--released-root", type=Path, required=True)
    parser.add_argument("--tabletop-root", type=Path, required=True)
    args = parser.parse_args()
    result = validate_paired_outputs(args.released_root, args.tabletop_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
