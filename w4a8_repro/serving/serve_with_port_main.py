#!/usr/bin/env python3
"""vllm serve launcher with the dense W4A8_DYNAMIC linear scheme pre-registered.

Why a wrapper: vllm-ascend 0.30.0rc2 only registers W4A8_DYNAMIC for MoE
layers, so msmodelslim dense W4A8 exports need the runtime scheme from
/home/w30022218/w4a8_tuning/scripts/w4a8_main_port.py to load.

Registration propagation across the serve process tree (supervisor, per-DP
API-server subprocesses, multiprocessing-spawned engine workers):
  1. this wrapper imports it directly (covers the supervisor and any spawn
     child that re-imports this file as __mp_main__);
  2. run_ceval.sh additionally puts ceval/bootstrap on PYTHONPATH, whose
     sitecustomize.py re-imports it in EVERY interpreter of the tree
     (vllm 0.30.0 spawns API servers as plain subprocesses, which do NOT
     re-import the launcher -- the sitecustomize path is what covers them).

Usage: python3 serve_with_port.py serve <model> [vllm flags...]
"""
import sys

SCRIPTS = "/home/w30022218/w4a8_tuning/scripts"
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import w4a8_main_port  # noqa: F401,E402  (registers W4A8_DYNAMIC for linear)

from vllm.entrypoints.cli.main import main  # noqa: E402

if __name__ == "__main__":
    main()
