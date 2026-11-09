"""Interpreter-startup bootstrap for the C-Eval serving process tree.

`vllm serve` (vLLM 0.30.0) launches per-DP API servers as subprocesses and
engine workers via multiprocessing; a quantization scheme registered only in
the launcher process does not reach them. run_ceval.sh puts this directory on
PYTHONPATH for the serve command; Python then auto-imports sitecustomize at
startup in every process of the tree, each time registering the dense
W4A8_DYNAMIC linear scheme (see /home/w30022218/w4a8_tuning/scripts/
w4a8_dense_port.py, found via the scripts dir also being on PYTHONPATH).

Never let a bootstrap failure break interpreter startup.
"""
try:
    import w4a8_dense_port  # noqa: F401
except Exception:  # pragma: no cover
    pass
