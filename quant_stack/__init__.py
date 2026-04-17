"""HydraPredict 5 core package."""

import os
import sys

# Windows-specific runtime defaults:
# - avoid loky physical-core probe warnings in constrained environments
# - avoid known sklearn+MKL KMeans leak warnings when threads exceed chunks
if sys.platform.startswith("win"):
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

from quant_stack.module5 import run_backtest
from quant_stack.module6 import run_hpo_study
from quant_stack.module10 import run_health_check
from quant_stack.run_shadow_trade import run_shadow_trade

__all__ = ["run_backtest", "run_hpo_study", "run_health_check", "run_shadow_trade"]
