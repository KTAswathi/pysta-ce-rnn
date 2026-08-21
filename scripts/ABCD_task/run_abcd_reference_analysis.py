"""Run the frozen ABCD reference analyses for one managed-run basename.

This user-facing entry point is intentionally thin.  It selects no scientific
estimator: the four existing analysis programs remain authoritative and are
invoked, in order, by the sibling orchestration module.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence


if __package__:
    from . import abcd_reference_analysis_orchestrator as _orchestrator
else:  # Direct ``python scripts/ABCD_task/run_....py`` execution.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import abcd_reference_analysis_orchestrator as _orchestrator


REPO_ROOT = Path(__file__).resolve().parents[2]

# Small public surface retained for lightweight path/orchestration tests and
# callers that previously imported the new wrapper while it was being built.
PipelinePaths = _orchestrator.PipelinePaths
PIPELINE_STATUS_NAME = _orchestrator.PIPELINE_STATUS_NAME
STAGE_ORDER = _orchestrator.STAGE_ORDER
STAGE_SOURCE_FILES = _orchestrator.STAGE_SOURCE_FILES
resolve_managed_run = _orchestrator.resolve_managed_run
resolve_run_folder = _orchestrator.resolve_run_folder
build_stage_commands = _orchestrator.build_stage_commands
stage_status = _orchestrator.stage_status
run_pipeline = _orchestrator.run_pipeline


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate CLI parsing and execution to the non-scientific orchestrator."""

    return _orchestrator.main(
        argv,
        repo_root=REPO_ROOT,
        runner=subprocess.run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
