from src.sdk.models import WorkflowConfig

BUG_FIX_CONFIG = WorkflowConfig(
    has_result=True,
    result_criteria="Bug is fixed and verified: cannot be reproduced, tests pass",
    on_result_found="stop_all"
)