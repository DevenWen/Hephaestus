from src.sdk.models import Phase

PHASE_1_REPRODUCTION = Phase(
    id=1,
    name="bug_reproduction",
    description="Reproduce the reported bug and capture evidence",
    done_definitions=[
        "Bug reproduced successfully",
        "Reproduction steps documented",
        "Error logs captured",
        "Phase 2 investigation task created",
        "Task marked as done"
    ],
    working_directory=".",
    additional_notes="""
    🎯 YOUR MISSION: Confirm the bug exists

    STEP 1: Read the bug report in your task description
    STEP 2: Follow the reproduction steps
    STEP 3: Capture error messages and logs
    STEP 4: If bug confirmed: Create Phase 2 task
    STEP 5: Mark your task as done

    ✅ GOOD: "Bug reproduced. Error: 'Cannot read property of undefined' at login.js:47"
    ❌ BAD: "It crashes sometimes"
    """
)

PHASE_2_INVESTIGATION = Phase(
    id=2,
    name="root_cause_analysis",
    description="Find the root cause of the bug",
    done_definitions=[
        "Root cause identified",
        "Affected code located",
        "Fix approach proposed",
        "Phase 3 implementation task created",
        "Task marked as done"
    ],
    working_directory=".",
    additional_notes="""
    🎯 YOUR MISSION: Find WHY the bug happens

    STEP 1: Review reproduction evidence from Phase 1
    STEP 2: Trace through the code
    STEP 3: Identify the faulty code
    STEP 4: Propose a fix
    STEP 5: Create Phase 3 task with fix details
    STEP 6: Mark done
    """
)

PHASE_3_FIX = Phase(
    id=3,
    name="fix_implementation",
    description="Implement the bug fix and verify it works",
    done_definitions=[
        "Bug fix implemented",
        "Tests added to prevent regression",
        "All tests pass",
        "Bug cannot be reproduced anymore",
        "Task marked as done"
    ],
    working_directory=".",
    additional_notes="""
    🎯 YOUR MISSION: Apply the fix and verify

    STEP 1: Implement the proposed fix
    STEP 2: Add regression test
    STEP 3: Run all tests
    STEP 4: Verify bug is fixed
    STEP 5: Mark done
    """
)

BUG_FIX_PHASES = [
    PHASE_1_REPRODUCTION,
    PHASE_2_INVESTIGATION,
    PHASE_3_FIX
]