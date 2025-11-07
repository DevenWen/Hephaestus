#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Import phases
from my_workflow.phases import BUG_FIX_PHASES
from my_workflow.config import BUG_FIX_CONFIG

from src.sdk import HephaestusSDK

# Load environment
load_dotenv()

def main():
    # Initialize SDK
    sdk = HephaestusSDK(
        phases=BUG_FIX_PHASES,
        workflow_config=BUG_FIX_CONFIG,
        database_path="./hephaestus.db",
        qdrant_url="http://localhost:6333",
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        working_directory=".",
        mcp_port=8000,
        monitoring_interval=60
    )

    # Start services
    print("[Hephaestus] Starting services...")
    sdk.start()

    print("[Hephaestus] Loaded phases:")
    for phase_id, phase in sorted(sdk.phases_map.items()):
        print(f"  - Phase {phase_id}: {phase.name}")

    # Create initial task
    print("\n[Task] Creating Phase 1 bug reproduction task...")
    task_id = sdk.create_task(
        description="""
        Phase 1: Reproduce Bug - "Login fails with special characters"

        Bug Report:
        - User enters password with @ symbol
        - Login button becomes unresponsive
        - Error in console: "Invalid character in auth string"

        Reproduce this bug and capture evidence.
        """,
        phase_id=1,
        priority="high",
        agent_id="main-session-agent"
    )
    print(f"[Task] Created task: {task_id}")

    # Keep running
    print("\n[Hephaestus] Workflow running. Press Ctrl+C to stop.\n")
    try:
        while True:
            import time
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n[Hephaestus] Shutting down...")
        sdk.shutdown(graceful=True, timeout=10)
        print("[Hephaestus] Shutdown complete")

if __name__ == "__main__":
    main()