"""Compare failure-aware federation against a failure-unaware baseline.

Default: ``fast=true`` on every query (graph lookup only, no ML training) so the
demo finishes quickly and isolates unaware vs aware failure handling.
"""

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import requests

PROJECT_ROOT = Path(__file__).resolve().parent
COORDINATOR_URL = "http://localhost:8000/global_retrieve"
CLIENT_2_PORT = 8002
CLIENT_2_COMMAND = [
    sys.executable,
    "client_api.py",
    "--port",
    str(CLIENT_2_PORT),
    "--file",
    "data/client_2_graph.graphml",
    "--name",
    "Client_2",
]
DRUG_IDS = [
    "CID000003488",
    "CID000000271",
    "CID000002764",
    "CID000003066",
    "CID000003168",
    "CID000051263",
    "CID000004927",
    "CID000005152",
]
TOTAL_REQUESTS_PER_MODE = 20
FAULT_RATE = 0.35
PRE_QUERY_FAILURE_SETTLE_SECONDS = 0.25
COORDINATOR_TIMEOUT_FAST_SECONDS = 45.0
COORDINATOR_TIMEOUT_FULL_SECONDS = 120.0
CLIENT_RESTART_WAIT_SECONDS = 12.0
POST_RESTART_SETTLE_SECONDS = 1.0


def find_pid_by_port(port: int) -> int | None:
    """Return the PID currently listening on a local TCP port."""
    for connection in psutil.net_connections(kind="inet"):
        if not connection.laddr or connection.status != psutil.CONN_LISTEN:
            continue

        if connection.laddr.port == port:
            return connection.pid

    return None


def wait_for_port(port: int, timeout: float = CLIENT_RESTART_WAIT_SECONDS) -> bool:
    """Wait until a process starts listening on the given local port."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if find_pid_by_port(port) is not None:
            return True
        time.sleep(0.25)

    return False


def restart_client_2() -> None:
    """Restart Client 2 if it is not already listening."""
    if find_pid_by_port(CLIENT_2_PORT) is not None:
        return

    process = subprocess.Popen(
        CLIENT_2_COMMAND,
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if wait_for_port(CLIENT_2_PORT):
        print(f"[chaos] Client 2 restarted on port {CLIENT_2_PORT}.")
    elif process.poll() is not None:
        print(f"[chaos] Client 2 restart exited with code {process.returncode}.")
    else:
        print(f"[chaos] Client 2 restart did not bind to port {CLIENT_2_PORT}.")


def kill_client_2() -> bool:
    """Kill the process listening as Client 2."""
    pid = find_pid_by_port(CLIENT_2_PORT)
    if pid is None:
        print(f"[chaos] No process found on port {CLIENT_2_PORT}; Client 2 is already down.")
        return False

    try:
        process = psutil.Process(pid)
        print(f"[chaos] Killing Client 2 on port {CLIENT_2_PORT} with PID {pid}.")
        process.kill()
        process.wait(timeout=3)
        return True
    except psutil.NoSuchProcess:
        print("[chaos] Client 2 process disappeared before it could be killed.")
        return False
    except psutil.AccessDenied:
        print(f"[chaos] Access denied while killing Client 2 PID {pid}.")
        return False
    except psutil.TimeoutExpired:
        print(f"[chaos] Client 2 PID {pid} did not exit before timeout.")
        return False


def query_coordinator(mode: str, drug_id: str, *, fast: bool) -> tuple[int, dict]:
    """Send one federated retrieval request and return HTTP status plus payload."""
    timeout = COORDINATOR_TIMEOUT_FAST_SECONDS if fast else COORDINATOR_TIMEOUT_FULL_SECONDS
    params: dict[str, str] = {"drug_id": drug_id, "mode": mode}
    if fast:
        params["fast"] = "true"

    response = requests.get(
        COORDINATOR_URL,
        params=params,
        timeout=timeout,
    )

    try:
        payload = response.json()
    except ValueError:
        payload = {"detail": response.text}

    return response.status_code, payload


def build_experiment_plan(total_requests: int) -> list[tuple[str, bool]]:
    """Create a reproducible query and failure plan shared by both baselines."""
    return [
        (random.choice(DRUG_IDS), random.random() < FAULT_RATE)
        for _ in range(total_requests)
    ]


def run_mode_experiment(
    mode: str,
    experiment_plan: list[tuple[str, bool]],
    *,
    fast: bool,
) -> dict[str, int]:
    """Run one baseline mode under controlled Client 2 failures."""
    metrics = {
        "full_success": 0,
        "degraded_success": 0,
        "failed_stalled": 0,
    }

    ml_label = "fast (no ML)" if fast else "full ML"
    print(f"\n=== Running {mode.upper()} baseline ({len(experiment_plan)} queries, {ml_label}) ===")
    if mode == "unaware":
        print(
            "[note] unaware mode returns HTTP 500 when any client is down during the query "
            "(expected on chaos queries; not 3/3)."
        )

    for query_number, (drug_id, inject_fault) in enumerate(experiment_plan, start=1):
        if inject_fault:
            # Client 2 stays down until after this query completes.
            kill_client_2()
            time.sleep(PRE_QUERY_FAILURE_SETTLE_SECONDS)
            chaos_note = " | chaos=Client_2_down"
        else:
            restart_client_2()
            if POST_RESTART_SETTLE_SECONDS > 0:
                time.sleep(POST_RESTART_SETTLE_SECONDS)
            chaos_note = ""

        status_code = 0
        payload = {}
        try:
            status_code, payload = query_coordinator(mode, drug_id, fast=fast)
        except requests.RequestException as exc:
            metrics["failed_stalled"] += 1
            print(
                f"[{mode} query {query_number:02d}] request_failed | "
                f"drug_id={drug_id}{chaos_note} | error={exc}"
            )
            if inject_fault:
                restart_client_2()
                if POST_RESTART_SETTLE_SECONDS > 0:
                    time.sleep(POST_RESTART_SETTLE_SECONDS)
            continue

        if status_code == 500:
            completeness_score = "n/a (stalled)"
        else:
            completeness_score = payload.get("completeness_score", "0/3")

        if status_code == 200:
            if completeness_score == "3/3":
                metrics["full_success"] += 1
            else:
                metrics["degraded_success"] += 1
        elif status_code == 500:
            metrics["failed_stalled"] += 1
        else:
            metrics["failed_stalled"] += 1

        print(
            f"[{mode} query {query_number:02d}] HTTP {status_code} | "
            f"drug_id={drug_id} | completeness={completeness_score}{chaos_note}"
        )

        if inject_fault:
            restart_client_2()
            if POST_RESTART_SETTLE_SECONDS > 0:
                time.sleep(POST_RESTART_SETTLE_SECONDS)

    restart_client_2()
    return metrics


def create_comparison_chart(unaware_metrics: dict[str, int], aware_metrics: dict[str, int]) -> None:
    """Create the side-by-side reliability chart."""
    plt.style.use("ggplot")

    categories = ["Successful Queries", "Failed/Stalled Queries"]
    x_positions = np.arange(len(categories))
    bar_width = 0.35

    unaware_success = unaware_metrics["full_success"] + unaware_metrics["degraded_success"]
    unaware_failed = unaware_metrics["failed_stalled"]
    aware_full = aware_metrics["full_success"]
    aware_degraded = aware_metrics["degraded_success"]
    aware_failed = aware_metrics["failed_stalled"]

    fig, ax = plt.subplots(figsize=(10, 6))
    unaware_bars = ax.bar(
        x_positions - bar_width / 2,
        [unaware_success, unaware_failed],
        bar_width,
        label="Failure-Unaware",
        color="#C62828",
    )
    aware_full_bars = ax.bar(
        x_positions[0] + bar_width / 2,
        aware_full,
        bar_width,
        label="Failure-Aware: Full (3/3)",
        color="#2E7D32",
    )
    aware_degraded_bars = ax.bar(
        x_positions[0] + bar_width / 2,
        aware_degraded,
        bar_width,
        bottom=aware_full,
        label="Failure-Aware: Degraded (2/3)",
        color="#F9A825",
    )
    aware_failed_bars = ax.bar(
        x_positions[1] + bar_width / 2,
        aware_failed,
        bar_width,
        label="Failure-Aware: Failed",
        color="#6A1B9A",
    )

    ax.set_title("System Reliability: Failure-Aware vs. Failure-Unaware Baseline")
    ax.set_ylabel("Number of Queries")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, TOTAL_REQUESTS_PER_MODE)
    ax.legend()

    for bars in (unaware_bars, aware_full_bars, aware_degraded_bars, aware_failed_bars):
        ax.bar_label(bars, padding=3)

    fig.tight_layout()
    output_path = PROJECT_ROOT / "baseline_comparison.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"\nSaved baseline comparison chart to {output_path}")


def print_summary(unaware_metrics: dict[str, int], aware_metrics: dict[str, int]) -> None:
    """Print a concise terminal summary for the A/B experiment."""
    unaware_success = unaware_metrics["full_success"] + unaware_metrics["degraded_success"]
    aware_success = aware_metrics["full_success"] + aware_metrics["degraded_success"]

    print("\n=== Baseline Comparison Summary ===")
    print(
        "Failure-Unaware: "
        f"{unaware_success} successful, {unaware_metrics['failed_stalled']} failed/stalled"
    )
    print(
        "Failure-Aware: "
        f"{aware_success} successful "
        f"({aware_metrics['full_success']} full, {aware_metrics['degraded_success']} degraded), "
        f"{aware_metrics['failed_stalled']} failed"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare failure-unaware vs failure-aware federation under Client 2 chaos."
    )
    parser.add_argument(
        "--full-ml",
        action="store_true",
        help="Run full ML training per query (slow). Default is fast graph-only mode.",
    )
    args = parser.parse_args()
    use_fast = not args.full_ml

    if use_fast:
        print("[load-test] Using fast=true — no ML training; graph neighbor lookup only.")

    random.seed(42)
    shared_experiment_plan = build_experiment_plan(TOTAL_REQUESTS_PER_MODE)
    unaware_results = run_mode_experiment("unaware", shared_experiment_plan, fast=use_fast)
    aware_results = run_mode_experiment("aware", shared_experiment_plan, fast=use_fast)
    print_summary(unaware_results, aware_results)
    create_comparison_chart(unaware_results, aware_results)
