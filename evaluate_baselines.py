"""Compare failure-aware federation against a failure-unaware baseline."""

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


def find_pid_by_port(port: int) -> int | None:
    """Return the PID currently listening on a local TCP port."""
    for connection in psutil.net_connections(kind="inet"):
        if not connection.laddr or connection.status != psutil.CONN_LISTEN:
            continue

        if connection.laddr.port == port:
            return connection.pid

    return None


def wait_for_port(port: int, timeout: float = 8.0) -> bool:
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


def query_coordinator(mode: str, drug_id: str) -> tuple[int, dict]:
    """Send one federated retrieval request and return HTTP status plus payload."""
    response = requests.get(
        COORDINATOR_URL,
        params={"drug_id": drug_id, "mode": mode},
        timeout=12,
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


def run_mode_experiment(mode: str, experiment_plan: list[tuple[str, bool]]) -> dict[str, int]:
    """Run one baseline mode under controlled Client 2 failures."""
    metrics = {
        "full_success": 0,
        "degraded_success": 0,
        "failed_stalled": 0,
    }

    print(f"\n=== Running {mode.upper()} baseline ({len(experiment_plan)} queries) ===")
    for query_number, (drug_id, inject_fault) in enumerate(experiment_plan, start=1):
        if inject_fault:
            # Keep Client 2 down during the request so the query observes the failure.
            kill_client_2()
            time.sleep(PRE_QUERY_FAILURE_SETTLE_SECONDS)
        else:
            restart_client_2()

        status_code = 0
        payload = {}
        try:
            status_code, payload = query_coordinator(mode, drug_id)
        except requests.RequestException as exc:
            metrics["failed_stalled"] += 1
            print(
                f"[{mode} query {query_number:02d}] request_failed | "
                f"drug_id={drug_id} | error={exc}"
            )
            continue
        finally:
            if inject_fault:
                restart_client_2()

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
            f"drug_id={drug_id} | completeness={completeness_score}"
        )

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
    random.seed(42)
    shared_experiment_plan = build_experiment_plan(TOTAL_REQUESTS_PER_MODE)
    unaware_results = run_mode_experiment("unaware", shared_experiment_plan)
    aware_results = run_mode_experiment("aware", shared_experiment_plan)
    print_summary(unaware_results, aware_results)
    create_comparison_chart(unaware_results, aware_results)
