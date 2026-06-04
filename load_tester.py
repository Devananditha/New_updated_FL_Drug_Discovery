"""Fault-injection load tester for the federated graph retrieval coordinator."""

import argparse
import asyncio
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent
METRICS_DIR = PROJECT_ROOT / "metrics"
LOAD_TEST_METRICS_PATH = METRICS_DIR / "load_test_metrics.json"
COORDINATOR_URL = "http://localhost:8000/global_retrieve"
COORDINATOR_TIMEOUT_SECONDS = 120.0
CLIENT_DOWNTIME_SECONDS = 2.0
CLIENT_RESTART_WAIT_SECONDS = 10.0
POST_RESTART_SETTLE_SECONDS = 1.0
INTER_QUERY_PAUSE_SECONDS = 0.5
DEFAULT_QUERIES = 10
# Per-query probability of killing Client 2 before the request (random each run).
DEFAULT_FAULT_RATE = 0.4
CLIENT_2_COMMAND = [
    sys.executable,
    "client_api.py",
    "--port",
    "8002",
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


def find_pid_by_port(port: int) -> int | None:
    """Return the PID of the process listening on a local TCP port."""
    for connection in psutil.net_connections(kind="inet"):
        if not connection.laddr or connection.status != psutil.CONN_LISTEN:
            continue

        if connection.laddr.port == port:
            return connection.pid

    return None


async def wait_for_port(port: int, timeout: float = CLIENT_RESTART_WAIT_SECONDS) -> bool:
    """Wait until a restarted client is listening on the expected port."""
    deadline = time.perf_counter() + timeout

    while time.perf_counter() < deadline:
        if find_pid_by_port(port) is not None:
            return True
        await asyncio.sleep(0.25)

    return False


def kill_client_on_port(port: int = 8002) -> None:
    """Stop the client process bound to the given port."""
    pid = find_pid_by_port(port)

    if pid is None:
        print(f"[chaos] No process found on port {port}; client already down.")
        return

    try:
        print(f"[chaos] Killing Client 2 on port {port} with PID {pid}.")
        try:
            os.kill(pid, signal.SIGKILL)
        except AttributeError:
            psutil.Process(pid).kill()
        except OSError:
            psutil.Process(pid).kill()
    except psutil.NoSuchProcess:
        print(f"[chaos] Process on port {port} disappeared before it could be killed.")
    except psutil.AccessDenied:
        print(f"[chaos] Access denied while killing PID {pid}; skipping kill.")


async def restart_client_on_port(
    port: int = 8002,
    restart_wait_seconds: float = CLIENT_RESTART_WAIT_SECONDS,
) -> bool:
    """Start Client 2 and wait until the HTTP port is accepting connections."""
    print("[chaos] Restarting Client 2.")
    restart_process = subprocess.Popen(
        CLIENT_2_COMMAND,
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    restarted = await wait_for_port(port, timeout=restart_wait_seconds)
    if restarted:
        print(f"[chaos] Client 2 restarted on port {port}.")
        return True

    if restart_process.poll() is not None:
        print(
            f"[chaos] Client 2 restart exited early with code "
            f"{restart_process.returncode}."
        )
    else:
        print(f"[chaos] Client 2 restart did not bind to port {port} in time.")

    return False


async def fetch_coordinator(
    session: aiohttp.ClientSession,
    drug_id: str,
) -> tuple[int, dict]:
    """Send one failure-aware federated retrieval query (full ML on each client)."""
    timeout = aiohttp.ClientTimeout(total=COORDINATOR_TIMEOUT_SECONDS)
    params = {"drug_id": drug_id, "mode": "aware"}

    async with session.get(COORDINATOR_URL, params=params, timeout=timeout) as response:
        payload = await response.json()
        return response.status, payload


async def run_load_test(
    total_queries: int = DEFAULT_QUERIES,
    fault_rate: float = DEFAULT_FAULT_RATE,
    downtime_seconds: float = CLIENT_DOWNTIME_SECONDS,
    restart_wait_seconds: float = CLIENT_RESTART_WAIT_SECONDS,
    post_restart_settle_seconds: float = POST_RESTART_SETTLE_SECONDS,
    inter_query_pause_seconds: float = INTER_QUERY_PAUSE_SECONDS,
) -> dict[str, float | int]:
    """Run coordinator queries with sequential Client 2 kill/restart (demo-friendly)."""
    full_answers = 0
    degraded_answers = 0
    failed_queries = 0
    start_time = time.perf_counter()

    print(
        f"\n[load-test] {total_queries} queries | full ML per client | "
        f"fault_rate={fault_rate:.0%} (random per query) | "
        f"coordinator_timeout={COORDINATOR_TIMEOUT_SECONDS}s | downtime={downtime_seconds}s | "
        f"pause={inter_query_pause_seconds}s\n"
    )

    async with aiohttp.ClientSession() as session:
        for query_number in range(1, total_queries + 1):
            drug_id = random.choice(DRUG_IDS)
            inject_fault_this_query = random.random() < fault_rate

            if inject_fault_this_query:
                kill_client_on_port(8002)
                print(f"[chaos] Keeping Client 2 down for {downtime_seconds}s before query.")
                await asyncio.sleep(downtime_seconds)

            try:
                status_code, payload = await fetch_coordinator(session, drug_id)
                completeness_score = payload.get("completeness_score", "0/3")

                if completeness_score == "3/3":
                    full_answers += 1
                else:
                    degraded_answers += 1

                print(
                    f"[query {query_number:03d}] "
                    f"HTTP {status_code} | drug_id={drug_id} | "
                    f"completeness={completeness_score}"
                )
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                failed_queries += 1
                degraded_answers += 1
                error_label = repr(exc) if str(exc) else type(exc).__name__
                print(
                    f"[query {query_number:03d}] request_failed | "
                    f"drug_id={drug_id} | error={error_label}"
                )

            if inject_fault_this_query:
                restarted = await restart_client_on_port(
                    8002, restart_wait_seconds=restart_wait_seconds
                )
                if restarted and post_restart_settle_seconds > 0:
                    await asyncio.sleep(post_restart_settle_seconds)
            elif inter_query_pause_seconds > 0:
                await asyncio.sleep(inter_query_pause_seconds)

    elapsed_seconds = time.perf_counter() - start_time

    print("\n=== Load Test Summary ===")
    print(f"Total Queries: {total_queries}")
    print(f"Full Answers (3/3): {full_answers}")
    print(f"Degraded Answers (<3/3): {degraded_answers}")
    print(f"Failed Coordinator Requests: {failed_queries}")
    print(f"Elapsed Time: {elapsed_seconds:.2f} seconds")

    metrics = {
        "total_queries": total_queries,
        "full_answers": full_answers,
        "degraded_answers": degraded_answers,
        "failed_coordinator_requests": failed_queries,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "fault_rate": fault_rate,
        "downtime_seconds": downtime_seconds,
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    LOAD_TEST_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved load-test metrics to {LOAD_TEST_METRICS_PATH}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run fault-injection load tests against the coordinator (full ML training)."
    )
    parser.add_argument("--queries", type=int, default=DEFAULT_QUERIES)
    parser.add_argument("--fault-rate", type=float, default=DEFAULT_FAULT_RATE)
    parser.add_argument("--downtime", type=float, default=CLIENT_DOWNTIME_SECONDS)
    parser.add_argument("--restart-wait", type=float, default=CLIENT_RESTART_WAIT_SECONDS)
    parser.add_argument("--settle", type=float, default=POST_RESTART_SETTLE_SECONDS)
    parser.add_argument("--pause", type=float, default=INTER_QUERY_PAUSE_SECONDS)
    args = parser.parse_args()

    asyncio.run(
        run_load_test(
            total_queries=args.queries,
            fault_rate=args.fault_rate,
            downtime_seconds=args.downtime,
            restart_wait_seconds=args.restart_wait,
            post_restart_settle_seconds=args.settle,
            inter_query_pause_seconds=args.pause,
        )
    )
