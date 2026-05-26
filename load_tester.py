"""Fault-injection load tester for the federated graph retrieval coordinator."""

import argparse
import asyncio
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
COORDINATOR_URL = "http://localhost:8000/global_retrieve"
CLIENT_DOWNTIME_SECONDS = 2.5
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


async def wait_for_port(port: int, timeout: float = 8.0) -> bool:
    """Wait until a restarted client is listening on the expected port."""
    deadline = time.perf_counter() + timeout

    while time.perf_counter() < deadline:
        if find_pid_by_port(port) is not None:
            return True
        await asyncio.sleep(0.25)

    return False


async def inject_fault(port: int = 8002, downtime_seconds: float = CLIENT_DOWNTIME_SECONDS) -> None:
    """Kill and restart the client listening on the given port."""
    pid = find_pid_by_port(port)

    if pid is None:
        print(f"[chaos] No process found on port {port}; treating client as already down.")
    else:
        try:
            process = psutil.Process(pid)
            print(f"[chaos] Killing Client 2 on port {port} with PID {pid}.")
            try:
                os.kill(pid, signal.SIGKILL)
            except AttributeError:
                process.kill()
            except OSError:
                process.kill()
        except psutil.NoSuchProcess:
            print(f"[chaos] Process on port {port} disappeared before it could be killed.")
        except psutil.AccessDenied:
            print(f"[chaos] Access denied while killing PID {pid}; skipping fault injection.")
            return

    await asyncio.sleep(downtime_seconds)

    print("[chaos] Restarting Client 2.")
    restart_process = subprocess.Popen(
        CLIENT_2_COMMAND,
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    restarted = await wait_for_port(port)
    if restarted:
        print(f"[chaos] Client 2 restarted on port {port}.")
    elif restart_process.poll() is not None:
        print(
            f"[chaos] Client 2 restart exited early with code "
            f"{restart_process.returncode}."
        )
    else:
        print(f"[chaos] Client 2 restart did not bind to port {port} in time.")


async def fetch_coordinator(session: aiohttp.ClientSession, drug_id: str) -> tuple[int, dict]:
    """Send one retrieval query to the coordinator."""
    async with session.get(COORDINATOR_URL, params={"drug_id": drug_id}, timeout=10) as response:
        payload = await response.json()
        return response.status, payload


async def run_load_test(
    total_queries: int = 100,
    fault_rate: float = 0.30,
    downtime_seconds: float = CLIENT_DOWNTIME_SECONDS,
) -> None:
    """Run 100 coordinator queries while randomly killing and restarting Client 2."""
    full_answers = 0
    degraded_answers = 0
    failed_queries = 0
    start_time = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        for query_number in range(1, total_queries + 1):
            drug_id = random.choice(DRUG_IDS)
            should_inject_fault = random.random() < fault_rate
            fault_task = None

            if should_inject_fault:
                fault_task = asyncio.create_task(
                    inject_fault(8002, downtime_seconds=downtime_seconds)
                )
                await asyncio.sleep(0.2)

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
                print(
                    f"[query {query_number:03d}] request_failed | "
                    f"drug_id={drug_id} | error={exc}"
                )
            except asyncio.CancelledError:
                failed_queries += 1
                degraded_answers += 1
                print(
                    f"[query {query_number:03d}] request_cancelled | "
                    f"drug_id={drug_id}"
                )
                if fault_task is not None:
                    await fault_task
                continue

            if fault_task is not None:
                await fault_task

    elapsed_seconds = time.perf_counter() - start_time

    print("\n=== Load Test Summary ===")
    print(f"Total Queries: {total_queries}")
    print(f"Full Answers (3/3): {full_answers}")
    print(f"Degraded Answers (<3/3): {degraded_answers}")
    print(f"Failed Coordinator Requests: {failed_queries}")
    print(f"Elapsed Time: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run fault-injection load tests against the coordinator."
    )
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--fault-rate", type=float, default=0.30)
    parser.add_argument("--downtime", type=float, default=CLIENT_DOWNTIME_SECONDS)
    args = parser.parse_args()

    asyncio.run(
        run_load_test(
            total_queries=args.queries,
            fault_rate=args.fault_rate,
            downtime_seconds=args.downtime,
        )
    )
