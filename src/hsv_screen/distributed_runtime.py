from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from .config import (
    REQUIRED_DISTRIBUTED_NODES,
    REQUIRED_WORKERS_PER_NODE,
    load_config,
    resolve_path,
)
from .io_utils import atomic_json


RANK_KEYS = ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "SLURM_PROCID")
SIZE_KEYS = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "SLURM_NTASKS")
LOCAL_RANK_KEYS = (
    "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "PMI_LOCAL_RANK",
    "SLURM_LOCALID",
)
RUN_ID_ENV = "HSV_SCREEN_DISTRIBUTED_RUN_ID"
CHILD_ENV = "HSV_SCREEN_DISTRIBUTED_CHILD"


def _first_int(environment: Mapping[str, str], keys: tuple[str, ...], default: int) -> int:
    for key in keys:
        value = environment.get(key)
        if value is not None:
            try:
                return int(value)
            except ValueError as exc:
                raise ValueError(f"Invalid integer in {key}={value!r}") from exc
    return default


def distributed_identity(environment: Mapping[str, str] | None = None) -> tuple[int, int, int]:
    env = os.environ if environment is None else environment
    # Scheduler allocation variables alone do not mean that this Python process
    # is an MPI rank.  Only trust a world size after a rank variable exists.
    if not any(key in env for key in RANK_KEYS):
        return 0, 1, 0
    return (
        _first_int(env, RANK_KEYS, 0),
        _first_int(env, SIZE_KEYS, 1),
        _first_int(env, LOCAL_RANK_KEYS, 0),
    )


def validate_layout(config: dict, world_size: int | None = None) -> dict:
    settings = config.get("distributed", {})
    nodes = int(settings.get("nodes", 0))
    workers = int(settings.get("workers_per_node", 0))
    local_workers = int(config.get("workers", 0))
    docking_workers = int(config.get("docking", {}).get("parallel_jobs_per_node", 0))
    if nodes != REQUIRED_DISTRIBUTED_NODES:
        raise ValueError(
            "This workflow requires distributed.nodes="
            f"{REQUIRED_DISTRIBUTED_NODES}, found {nodes}")
    if workers != REQUIRED_WORKERS_PER_NODE or local_workers != REQUIRED_WORKERS_PER_NODE:
        raise ValueError(
            f"This workflow requires {REQUIRED_WORKERS_PER_NODE} local workers per node; found "
            f"distributed.workers_per_node={workers}, workers={local_workers}")
    if docking_workers != REQUIRED_WORKERS_PER_NODE:
        raise ValueError(
            "This workflow requires docking.parallel_jobs_per_node="
            f"{REQUIRED_WORKERS_PER_NODE}, found "
            f"{docking_workers}")
    if int(config["docking"].get("cpu_per_job", 0)) != 1:
        raise ValueError("Distributed Smina requires docking.cpu_per_job=1")
    if world_size is not None and world_size != nodes:
        raise ValueError(f"Expected exactly {nodes} MPI ranks, found {world_size}")
    return {
        "nodes": nodes,
        "workers_per_node": workers,
        "total_workers": nodes * workers,
    }


def build_mpi_command(config: dict, config_path: Path, mode: str) -> list[str]:
    layout = validate_layout(config)
    settings = config["distributed"]
    launcher = shutil.which(str(settings.get("launcher", "mpirun")))
    if not launcher:
        raise FileNotFoundError(
            "mpirun is unavailable. Load the platform MPI module inside the allocated "
            "job (for example `module load mpi`) before running the Python entry point.")
    arguments = [str(value) for value in settings.get(
        "launcher_args", [
            "--map-by", f"ppr:1:node:PE={REQUIRED_WORKERS_PER_NODE}",
            "--bind-to", "core",
        ])]
    return [
        launcher, "-np", str(layout["nodes"]), *arguments,
        sys.executable, "-m", "src.hsv_screen.cli", mode,
        "--config", str(config_path.resolve()),
    ]


@dataclass
class RankContext:
    rank: int
    size: int
    local_rank: int
    hostname: str
    run_id: str
    sync_dir: Path
    timeout_seconds: float
    poll_seconds: float

    @classmethod
    def from_config(
        cls, config: dict, environment: Mapping[str, str] | None = None,
        hostname: str | None = None,
    ) -> "RankContext":
        env = os.environ if environment is None else environment
        rank, size, local_rank = distributed_identity(env)
        validate_layout(config, size)
        run_id = env.get(RUN_ID_ENV, "")
        if not run_id:
            raise RuntimeError(f"Missing {RUN_ID_ENV}; distributed workers must be launcher-managed")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
            raise ValueError(f"Unsafe distributed run ID: {run_id!r}")
        root = resolve_path(config, config["output_dir"])
        sync_dir = root / ".distributed" / run_id
        sync_dir.mkdir(parents=True, exist_ok=True)
        settings = config["distributed"]
        return cls(
            rank=rank, size=size, local_rank=local_rank,
            hostname=hostname or socket.gethostname(), run_id=run_id,
            sync_dir=sync_dir,
            timeout_seconds=float(settings.get("barrier_timeout_seconds", 604800)),
            poll_seconds=float(settings.get("barrier_poll_seconds", 2.0)),
        )

    @property
    def is_root(self) -> bool:
        return self.rank == 0

    def log(self, message: str) -> None:
        print(f"[node-rank {self.rank}/{self.size} {self.hostname}] {message}", flush=True)

    def _safe_name(self, name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

    def fail(self, exc: BaseException) -> None:
        atomic_json(self.sync_dir / f"error_rank_{self.rank:02d}.json", {
            "rank": self.rank, "hostname": self.hostname,
            "error_type": type(exc).__name__, "message": str(exc),
        })

    def _raise_peer_error(self) -> None:
        errors = sorted(self.sync_dir.glob("error_rank_*.json"))
        if not errors:
            return
        detail = json.loads(errors[0].read_text(encoding="utf-8"))
        raise RuntimeError(
            f"Distributed rank {detail.get('rank')} on {detail.get('hostname')} failed: "
            f"{detail.get('error_type')}: {detail.get('message')}")

    def barrier(self, name: str) -> None:
        barrier_dir = self.sync_dir / "barriers" / self._safe_name(name)
        barrier_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(barrier_dir / f"rank_{self.rank:02d}.json", {
            "rank": self.rank, "hostname": self.hostname, "time": time.time(),
        })
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            self._raise_peer_error()
            if len(list(barrier_dir.glob("rank_*.json"))) == self.size:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting at distributed barrier {name!r}; "
                    f"found {len(list(barrier_dir.glob('rank_*.json')))}/{self.size} ranks")
            time.sleep(self.poll_seconds)

    def root_value(self, name: str, producer: Callable[[], object]) -> object:
        path = self.sync_dir / "values" / f"{self._safe_name(name)}.json"
        if self.is_root:
            atomic_json(path, producer())
        self.barrier(f"value_{name}")
        return json.loads(path.read_text(encoding="utf-8"))

    def validate_unique_hosts(self, config: dict) -> list[str]:
        required_cpus = int(config["distributed"]["workers_per_node"])
        mapping = f"ppr:1:node:PE={required_cpus}"
        host_dir = self.sync_dir / "hosts"
        affinity_count = (
            len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
            else int(os.cpu_count() or 0))
        atomic_json(host_dir / f"rank_{self.rank:02d}.json", {
            "rank": self.rank, "hostname": self.hostname,
            "local_rank": self.local_rank, "available_cpus": affinity_count,
        })
        self.barrier("hosts_written")
        result_path = self.sync_dir / "host_layout.json"
        if self.is_root:
            records = [json.loads(path.read_text(encoding="utf-8"))
                       for path in sorted(host_dir.glob("rank_*.json"))]
            hosts = [str(record["hostname"]) for record in records]
            if len(set(hosts)) != self.size:
                raise RuntimeError(
                    "MPI ranks are not mapped one-per-node: " + ", ".join(hosts) +
                    f". Use the configured `--map-by {mapping}` launcher mapping.")
            insufficient = [record for record in records
                            if int(record.get("available_cpus", 0)) < required_cpus]
            if insufficient:
                raise RuntimeError(
                    f"At least one MPI rank can access fewer than {required_cpus} CPUs: " +
                    ", ".join(
                        f"{record['hostname']}={record.get('available_cpus')}"
                        for record in insufficient) +
                    f". Use `{mapping}`, bind to cores, and request ptile={required_cpus}.")
            atomic_json(result_path, {"hosts": hosts, "records": records})
        self.barrier("hosts_validated")
        return list(json.loads(result_path.read_text(encoding="utf-8"))["hosts"])


def launch(
    config_path: str | Path, mode: str,
    worker: Callable[[dict, RankContext, str], dict | None],
) -> dict | None:
    config_path = Path(config_path).resolve()
    config, _ = load_config(config_path)
    rank, size, _ = distributed_identity()
    if size == 1 and os.environ.get(CHILD_ENV) != "1":
        command = build_mpi_command(config, config_path, mode)
        run_id = f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        environment = os.environ.copy()
        environment.update({
            RUN_ID_ENV: run_id, CHILD_ENV: "1", "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        })
        process = subprocess.run(command, env=environment)
        if process.returncode != 0:
            raise RuntimeError(f"Distributed MPI run failed with exit code {process.returncode}")
        root = resolve_path(config, config["output_dir"])
        result_path = root / ".distributed" / run_id / "result.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Distributed result marker is missing: {result_path}")
        return json.loads(result_path.read_text(encoding="utf-8"))

    context = RankContext.from_config(config)
    try:
        result = worker(config, context, mode)
        if context.is_root:
            atomic_json(context.sync_dir / "result.json", result or {})
        context.barrier("distributed_result_written")
        return result if context.is_root else None
    except BaseException as exc:
        context.fail(exc)
        raise
