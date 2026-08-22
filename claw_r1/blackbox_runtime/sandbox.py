"""Sandbox leases for one black-box agent session per rollout."""

from __future__ import annotations

import abc
import asyncio
import os
import shutil
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SandboxLease:
    endpoint: str
    workspace: str
    process: asyncio.subprocess.Process | None = None
    provider: str = "local_bwrap"


class SandboxProvider(abc.ABC):
    @abc.abstractmethod
    async def acquire(self, *, trajectory_uid: str, metadata: dict[str, Any]) -> SandboxLease: ...

    @abc.abstractmethod
    async def release(self, lease: SandboxLease) -> None: ...


class LocalBwrapSandboxProvider(SandboxProvider):
    """Launch an isolated black-box runtime per lease.

    Bubblewrap is used when installed; tests and developer machines may set
    ``CLAW_R1_BWRAP=none`` to use the same lifecycle without namespace setup.
    """
    def __init__(self) -> None:
        self.bwrap = os.environ.get("CLAW_R1_BWRAP", "bwrap")
        self.host = os.environ.get("CLAW_R1_BLACKBOX_HOST", "127.0.0.1")

    async def acquire(self, *, trajectory_uid: str, metadata: dict[str, Any]) -> SandboxLease:
        workspace = tempfile.mkdtemp(prefix=f"claw-r1-{trajectory_uid[:8]}-")
        port = _free_port()
        cmd = ["python", "-m", "claw_r1.blackbox_runtime.server", "--port", str(port), "--workspace", workspace]
        if self.bwrap not in {"", "none"}:
            binary = shutil.which(self.bwrap)
            if binary is None:
                raise RuntimeError("Bubblewrap is required; install bwrap or set CLAW_R1_BWRAP=none for development")
            # The service needs Python and the current checkout visible, while
            # task writes remain confined to its dedicated workspace.
            cmd = [binary, "--die-with-parent", "--bind", "/", "/", "--bind", workspace, workspace, "--chdir", workspace, *cmd]
        process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        lease = SandboxLease(endpoint=f"http://{self.host}:{port}", workspace=workspace, process=process)
        await _wait_health(lease.endpoint)
        return lease

    async def release(self, lease: SandboxLease) -> None:
        if lease.process and lease.process.returncode is None:
            lease.process.terminate()
            try:
                await asyncio.wait_for(lease.process.wait(), timeout=10)
            except asyncio.TimeoutError:
                lease.process.kill()
                await lease.process.wait()
        shutil.rmtree(lease.workspace, ignore_errors=True)


class E2BSandboxProvider(SandboxProvider):
    """E2B lease adapter; importing E2B stays optional until it is selected."""
    async def acquire(self, *, trajectory_uid: str, metadata: dict[str, Any]) -> SandboxLease:
        endpoint = metadata.get("blackbox_endpoint")
        if not endpoint:
            raise RuntimeError("E2B provider requires metadata.blackbox_endpoint from the provisioned E2B template")
        return SandboxLease(endpoint=str(endpoint), workspace=str(metadata.get("workspace", "/workspace")), provider="e2b")

    async def release(self, lease: SandboxLease) -> None:
        del lease


def create_sandbox_provider(name: str | None = None) -> SandboxProvider:
    name = (name or os.environ.get("CLAW_R1_SANDBOX_PROVIDER", "local_bwrap")).lower()
    if name == "local_bwrap": return LocalBwrapSandboxProvider()
    if name == "e2b": return E2BSandboxProvider()
    raise ValueError(f"Unsupported sandbox provider: {name}")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _wait_health(endpoint: str) -> None:
    import httpx
    for _ in range(50):
        try:
            if (await httpx.AsyncClient(timeout=0.2).get(f"{endpoint}/health")).is_success: return
        except httpx.HTTPError: pass
        await asyncio.sleep(0.1)
    raise RuntimeError(f"black-box runtime did not become healthy: {endpoint}")
