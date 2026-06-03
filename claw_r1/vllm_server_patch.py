"""Runtime patches for verl's async vLLM HTTP server integration."""

from __future__ import annotations

import asyncio
import contextlib
import logging

import ray

logger = logging.getLogger(__name__)


def _get_config_value(config, key: str):
    if hasattr(config, "get"):
        return config.get(key, None)
    return getattr(config, key, None)


def _install_uvicorn_patch(vllm_server, rollout_utils, uvicorn) -> None:
    if getattr(vllm_server, "_claw_r1_uvicorn_patched", False):
        return

    async def run_uvicorn_until_started(app, server_args, server_address, max_retries=5):
        server_port, server_task = None, None
        last_error = None

        for i in range(max_retries):
            sock = None
            try:
                server_port, sock = rollout_utils.get_free_port(server_address)
                app.server_args = server_args
                config = uvicorn.Config(app, host=server_address, port=server_port, log_level="warning")
                server = uvicorn.Server(config)
                server_task = asyncio.create_task(server.serve(sockets=[sock]))

                for _ in range(200):
                    if server_task.done():
                        exc = server_task.exception()
                        raise RuntimeError(f"uvicorn server exited during startup: {exc}")
                    if server.started:
                        logger.info("HTTP server started on %s:%s", server_address, server_port)
                        return server_port, server_task
                    await asyncio.sleep(0.05)

                server.should_exit = True
                await server_task
                raise TimeoutError(f"uvicorn server did not start on {server_address}:{server_port}")
            except (OSError, RuntimeError, TimeoutError) as exc:
                last_error = exc
                logger.error("Failed to start HTTP server on port %s at try %s: %s", server_port, i, exc)
                if server_task is not None and not server_task.done():
                    server_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await server_task
                if sock is not None:
                    with contextlib.suppress(OSError):
                        sock.close()

        raise RuntimeError(f"Failed to start HTTP server after {max_retries} retries: {last_error}")

    rollout_utils.run_unvicorn = run_uvicorn_until_started
    vllm_server.run_unvicorn = run_uvicorn_until_started
    vllm_server._claw_r1_uvicorn_patched = True


def apply_verl_vllm_server_patches() -> None:
    """Patch verl vLLM server startup issues in the current process.

    The vendored verl implementation returns a port after briefly entering
    uvicorn with ``should_exit=True``. In practice this can hand back an
    address with no listener, so the Gateway cannot connect.
    """
    try:
        import uvicorn

        import verl.workers.rollout.utils as rollout_utils
        import verl.workers.rollout.vllm_rollout.vllm_async_server as vllm_server
    except Exception as exc:
        logger.warning("Unable to patch verl vLLM server integration: %s", exc)
        return

    if getattr(vllm_server, "_claw_r1_patched", False):
        return

    _install_uvicorn_patch(vllm_server, rollout_utils, uvicorn)

    original_init = vllm_server.vLLMHttpServerBase.__init__

    def patched_init(self, config, model_config, *args, **kwargs):
        requested_max_model_len = _get_config_value(config, "max_model_len")
        original_init(self, config, model_config, *args, **kwargs)
        if requested_max_model_len is not None:
            self.config.max_model_len = int(requested_max_model_len)

    vllm_server.vLLMHttpServerBase.__init__ = patched_init
    vllm_server._claw_r1_patched = True


def _prepare_server_process_patch():
    import uvicorn

    import verl.workers.rollout.utils as rollout_utils
    import verl.workers.rollout.vllm_rollout.vllm_async_server as vllm_server

    _install_uvicorn_patch(vllm_server, rollout_utils, uvicorn)
    return vllm_server


def get_patched_vllm_http_server_class():
    """Return a Ray actor class that patches verl inside the server process."""
    import verl.workers.rollout.vllm_rollout.vllm_async_server as vllm_server

    @ray.remote(num_cpus=1)
    class PatchedVLLMHttpServer(vllm_server.vLLMHttpServerBase):
        def __init__(self, config, model_config, *args, **kwargs):
            requested_max_model_len = _get_config_value(config, "max_model_len")
            vllm_server_module = _prepare_server_process_patch()
            super().__init__(config, model_config, *args, **kwargs)
            if requested_max_model_len is not None:
                self.config.max_model_len = int(requested_max_model_len)
            print(
                "ClawR1 PatchedVLLMHttpServer: "
                f"patched={getattr(vllm_server_module, '_claw_r1_uvicorn_patched', False)} "
                f"max_model_len={self.config.max_model_len}"
            )

    return PatchedVLLMHttpServer


def patch_rollout_replicas(replicas) -> None:
    """Install the patched server actor class onto vLLM rollout replicas."""
    server_class = get_patched_vllm_http_server_class()
    for replica in replicas:
        if replica.__class__.__name__ == "vLLMReplica":
            replica.server_class = server_class
