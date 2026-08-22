"""Claw-R1's self-contained runtime for CLI-based black-box agents.

Keep this module lightweight: Gateway imports ``protocol`` before a training
worker (and therefore Hydra/verl) is necessarily installed.
"""

__all__ = [
    "BlackBoxAgentFlowBase",
    "BlackBoxCliAgentFlow",
    "E2BSandboxProvider",
    "LocalBwrapSandboxProvider",
    "create_sandbox_provider",
]


def __getattr__(name: str):
    if name in {"BlackBoxAgentFlowBase", "BlackBoxCliAgentFlow"}:
        from .flow import BlackBoxAgentFlowBase, BlackBoxCliAgentFlow
        return {"BlackBoxAgentFlowBase": BlackBoxAgentFlowBase, "BlackBoxCliAgentFlow": BlackBoxCliAgentFlow}[name]
    if name in {"E2BSandboxProvider", "LocalBwrapSandboxProvider", "create_sandbox_provider"}:
        from .sandbox import E2BSandboxProvider, LocalBwrapSandboxProvider, create_sandbox_provider
        return {"E2BSandboxProvider": E2BSandboxProvider, "LocalBwrapSandboxProvider": LocalBwrapSandboxProvider, "create_sandbox_provider": create_sandbox_provider}[name]
    raise AttributeError(name)
