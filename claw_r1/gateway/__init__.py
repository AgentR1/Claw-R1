"""Gateway Server — the HTTP bridge between Agent Side and Training Side.

The Gateway is a standalone FastAPI process that proxies LLM generation,
reward computation, and trajectory submission between agents and the
Training Side (DataPool, Rollout Engine, Reward Model).
"""
