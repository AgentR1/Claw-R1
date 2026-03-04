# Core Concepts

Claw-R1's design is built around three tightly integrated ideas. Understanding them together is the key to understanding why the framework works the way it does.

<div class="grid cards" markdown>

-   **Base URL Integration**

    ---

    Any agent that speaks HTTP can join the training loop with a single configuration change. No SDK patches, no source-code modifications — the Gateway acts as a transparent network-layer proxy.

    [:octicons-arrow-right-24: Read more](base-url-integration.md)

-   **Middleware Layer**

    ---

    Gateway + DataPool form the sole bridge between Agent Side and Training Side. The two sides never communicate directly, enabling fully asynchronous, non-blocking co-existence of service and training.

    [:octicons-arrow-right-24: Read more](middleware-layer.md)

-   **Production Scenario**

    ---

    The framework is designed for agents that live in the real world: agents that must serve users without interruption while continuously improving from their own interactions.

    [:octicons-arrow-right-24: Read more](production-scenario.md)

</div>

---

## The Closed Loop

These three concepts are not independent features — they form a **flywheel**:

```
  Black-box agent changes base_url (zero code modification)
              │
              ▼
  Production agent can be deployed and serve real users
              │
              ▼
  Real users generate real interaction trajectories
              │
              ▼
  Middleware Layer intercepts and buffers trajectories asynchronously
              │
              ▼
  Trainer fetches batches, updates model weights
              │
              ▼
  Updated weights pushed back → agent improves
              │
              ▼
  Better agent generates higher-quality trajectories
              └─────────────────────────────────────┘
                    (positive feedback loop)
```

Remove any one of the three and the loop breaks:

- Without **base URL integration**, black-box agents need code changes → real-world deployment becomes impractical
- Without the **Middleware Layer**, Agent Side and Training Side must be coupled → training blocks service
- Without the **production scenario** focus, the first two points are just a more convenient RLVR framework — the core value of learning from real user interactions is lost
