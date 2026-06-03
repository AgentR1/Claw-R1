const state = {
  channel: "train",
  steps: [],
  selected: new Set(),
  eventCursor: 0,
  expanded: new Set(),
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    throw new Error(await res.text());
  }
  return res.json();
}

function channel() {
  state.channel = $("channel").value || "train";
  return state.channel;
}

async function loadConfig() {
  const cfg = await api("/api/config");
  $("channel").value = cfg.channel;
  state.channel = cfg.channel;
  $("connection").textContent = cfg.mock
    ? `Mock mode · refresh ${cfg.refresh_interval_ms}ms`
    : `Ray actor ${cfg.actor_name} · refresh ${cfg.refresh_interval_ms}ms`;
}

async function refreshAll() {
  await Promise.all([loadStats(), loadSteps(), loadEvents()]);
}

async function loadStats() {
  const stats = await api(`/api/stats?channel=${encodeURIComponent(channel())}`);
  const keys = [
    ["total_steps", "Steps"],
    ["total_trajectories", "Trajectories"],
    ["complete_trajectories", "Complete"],
    ["ready_prompt_groups", "Ready prompts"],
    ["curated_steps", "Curated"],
    ["event_cursor", "Cursor"],
  ];
  $("stats").innerHTML = keys
    .map(([key, label]) => `<div class="stat"><span>${label}</span><strong>${stats[key] ?? 0}</strong></div>`)
    .join("");
}

function filterQuery() {
  const params = new URLSearchParams({ channel: channel(), limit: "500" });
  if ($("filterPrompt").value) params.set("prompt_uid", $("filterPrompt").value);
  if ($("filterTrajectory").value) params.set("trajectory_uid", $("filterTrajectory").value);
  if ($("filterQuality").value) params.set("quality", $("filterQuality").value);
  if ($("filterTrainable").value) params.set("trainable", $("filterTrainable").value);
  return params.toString();
}

async function loadSteps() {
  const data = await api(`/api/steps?${filterQuery()}`);
  state.steps = data.steps;
  renderSteps();
}

function renderSteps() {
  $("stepsBody").innerHTML = state.steps
    .map((step) => {
      const checked = state.selected.has(step.step_key) ? "checked" : "";
      const expanded = state.expanded.has(step.step_key);
      const quality = step.curation?.quality || "unreviewed";
      const trainable = step.curation?.trainable !== false;
      const reward = step.reward === null || step.reward === undefined ? "missing" : Number(step.reward).toFixed(3);
      const metadata = step.task || step.agent || JSON.stringify(step.metadata || {}).slice(0, 80);
      return `<tr class="${expanded ? "expanded" : ""}">
        <td><input type="checkbox" class="rowSelect" data-key="${step.step_key}" ${checked}></td>
        <td><code>${step.trajectory_uid}</code><br><span class="pill">${step.prompt_uid}</span></td>
        <td>${step.step_index}</td>
        <td>${step.complete ? "complete" : "open"}${step.is_last ? " · last" : ""}</td>
        <td>${tokenPreview(step.prompt_ids, step.prompt_len, "State", step.step_key)}</td>
        <td>${tokenPreview(step.response_ids, step.response_len, "Action", step.step_key)}</td>
        <td>${reward}</td>
        <td>${step.policy_version}</td>
        <td><span class="pill ${qualityClass(quality)}">${quality}</span></td>
        <td>${trainable ? "yes" : "no"}</td>
        <td>${metadata || ""}</td>
      </tr>
      ${expanded ? expandedRow(step) : ""}`;
    })
    .join("");
  document.querySelectorAll(".rowSelect").forEach((box) => {
    box.addEventListener("change", (event) => {
      const key = event.target.dataset.key;
      event.target.checked ? state.selected.add(key) : state.selected.delete(key);
    });
  });
  document.querySelectorAll(".expandTokens").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.key;
      state.expanded.has(key) ? state.expanded.delete(key) : state.expanded.add(key);
      renderSteps();
    });
  });
}

function tokenPreview(tokens = [], count = 0, label, stepKey) {
  const visible = tokens.slice(0, 8).join(", ");
  const suffix = tokens.length > 8 ? ", ..." : "";
  return `<div class="token-preview">
    <code>[${visible}${suffix}]</code>
    <button class="expandTokens" data-key="${stepKey}" type="button">${label} ${count}</button>
  </div>`;
}

function expandedRow(step) {
  return `<tr class="detail-row">
    <td></td>
    <td colspan="10">
      <div class="token-detail">
        <div>
          <strong>State / prompt_ids</strong>
          <code>${formatTokens(step.prompt_ids)}</code>
        </div>
        <div>
          <strong>Action / response_ids</strong>
          <code>${formatTokens(step.response_ids)}</code>
        </div>
      </div>
    </td>
  </tr>`;
}

function formatTokens(tokens = []) {
  return `[${tokens.join(", ")}]`;
}

function qualityClass(quality) {
  if (quality === "good") return "good";
  if (quality === "bad") return "bad";
  if (quality === "needs_reward") return "warn";
  return "";
}

async function loadEvents() {
  const data = await api(`/api/events?channel=${encodeURIComponent(channel())}&cursor=${state.eventCursor}&limit=50`);
  state.eventCursor = data.cursor;
  $("eventCursor").textContent = `cursor ${state.eventCursor}`;
  if (!data.events.length && $("timeline").children.length) return;
  $("timeline").innerHTML = data.events
    .slice()
    .reverse()
    .map((event) => `<div class="event">
      <span class="pill">${event.type}</span>
      <div>
        <code>${event.step_key}</code><br>
        <small>${event.prompt_uid || ""} ${event.trajectory_uid || ""}</small>
      </div>
      <small>#${event.cursor}</small>
    </div>`)
    .join("");
}

async function applyCuration() {
  const updates = [...state.selected].map((step_key) => ({
    step_key,
    quality: $("curationQuality").value,
    trainable: $("curationTrainable").checked,
    tags: $("curationTags").value.split(",").map((x) => x.trim()).filter(Boolean),
    note: $("curationNote").value || undefined,
  }));
  const result = await api("/api/curation", {
    method: "POST",
    body: JSON.stringify({ channel: channel(), updates }),
  });
  $("curationResult").textContent = JSON.stringify(result, null, 2);
  await refreshAll();
}

async function previewBatch() {
  const body = {
    channel: channel(),
    algorithm: $("batchAlgorithm").value,
    batch_size: Number($("batchSize").value),
    n_rollouts: Number($("nRollouts").value),
    max_policy_staleness: Number($("policyFreshness").value),
  };
  const result = await api("/api/batch-preview", { method: "POST", body: JSON.stringify(body) });
  $("batchManifest").textContent = JSON.stringify(result, null, 2);
}

async function previewTree() {
  const prompt = $("filterPrompt").value;
  const params = new URLSearchParams({ channel: channel(), limit: "500" });
  if (prompt) params.set("prompt_uid", prompt);
  const result = await api(`/api/prefix-tree-preview?${params.toString()}`);
  $("treeSummary").innerHTML = [
    ["Sequences", result.sequence_count],
    ["Original tokens", result.original_tokens],
    ["Packed tokens", result.packed_tokens],
    ["Saved tokens", result.saved_tokens],
    ["Token ratio", `${(result.token_ratio * 100).toFixed(1)}%`],
  ]
    .map(([label, value]) => `<div class="stat"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
  $("treeNodes").innerHTML = result.nodes
    .map((node) => `<div class="node">
      <strong>Node ${node.node_id}</strong>
      <small>pos [${node.start_pos}, ${node.end_pos}] · seqs ${node.sequence_ids.join(",")}</small>
      <code>${JSON.stringify(node.tokens.slice(0, 32))}${node.tokens.length > 32 ? " ..." : ""}</code>
    </div>`)
    .join("");
  renderMask(result.attention_mask);
  $("sequencePaths").textContent = JSON.stringify(result.sequence_paths, null, 2);
}

function renderMask(mask) {
  const el = $("mask");
  el.style.gridTemplateColumns = `repeat(${mask.size}, 8px)`;
  el.innerHTML = mask.matrix
    .flatMap((row) => row.map((value) => `<span class="mask-cell ${value ? "on" : ""}"></span>`))
    .join("");
}

function bind() {
  $("refresh").addEventListener("click", refreshAll);
  $("applyFilters").addEventListener("click", loadSteps);
  $("applyCuration").addEventListener("click", applyCuration);
  $("previewBatch").addEventListener("click", previewBatch);
  $("previewTree").addEventListener("click", previewTree);
  $("selectAll").addEventListener("change", (event) => {
    state.selected = event.target.checked ? new Set(state.steps.map((s) => s.step_key)) : new Set();
    renderSteps();
  });
  document.querySelectorAll(".tabs button").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tabs button").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
      button.classList.add("active");
      $(button.dataset.tab).classList.add("active");
    });
  });
}

bind();
loadConfig().then(refreshAll).catch((error) => {
  $("connection").textContent = error.message;
});
