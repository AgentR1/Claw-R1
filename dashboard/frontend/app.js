const state = {
  channels: ["train"],
  steps: [],
  stats: {},
  sync: {},
  treePreview: null,
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
  const el = $("channel");
  if (el?.matches("fieldset")) {
    const selected = [...el.querySelectorAll('input[type="checkbox"]:checked')]
      .map((input) => input.value)
      .filter(Boolean);
    state.channels = selected.length ? selected : ["train"];
  } else if (el?.multiple) {
    const selected = [...el.selectedOptions].map((option) => option.value).filter(Boolean);
    state.channels = selected.length ? selected : ["train"];
  } else {
    state.channels = [(el?.value || "train").trim()].filter(Boolean);
  }
  return state.channels.join(",");
}

async function loadConfig() {
  const cfg = await api("/api/config");
  const configured = (cfg.channel || "train").split(",").map((x) => x.trim()).filter(Boolean);
  state.channels = configured.length ? configured : ["train"];
  if ($("channel").matches("fieldset")) {
    $("channel").querySelectorAll('input[type="checkbox"]').forEach((input) => {
      input.checked = state.channels.includes(input.value);
    });
  } else if ($("channel").multiple) {
    [...$("channel").options].forEach((option) => {
      option.selected = state.channels.includes(option.value);
    });
  } else {
    $("channel").value = state.channels[0];
  }
  $("connection").textContent = `Ray actor ${cfg.actor_name} · sync ${cfg.sync_actor_name} · refresh ${cfg.refresh_interval_ms}ms`;
  return cfg;
}

async function refreshAll() {
  await Promise.all([loadStats(), loadSync(), loadSteps(), loadEvents(), loadOverviewTree()]);
  renderOverview();
}

async function loadSync() {
  state.sync = await api("/api/sync");
  renderConsumption();
}

async function loadStats() {
  const stats = await api(`/api/stats?channel=${encodeURIComponent(channel())}`);
  state.stats = stats;
  const keys = [
    ["total_steps", "Steps"],
    ["total_trajectories", "Trajectories"],
    ["complete_trajectories", "Complete"],
    ["ready_prompt_groups", "Ready prompts"],
    ["fetch_count", "Fetches"],
    ["consumed_prompt_groups", "Consumed"],
  ];
  $("stats").innerHTML = keys
    .map(([key, label]) => `<div class="stat"><span>${label}</span><strong>${stats[key] ?? 0}</strong></div>`)
    .join("");
  renderLifecycle();
  renderConsumption();
}

function eventLabel(type = "") {
  const labels = {
    step_submitted: "step",
    trajectory_completed: "done",
    curation_updated: "curated",
  };
  return labels[type] || type.replaceAll("_", " ");
}

function filterQuery() {
  const params = new URLSearchParams({ channel: channel(), limit: "500" });
  if ($("filterPrompt").value) params.set("prompt_uid", $("filterPrompt").value);
  if ($("filterTrajectory").value) params.set("trajectory_uid", $("filterTrajectory").value);
  if ($("filterAgent").value) params.set("agent", $("filterAgent").value);
  if ($("filterQuality").value) params.set("quality", $("filterQuality").value);
  if ($("filterTrainable").value) params.set("trainable", $("filterTrainable").value);
  return params.toString();
}

async function loadSteps() {
  const data = await api(`/api/steps?${filterQuery()}`);
  state.steps = data.steps;
  renderSteps();
  renderLifecycle();
  renderPoolStrip();
  renderCurationBoard();
  renderOverview();
}

async function loadOverviewTree() {
  try {
    const params = new URLSearchParams({ channel: channel(), limit: "500" });
    state.treePreview = await api(`/api/prefix-tree-preview?${params.toString()}`);
  } catch (_error) {
    state.treePreview = null;
  }
}

function renderSteps() {
  $("stepsBody").innerHTML = state.steps
    .map((step) => {
      const checked = state.selected.has(step.step_key) ? "checked" : "";
      const expanded = state.expanded.has(step.step_key);
      const quality = qualityForStep(step);
      const trainable = step.curation?.trainable !== false;
      const reward = step.reward === null || step.reward === undefined ? "missing" : Number(step.reward).toFixed(3);
      const metadata = metadataLine(step);
      return `<tr class="${expanded ? "expanded" : ""}">
        <td><input type="checkbox" class="rowSelect" data-key="${step.step_key}" ${checked}></td>
        <td><code>${step.trajectory_uid}</code><br><span class="pill">${step.prompt_uid}</span></td>
        <td>${step.step_index}</td>
        <td>${statusPill(step)}</td>
        <td>${tokenPreview(step.prompt_ids, step.prompt_len, "State", step.step_key)}</td>
        <td>${tokenPreview(step.response_ids, step.response_len, "Action", step.step_key)}</td>
        <td>${reward}</td>
        <td>${step.policy_version}</td>
        <td><span class="pill ${qualityClass(quality)}">${quality}</span></td>
        <td>${trainable ? '<span class="pill good">yes</span>' : '<span class="pill bad">no</span>'}</td>
        <td>${metadata || ""}</td>
      </tr>
      ${expanded ? expandedRow(step) : ""}`;
    })
    .join("");
  document.querySelectorAll(".rowSelect").forEach((box) => {
    box.addEventListener("change", (event) => {
      const key = event.target.dataset.key;
      event.target.checked ? state.selected.add(key) : state.selected.delete(key);
      renderCurationBoard();
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

function metadataLine(step) {
  const source = step.source || step.metadata?.source || step.metadata?.data_source || "rollout";
  const agent = step.agent || step.metadata?.agent || "agent";
  const task = step.task || step.metadata?.task || step.metadata?.dataset || "task";
  const tool = step.metadata?.tool || step.metadata?.tool_name || "";
  return `<strong>${task}</strong><br><small>${source} · ${agent}${tool ? ` · ${tool}` : ""}</small>`;
}

function statusPill(step) {
  const status = step.complete ? "complete" : "collecting";
  const cls = step.complete ? "good" : "warn";
  return `<span class="pill ${cls}">${status}</span>${step.is_last ? '<br><small>terminal step</small>' : ""}`;
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
        <div>
          <strong>Step metadata</strong>
          <code>${JSON.stringify(step.metadata || {}, null, 2)}</code>
        </div>
        <div>
          <strong>Curation state</strong>
          <code>${JSON.stringify(step.curation || {}, null, 2)}</code>
        </div>
      </div>
    </td>
  </tr>`;
}

function formatTokens(tokens = []) {
  return `[${tokens.join(", ")}]`;
}

function qualityClass(quality) {
  if (quality === "correct" || quality === "good") return "good";
  if (quality === "incorrect" || quality === "bad") return "bad";
  if (quality === "needs_reward") return "warn";
  return "";
}

async function loadEvents() {
  const data = await api(`/api/events?channel=${encodeURIComponent(channel())}&cursor=${state.eventCursor}&limit=50`);
  state.eventCursor = data.cursor;
  $("eventCursor").textContent = `cursor ${state.eventCursor}`;
  if ($("collectionCursor")) $("collectionCursor").textContent = `cursor ${state.eventCursor}`;
  if (!data.events.length && $("timeline").children.length) return;
  const markup = data.events
    .slice()
    .reverse()
    .map((event) => `<div class="event">
      <span class="pill event-type" title="${event.type}">${eventLabel(event.type)}</span>
      <div>
        <code>${event.step_key}</code><br>
        <small>${event.prompt_uid || event.payload?.prompt_uid || ""} ${event.trajectory_uid || event.payload?.trajectory_uid || ""}</small>
      </div>
      <small>#${event.cursor}</small>
    </div>`)
    .join("");
  $("timeline").innerHTML = markup;
  if ($("collectionTimeline")) $("collectionTimeline").innerHTML = markup;
}

function renderLifecycle() {
  if (!$("pipeline") || !state.steps.length) return;
  const reviewed = state.steps.filter((s) => qualityForStep(s) !== "missing").length;
  const trainable = state.steps.filter((s) => s.curation?.trainable !== false).length;
  const rewarded = state.steps.filter((s) => s.reward !== null && s.reward !== undefined).length;
  const ready = state.stats.ready_prompt_groups ?? new Set(state.steps.map((s) => s.prompt_uid)).size;
  const stages = [
    ["Collect", state.steps.length, "steps"],
    ["Reward", rewarded, "scored"],
    ["Curate", reviewed, "reviewed"],
    ["Optimize", promptCount(), "groups"],
    ["Consume", ready, "ready"],
  ];
  $("pipeline").innerHTML = stages
    .map(([label, value, hint], idx) => `<div class="pipe-card">
      <span>0${idx + 1}</span>
      <strong>${label}</strong>
      <b>${value}</b>
      <small>${hint}</small>
    </div>`)
    .join("");
  renderSourceGrid();
}

function renderOverview() {
  if (!$("overviewFlow") || !state.steps.length) return;
  const latestPolicy = state.sync.policy_version ?? Math.max(...state.steps.map((s) => s.policy_version || 0));
  const rewarded = state.steps.filter((s) => s.reward !== null && s.reward !== undefined).length;
  const reviewed = state.steps.filter((s) => qualityForStep(s) !== "missing").length;
  const trainable = state.steps.filter((s) => s.curation?.trainable !== false).length;
  const highReward = state.steps.filter((s) => Number(s.reward ?? -1) >= 0.7).length;
  const tokenTotal = sum(state.steps.map((s) => s.token_count || ((s.prompt_len || 0) + (s.response_len || 0))));
  const ready = state.stats.ready_prompt_groups ?? promptCount();
  const tree = state.treePreview;
  const tokenRatio = tree ? `${(tree.token_ratio * 100).toFixed(1)}%` : "pending";
  const savedTokens = tree ? compact(tree.saved_tokens) : "pending";

  $("overviewMetrics").innerHTML = [
    ["Steps", state.steps.length],
    ["Trajectories", state.stats.total_trajectories ?? new Set(state.steps.map((s) => s.trajectory_uid)).size],
    ["Prompts", promptCount()],
    ["Tokens", compact(tokenTotal)],
  ]
    .map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`)
    .join("");

  const stages = [
    ["Collect", state.steps.length, "steps"],
    ["Reward", `${rewarded}/${state.steps.length}`, "scored"],
    ["Curate", highReward, "candidates"],
    ["Optimize", tokenRatio, "ratio"],
    ["Consume", ready, "groups"],
  ];
  $("overviewFlow").innerHTML = stages
    .map(([label, value, hint], idx) => `<div class="flow-step">
      <span>0${idx + 1}</span>
      <strong>${label}</strong>
      <b>${value}</b>
      <small>${hint}</small>
    </div>`)
    .join("");

  $("overviewHealth").innerHTML = [
    ["Policy", `v${latestPolicy ?? 0}`, "synced"],
    ["Trainable", `${trainable}/${state.steps.length}`, "eligible"],
    ["Fetches", state.stats.fetch_count ?? 0, "batches"],
    ["Synced", state.sync.sync_count ?? 0, "times"],
  ]
    .map(([label, value, hint]) => `<div class="health-row">
      <span>${label}</span>
      <strong>${value}</strong>
      <small>${hint}</small>
    </div>`)
    .join("");

  renderOverviewCuration();
  renderSourceGrid();
  renderOverviewOptimization();
  renderOverviewConsumption();
}

function renderSourceGrid() {
  const targets = document.querySelectorAll("[data-source-grid]");
  if (!targets.length || !state.steps.length) return;
  const groups = groupBy(state.steps, (s) => s.source || s.metadata?.source || s.metadata?.data_source || "rollout");
  const markup = Object.entries(groups)
    .map(([source, rows]) => {
      const rewards = rows.map((s) => s.reward).filter((x) => x !== null && x !== undefined);
      const avg = rewards.length ? mean(rewards).toFixed(2) : "n/a";
      return `<div class="source-card">
        <span class="pill">${source}</span>
        <strong>${rows.length} steps</strong>
        <small>${new Set(rows.map((s) => s.trajectory_uid)).size} traj · r ${avg}</small>
      </div>`;
    })
    .join("");
  targets.forEach((target) => {
    target.innerHTML = markup;
  });
}

function renderPoolStrip() {
  if (!$("poolStrip") || !state.steps.length) return;
  const tokenTotal = sum(state.steps.map((s) => s.token_count || ((s.prompt_len || 0) + (s.response_len || 0))));
  const missingReward = state.steps.filter((s) => s.reward === null || s.reward === undefined).length;
  const latestPolicy = Math.max(...state.steps.map((s) => s.policy_version || 0));
  const agents = new Set(state.steps.map((s) => s.agent).filter(Boolean)).size;
  $("poolStrip").innerHTML = [
    ["Token volume", compact(tokenTotal)],
    ["Missing reward", missingReward],
    ["Latest policy", `v${latestPolicy}`],
    ["Agents", agents],
  ]
    .map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
}

function renderCurationBoard() {
  if (!$("curationBoard") || !state.steps.length) {
    renderOverviewCuration();
    return;
  }
  const selected = state.selected.size;
  const correct = state.steps.filter((s) => qualityForStep(s) === "correct").length;
  const incorrect = state.steps.filter((s) => qualityForStep(s) === "incorrect").length;
  const missing = state.steps.filter((s) => qualityForStep(s) === "missing").length;
  const blocked = state.steps.filter((s) => s.curation?.trainable === false).length;
  $("curationBoard").innerHTML = [
    ["Selected", selected, "Rows staged for curation"],
    ["Correct", correct, "Reward-positive steps"],
    ["Incorrect", incorrect, "Zero or negative reward"],
    ["Missing", missing, "No reward yet"],
    ["Excluded", blocked, "Marked non-trainable"],
  ]
    .map(([label, value, hint]) => `<div class="curation-card"><span>${label}</span><strong>${value}</strong><small>${hint}</small></div>`)
    .join("");
  renderOverviewCuration();
  renderReviewQueue();
}

function renderOverviewCuration() {
  if (!$("overviewCurationBoard") || !state.steps.length) return;
  const correct = state.steps.filter((s) => qualityForStep(s) === "correct").length;
  const incorrect = state.steps.filter((s) => qualityForStep(s) === "incorrect").length;
  const missingReward = state.steps.filter((s) => s.reward === null || s.reward === undefined).length;
  const trainable = state.steps.filter((s) => s.curation?.trainable !== false).length;
  $("overviewCurationBoard").innerHTML = [
    ["Correct", correct],
    ["Incorrect", incorrect],
    ["No reward", missingReward],
    ["Trainable", trainable],
  ]
    .map(([label, value]) => `<div class="mini-signal"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
}

function renderOverviewOptimization() {
  if (!$("overviewOptimization")) return;
  const tree = state.treePreview;
  if (!tree) {
    $("overviewOptimization").innerHTML = `<div class="empty-note">Prefix-tree preview pending</div>`;
    return;
  }
  const shared = sharedNodes(tree).length;
  const branches = branchNodes(tree).length;
  $("overviewOptimization").innerHTML = `<div class="optimization-meter">
    <div><span>Original</span><strong>${compact(tree.original_tokens)}</strong></div>
    <div><span>Packed</span><strong>${compact(tree.packed_tokens)}</strong></div>
    <div><span>Saved</span><strong>${compact(tree.saved_tokens)}</strong></div>
    <div><span>Ratio</span><strong>${(tree.token_ratio * 100).toFixed(1)}%</strong></div>
    <div><span>Shared nodes</span><strong>${shared}</strong></div>
    <div><span>Branches</span><strong>${branches}</strong></div>
  </div>
  <div class="ratio-bar"><span style="width:${Math.max(4, tree.token_ratio * 100)}%"></span></div>
  <small class="preview-note">Live prefix-tree merge preview · ${tree.prompt_uid || "latest prompt"}</small>`;
}

function renderOverviewConsumption() {
  renderConsumption("overviewConsumption");
}

function renderReviewQueue() {
  if (!$("reviewQueue")) return;
  const rows = [...state.steps]
    .sort((a, b) => scoreForReview(b) - scoreForReview(a))
    .slice(0, 6);
  $("reviewQueue").innerHTML = rows
    .map((s) => `<div class="review-item">
      <div><code>${s.step_key}</code><br><small>${s.prompt_uid} · ${s.agent || "agent"} · policy v${s.policy_version}</small></div>
      <div><span class="pill ${Number(s.reward ?? 0) >= 0.7 ? "good" : Number(s.reward ?? 0) < 0.25 ? "bad" : "warn"}">${s.reward ?? "missing"}</span></div>
    </div>`)
    .join("");
}

function scoreForReview(step) {
  const reward = Number(step.reward ?? 0);
  const unreviewed = (step.curation?.quality || "unreviewed") === "unreviewed" ? 1 : 0;
  return reward + unreviewed + (step.reward === null || step.reward === undefined ? 0.5 : 0);
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
  renderManifest(result);
  $("batchManifest").textContent = JSON.stringify(result, null, 2);
}

function renderManifest(result) {
  if (!$("manifestSummary")) return;
  $("manifestSummary").innerHTML = [
    ["Algorithm", result.algorithm],
    ["Prompt groups", result.prompt_uids?.length || 0],
    ["Selected steps", result.step_count],
    ["Token budget", compact(result.token_count || 0)],
    ["Freshness", result.max_policy_staleness === null ? "any" : `<= ${result.max_policy_staleness}`],
  ]
    .map(([label, value]) => `<div class="manifest-card"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
  const groups = result.prompt_groups || [];
  $("manifestGroups").innerHTML = groups
    .map((group) => `<div class="group-row">
      <div><code>${group.prompt_uid}</code><br><small>${group.trajectory_count} trajectories · ${group.step_count} steps</small></div>
      <div><span>reward</span><strong>${formatMaybe(group.reward_mean)}</strong></div>
      <div><span>tokens</span><strong>${compact(group.token_count)}</strong></div>
      <div><span>quality</span><strong>${qualityMix(group.quality_counts)}</strong></div>
    </div>`)
    .join("");
}

function renderConsumption(targetId = "consumptionStatus") {
  const target = $(targetId);
  if (!target) return;
  const lastFetch = state.stats.last_fetch || {};
  const lastSync = state.sync.last_sync || {};
  target.innerHTML = [
    ["Fetch batches", state.stats.fetch_count ?? 0, "trainer fetch_batch calls"],
    ["Consumed prompts", state.stats.consumed_prompt_groups ?? 0, "prompt groups"],
    ["Policy version", `v${state.sync.policy_version ?? state.sync.current_version ?? 0}`, "weight sync version"],
    ["Weight syncs", state.sync.sync_count ?? 0, "completed syncs"],
    ["Last fetch", lastFetch.batch_size ? `${lastFetch.batch_size} prompts` : "pending", lastFetch.prompt_uids?.slice(0, 3).join(" · ") || "no batch fetched"],
    ["Last sync step", lastSync.global_steps ?? "pending", lastSync.validate ? "validation sync" : "train sync"],
  ]
    .map(([label, value, hint]) => `<div class="consume-row">
      <span>${label}</span>
      <strong>${value}</strong>
      <small>${hint}</small>
    </div>`)
    .join("");
}

function qualityForStep(step) {
  const curated = step.curation?.quality;
  if (curated && curated !== "unreviewed") return curated;
  if (step.reward === null || step.reward === undefined) return "missing";
  return Number(step.reward) > 0 ? "correct" : "incorrect";
}

async function previewTree() {
  const prompt = $("filterPrompt").value;
  const params = new URLSearchParams({ channel: channel(), limit: "500" });
  if (prompt) params.set("prompt_uid", prompt);
  const result = await api(`/api/prefix-tree-preview?${params.toString()}`);
  state.treePreview = result;
  const shared = sharedNodes(result).length;
  const branches = branchNodes(result).length;
  $("treeSummary").innerHTML = [
    ["Sequences", result.sequence_count],
    ["Original tokens", result.original_tokens],
    ["Packed tokens", result.packed_tokens],
    ["Saved tokens", result.saved_tokens],
    ["Token ratio", `${(result.token_ratio * 100).toFixed(1)}%`],
    ["Shared nodes", shared],
    ["Branch points", branches],
  ]
    .map(([label, value]) => `<div class="stat"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
  $("treeNodes").innerHTML = result.nodes
    .map((node) => `<div class="node ${node.sequence_ids.length > 1 ? "shared" : ""} ${node.child_ids.length > 1 ? "branch" : ""}">
      <strong>Node ${node.node_id}</strong>
      <small>pos [${node.start_pos}, ${node.end_pos}] · seqs ${node.sequence_ids.join(",")} · ${node.sequence_ids.length > 1 ? "shared prefix" : "private span"}${node.child_ids.length > 1 ? " · branch point" : ""}</small>
      <code>${JSON.stringify(node.tokens.slice(0, 32))}${node.tokens.length > 32 ? " ..." : ""}</code>
    </div>`)
    .join("");
  $("packedRibbon").innerHTML = renderPackedRibbon(result);
  renderMask(result.attention_mask);
  $("sequencePaths").textContent = JSON.stringify(result.sequence_paths, null, 2);
}

function renderPackedRibbon(result) {
  if (!result.nodes?.length) return "";
  const maxTokens = Math.max(1, result.packed_tokens || 1);
  return result.nodes
    .slice(0, 18)
    .map((node) => {
      const width = Math.max(7, (node.num_tokens / maxTokens) * 100);
      const cls = `${node.sequence_ids.length > 1 ? "shared" : ""} ${node.child_ids.length > 1 ? "branch" : ""}`;
      return `<div class="ribbon-node ${cls}" style="flex-basis:${width}%">
        <span>N${node.node_id}</span>
        <small>${node.num_tokens} tok · ${node.sequence_ids.length} seq</small>
      </div>`;
    })
    .join("");
}

function sharedNodes(tree) {
  return (tree.nodes || []).filter((node) => node.sequence_ids.length > 1);
}

function branchNodes(tree) {
  return (tree.nodes || []).filter((node) => node.child_ids.length > 1);
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
  $("channel").querySelectorAll('input[type="checkbox"]').forEach((input) => input.addEventListener("change", () => {
    state.eventCursor = 0;
    refreshAll();
  }));
  $("applyFilters").addEventListener("click", loadSteps);
  $("applyCuration").addEventListener("click", applyCuration);
  $("previewBatch").addEventListener("click", previewBatch);
  $("previewTree").addEventListener("click", previewTree);
  $("selectAll").addEventListener("change", (event) => {
    state.selected = event.target.checked ? new Set(state.steps.map((s) => s.step_key)) : new Set();
    renderSteps();
    renderCurationBoard();
  });
  document.querySelectorAll(".tabs button").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tabs button").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
      button.classList.add("active");
      $(button.dataset.tab).classList.add("active");
      if (button.dataset.tab === "optimize" && !$("treeSummary").children.length) {
        previewTree();
      }
    });
  });
}

bind();
loadConfig().then((cfg) => {
  refreshAll();
  setInterval(refreshAll, cfg.refresh_interval_ms || 2000);
}).catch((error) => {
  $("connection").textContent = error.message;
});

function groupBy(rows, keyFn) {
  return rows.reduce((acc, row) => {
    const key = keyFn(row);
    acc[key] ||= [];
    acc[key].push(row);
    return acc;
  }, {});
}

function sum(values) {
  return values.reduce((acc, value) => acc + Number(value || 0), 0);
}

function mean(values) {
  return sum(values) / Math.max(1, values.length);
}

function promptCount() {
  return new Set(state.steps.map((s) => s.prompt_uid)).size;
}

function compact(value) {
  return Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value || 0);
}

function formatMaybe(value) {
  return value === null || value === undefined ? "n/a" : Number(value).toFixed(2);
}

function qualityMix(counts = {}) {
  const entries = Object.entries(counts);
  if (!entries.length) return "unreviewed";
  return entries.map(([k, v]) => `${k}:${v}`).join(" · ");
}
