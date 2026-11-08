"use strict";
// Read-only views over /api/*. Every string from a run is inserted as text, never as HTML.

const ARMS = {
  naive: ["Chapter only", "with the chapter alone"],
  glossary: ["Glossary", "with the glossary"],
  contextual: ["Memory + glossary", "with memory and the glossary"],
  harness: ["Full harness", "after the full harness"],
  supplied_draft: ["Supplied draft", "in the supplied draft"],
};
const STATUS = {
  delivered: ["Delivered", "good"],
  completed: ["Completed", "good"],
  delivered_with_warnings: ["Delivered with warnings", "warn"],
  failed_partial: ["Partial: some chapters failed", "bad"],
  failed_packaging: ["Book packaging failed", "bad"],
  failed: ["Failed", "bad"],
  not_started: ["Not started", ""],
  running: ["Unfinished", "warn"],
};
const SHORT_STATUS = { delivered_with_warnings: "Warnings", failed_partial: "Partial", failed_packaging: "Not packaged" };
const LINES = /\r\n|\r|\n|\u2028|\u2029|\v|\f|\x1c|\x1d|\x1e|\x85/;
const OPERATIONS = {
  naive: ["Translate the chapter alone", "Chapter only"],
  glossary: ["Translate with the glossary", "Glossary"],
  memory_extract: ["Read the chapter into story memory", "Memory"],
  contextual: ["Translate with memory and the glossary", "Memory + glossary"],
  term_alignment: ["Align glossary terms in the draft", "Full harness"],
  segment_review: ["Review the draft against the source", "Full harness"],
};
const PROVIDERS = {
  deepseek: ["DeepSeek", "deepseek-flash"],
  openai: ["OpenAI", "Model ID your key can use"],
  anthropic: ["Anthropic (OpenAI-compatible)", "Model ID your key can use"],
  openrouter: ["OpenRouter", "provider/model"],
  custom: ["Other OpenAI-compatible endpoint", "Model ID"],
};
const view = document.getElementById("view");
const cache = { index: null, setup: null, runs: new Map(), diffs: new Map() };
let animatedRun = null;
let poller = null;
let apiKey = "";  // Page memory only: never stored, sent only with a start request.

// Remembered per browser: everything on the form except the API key.
const settings = Object.assign({ provider: "deepseek", model: "", base_url: "", input_price: "", output_price: "",
  max_usd: "", compare: true, limit: 3 }, readStored());
function readStored() {
  try { return JSON.parse(localStorage.getItem("workbench.settings") || "{}"); } catch { return {}; }
}
function remember() {
  try { localStorage.setItem("workbench.settings", JSON.stringify(settings)); } catch { /* private mode */ }
}

const armName = (arm) => (ARMS[arm] || [String(arm).replace(/_/g, " ")])[0];
const armPhrase = (arm) => (ARMS[arm] || [, `with ${String(arm).replace(/_/g, " ")}`])[1];
const plural = (n, word, many = word + "s") => `${n.toLocaleString()} ${n === 1 ? word : many}`;
// Same rule as workbench.py: each non-empty line is a paragraph.
const paragraphs = (text) => (text || "").split(LINES).map((line) => line.trim()).filter(Boolean);
const baseName = (id) => id.split("/").pop();
// A run or report at the scanned folder itself has the id "."; the server sends its folder name.
const displayName = (item) => (item.id === "." ? item.name : baseName(item.id));
const parentPath = (id) => id.split("/").slice(0, -1).join(" / ");
const runFile = (id, path) => `/runs/${id.split("/").map(encodeURIComponent).join("/")}/${path}`;

function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value == null || value === false) continue;
    if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child != null && child !== false) el.append(child instanceof Node ? child : String(child));
  }
  return el;
}

function statusTag(status, { short = false, dot = false } = {}) {
  const [label, tone] = STATUS[status] || [String(status || "Unknown").replace(/_/g, " "), ""];
  if (dot) return h("span", { class: `status ${tone}`, title: label }, h("span", { class: "sr" }, label));
  return h("span", { class: `status ${tone}`, title: label }, short ? SHORT_STATUS[status] || label : label);
}

function modeLabel(mode) {
  if (mode === "live model") return "Live model";
  if (mode === "cache replay") return "Replay of recorded calls";
  return mode ? mode[0].toUpperCase() + mode.slice(1) : "Not recorded";
}

function duration(seconds) {
  if (seconds == null) return "Not recorded";
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  return `${Math.floor(seconds / 60)} min ${Math.round(seconds % 60)} s`;
}

const usd = (value) => (value == null ? "No price given" : value === 0 ? "$0" : `$${value < 0.1 ? value.toFixed(4) : value.toFixed(2)}`);
const count = (value) => (value == null ? "Not reported" : value.toLocaleString());

async function api(path) {
  const response = await fetch(path, { cache: "no-store" });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw Object.assign(new Error(body.error || `The workbench answered ${response.status}`), { status: response.status });
  return body;
}

/* Routing: everything a view needs lives in the hash, so reloads and links keep their place. */
function route() {
  return Object.fromEntries(new URLSearchParams(location.hash.slice(1)));
}
function href(patch, base = route()) {
  const next = { ...base, ...patch };
  for (const key of Object.keys(next)) if (next[key] == null || next[key] === "") delete next[key];
  return "#" + new URLSearchParams(next).toString();
}
function go(patch) {
  location.hash = href(patch);
}

async function render() {
  const params = route();
  if (poller && params.job !== poller.id) stopPolling();
  try {
    if (!cache.index) await refreshIndex();
    if (params.job) {
      markCurrent(params.job);
      return activityPage(params.job);
    }
    if (!params.run) {
      markCurrent("new");
      return composer();
    }
    markCurrent(params.run);
    if (!cache.runs.has(params.run)) {
      view.replaceChildren(h("p", { class: "muted" }, "Opening the run…"));
      cache.runs.set(params.run, await api(`/api/run?id=${encodeURIComponent(params.run)}`));
    }
    await renderRun(cache.runs.get(params.run), params);
  } catch (error) {
    view.replaceChildren(h("div", { class: "notice bad" }, error.message,
      ". Check that the workbench is still running, then reload."));
  }
}

async function refreshIndex() {
  [cache.index, cache.setup] = await Promise.all([api("/api/runs"), api("/api/setup")]);
  renderRail();
  markCurrent(route().job || route().run || "new");
}

function renderRail() {
  const { root, runs, reports } = cache.index;
  const active = cache.setup && cache.setup.active;
  const rootLabel = document.getElementById("root-path");
  rootLabel.textContent = root.split(/[\\/]/).filter(Boolean).slice(-1)[0] || root;
  rootLabel.title = root;
  const list = document.getElementById("run-list");
  const items = [h("a", { class: "new-run", href: "#", "data-run": "new", onclick: closeRail }, "New run")];
  if (active) {
    items.push(h("h2", { class: "rail-group" }, "Running now"),
      h("a", { class: "run-link", href: href({ job: active.id }, {}), "data-run": active.id, onclick: closeRail },
        h("span", { class: "parent" }, parentPath(active.id)), h("span", { class: "name" }, baseName(active.id)),
        h("span", { class: "meta" }, h("span", { class: "status live" }, "Running"))));
  }
  if (runs.length) {
    items.push(h("h2", { class: "rail-group" }, "Demo runs"));
    for (const run of runs) {
      items.push(h("a", { class: "run-link", href: href({ run: run.id }, {}), "data-run": run.id, onclick: closeRail },
        parentPath(run.id) && h("span", { class: "parent" }, parentPath(run.id)),
        h("span", { class: "name" }, displayName(run)),
        h("span", { class: "meta" }, statusTag(run.status, { short: true }), plural(run.chapters, "chapter"))));
    }
  } else {
    items.push(h("p", { class: "rail-empty" }, "No demo runs here yet."));
  }
  if (reports.length) {
    items.push(h("h2", { class: "rail-group" }, "Other reports"));
    for (const report of reports) {
      items.push(h("a", { class: "run-link", href: runFile(report.id, "report.html"), target: "_blank", rel: "noopener" },
        parentPath(report.id) && h("span", { class: "parent" }, parentPath(report.id)),
        h("span", { class: "name" }, displayName(report)),
        h("span", { class: "meta" }, "Saved HTML report")));
    }
  }
  list.replaceChildren(...items);
}

function markCurrent(id) {
  for (const link of document.querySelectorAll("#run-list [data-run]")) {
    if (link.dataset.run === id) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
}

function closeRail() {
  document.querySelector(".rail").classList.remove("open");
  document.getElementById("rail-toggle").setAttribute("aria-expanded", "false");
}

/* Run frame */
async function renderRun(run, params) {
  const { results, summary } = run;
  const tab = ["overview", "chapters", "memory", "activity"].includes(params.tab) ? params.tab : "overview";
  const facts = run.memory.filter((record) => record.type === "memory_entry").length;
  const provider = run.config.provider || {};
  const files = {
    "delivery/book.txt": "Read the translation",
    "delivery/book.epub": "EPUB",
    "report.html": "Saved report",
    "results.json": "results.json",
  };
  const tabLink = (key, label, extra) => h("a", {
    href: href({ tab: key }), "aria-current": tab === key ? "page" : null,
  }, label, extra != null && h("span", { class: "count" }, extra));

  const head = h("header", {},
    h("div", { class: "run-head" },
      h("div", {}, parentPath(run.id) && h("div", { class: "crumb" }, parentPath(run.id)),
        h("h2", {}, displayName(run.summary))),
      h("nav", { class: "files", "aria-label": "Run files" },
        run.files.filter((path) => files[path]).map((path) =>
          h("a", { href: runFile(run.id, path), target: "_blank", rel: "noopener" }, files[path])))),
    h("dl", { class: "facts" },
      fact("Status", statusTag(results.status)),
      fact("Evidence", modeLabel(results.mode)),
      provider.model && fact("Model", provider.provider ? `${provider.model} (${provider.provider})` : provider.model),
      fact("Chapters", results.chapters.length),
      fact("Recorded", new Date(summary.modified).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }))),
    h("nav", { class: "tabs", "aria-label": "Run views" },
      tabLink("overview", "Overview"),
      tabLink("chapters", "Chapters", results.chapters.length),
      tabLink("memory", "Story memory", facts || null),
      summary.activity && tabLink("activity", "Activity")));

  const body = tab === "chapters" ? await chaptersView(run, params)
    : tab === "memory" ? memoryView(run, params)
      : tab === "activity" && summary.activity ? await recordedActivity(run.id) : overviewView(run);
  view.replaceChildren(...[head, evidenceNotice(results), body].filter(Boolean));
}

function fact(label, value) {
  return h("div", {}, h("dt", {}, label), h("dd", {}, value));
}

function evidenceNotice(results) {
  if (results.mode === "cache replay") {
    return h("p", { class: "notice" }, "This run replayed recorded model calls. Nothing was sent to a provider, so its numbers repeat the original run rather than adding new evidence.");
  }
  if (results.mode && results.mode !== "live model") {
    return h("p", { class: "notice" }, `Evidence: ${results.mode}. The text comes from a script, not a model, so use it to inspect the mechanics, not translation quality.`);
  }
  return null;
}

function missingOutputs(results) {
  return results.chapters.filter((chapter) => results.arms.some((arm) => !(chapter.texts || {})[arm])).length;
}

/* Overview */
function overviewView(run) {
  const { results } = run;
  const summary = results.summary || {};
  const all = (summary.groups || {}).all || {};
  const arms = results.arms;
  const missing = missingOutputs(results);
  const parts = [];

  let headline = "This run recorded no checks. Inspect the chapters and the harness log instead.";
  const first = all[arms[0]], last = all[arms[arms.length - 1]];
  if (missing) {
    headline = `No comparable score: ${missing} of ${plural(results.chapters.length, "chapter")} lack text from at least one setup.`;
  } else if (summary.seeded_errors && summary.clean_controls) {
    const seeded = summary.seeded_errors, controls = summary.clean_controls;
    headline = `${seeded.fixed} of ${seeded.total} seeded errors fixed, and ${controls.passed} of ${controls.total} clean passages still pass their checks.`;
  } else if (first && last && first.total) {
    headline = `${first.passed} of ${first.total} checks passed ${armPhrase(arms[0])}; ${last.passed} of ${last.total} ${armPhrase(arms[arms.length - 1])}.`;
  }
  parts.push(h("p", { class: "headline" }, headline));
  const note = [summary.metric, summary.note].filter(Boolean).map((text) => text.replace(/\.$/, ""));
  if (note.length) parts.push(h("p", { class: "metric-note" }, note.join(". ") + "."));
  if (missing) {
    parts.push(h("p", { class: "notice bad" }, "Missing text counts as a failed check in the ledger below, so differences between setups are not a measurement. Open the chapters to see where the run stopped."));
  }

  parts.push(h("section", { class: "section" },
    h("h3", {}, "Check ledger"),
    h("p", {}, "One square per check, in chapter order. Setups run left to right, each adding one step to the one before."),
    ledger(run)));

  const gains = (summary.gains || []).filter((row) => row.percentage_points != null);
  const overall = gains.filter((row) => row.category === "all");
  const areas = gains.filter((row) => row.category !== "all");
  const gainTable = (rows, byArea) => h("div", { class: "scroll" }, h("table", { class: "table" },
    h("thead", {}, h("tr", {}, h("th", {}, "Step"), byArea && h("th", {}, "Area"), h("th", { class: "num" }, "Fixed"),
      h("th", { class: "num" }, "Broke"), h("th", { class: "num" }, "Net change"))),
    h("tbody", {}, rows.map((row) => h("tr", {},
      h("td", {}, `${armName(row.before)} to ${armName(row.after)}`),
      byArea && h("td", {}, row.category),
      h("td", { class: `num ${row.fixed ? "fixed" : ""}` }, row.fixed),
      h("td", { class: `num ${row.regressed ? "regressed" : ""}` }, row.regressed),
      h("td", { class: "num" }, `${row.percentage_points >= 0 ? "+" : "−"}${Math.abs(row.percentage_points).toFixed(1)} pts`))))));
  if (gains.length) {
    parts.push(h("section", { class: "section" },
      h("h3", {}, "What each step changed"),
      h("p", {}, "Checks a step fixed and checks it broke, counted separately. Net change is in percentage points."),
      overall.length ? gainTable(overall, false) : null,
      areas.length ? h("details", { class: "context", open: !overall.length }, h("summary", {}, `By area (${new Set(areas.map((row) => row.category)).size})`),
        gainTable(areas, true)) : null));
  }

  parts.push(h("section", { class: "section two-col" }, costBlock(results), setupBlock(run)));
  return h("div", { class: "measure" }, parts);
}

function ledger(run) {
  const { results } = run;
  const arms = results.arms;
  const rows = new Map();
  for (const chapter of results.chapters) {
    for (const [index, arm] of arms.entries()) {
      for (const check of ((chapter.arms || {})[arm] || {}).checks || []) {
        const key = check.source != null ? `term:${check.source}\u0000${check.target}` : `case:${chapter.id}`;
        if (!rows.has(key)) {
          rows.set(key, { check, chapter, category: check.category || "other", cells: arms.map(() => []) });
        }
        rows.get(key).cells[index].push({ check, chapter });
      }
    }
  }
  if (!rows.size) {
    return h("div", { class: "empty" }, h("strong", {}, "No checks in this run"),
      "Give demo.py a glossary with --glossary to count how often each term is used as specified.");
  }
  const scores = ((results.summary || {}).groups || {}).all || {};
  const groups = new Map();
  for (const row of rows.values()) {
    if (!groups.has(row.category)) groups.set(row.category, []);
    groups.get(row.category).push(row);
  }
  const animate = animatedRun !== run.id;
  animatedRun = run.id;

  const terms = [...rows.values()].some((row) => row.check.source != null);
  const header = h("tr", {}, h("th", { scope: "col" }, h("span", { class: "arm-name" }, terms ? "Glossary term" : "Case")),
    arms.map((arm) => {
      const score = scores[arm];
      return h("th", { scope: "col" }, h("span", { class: "arm-name" }, armName(arm)),
        score && h("span", { class: "arm-score" }, `${score.passed}/${score.total}`,
          score.rate != null && h("small", {}, `${Math.round(100 * score.rate)}%`)));
    }));
  const body = [];
  // Authored cases already name their area; grouping them would add a header per row.
  const ordered = terms ? [...groups].sort(([a], [b]) => a.localeCompare(b)) : [[null, [...rows.values()]]];
  for (const [category, members] of ordered) {
    if (category != null) body.push(h("tr", { class: "group" }, h("th", { colspan: arms.length + 1, scope: "colgroup" }, category)));
    for (const row of members) {
      const label = row.check.source != null
        ? [h("span", { class: "term-zh", lang: "zh" }, row.check.source), h("span", { class: "term-en" }, row.check.target)]
        : [h("span", {}, row.chapter.name)];
      body.push(h("tr", {}, h("th", { scope: "row" }, label),
        row.cells.map((cells, step) => h("td", {}, h("div", { class: "squares", style: `--step:${step}` },
          cells.map(({ check, chapter }) => square(check, chapter, arms[step])))))));
    }
  }
  return h("div", {},
    h("div", { class: "ledger-wrap" }, h("table", { class: `ledger${animate ? " animate" : ""}` },
      h("thead", {}, header), h("tbody", {}, body))),
    h("div", { class: "legend", "aria-hidden": "true" },
      h("span", {}, h("i", { class: "sq pass" }), "Passed"),
      h("span", {}, h("i", { class: "sq" }), terms ? "Specified term missing" : "Failed"),
      terms && h("span", {}, h("i", { class: "sq blocked" }), "Blocked variant used"),
      h("span", {}, h("i", { class: "sq missing" }), "No text from this setup")));
}

function square(check, chapter, arm) {
  let kind = "", state = "specified term missing";
  if (check.passed) [kind, state] = ["pass", "passed"];
  else if (check.output_present === false) [kind, state] = ["missing", "no text from this setup"];
  else if (check.blocked_variant_present) [kind, state] = ["blocked", "blocked variant used"];
  else if (check.source == null) state = "failed";
  const where = [chapter.name, check.segment].filter(Boolean).join(", ");
  return h("i", { class: `sq ${kind}`, role: "img", title: `${armName(arm)}, ${where}: ${state}`, "aria-label": `${armName(arm)}, ${where}: ${state}` });
}

function costBlock(results) {
  const total = results.resources || {};
  const perArm = results.arms.map((arm) => {
    // Unknown usage stays unknown instead of counting as zero.
    const sum = { calls: 0, input_tokens: 0, output_tokens: 0, estimated_usd: 0 };
    for (const chapter of results.chapters) {
      const resources = ((chapter.arms || {})[arm] || {}).resources;
      if (!resources) continue;
      for (const key of Object.keys(sum)) {
        sum[key] = sum[key] == null || resources[key] == null ? null : sum[key] + resources[key];
      }
    }
    return [arm, sum];
  });
  return h("div", {},
    h("h3", {}, "Calls and cost"),
    h("dl", { class: "kv", style: "margin-top:12px" },
      h("dt", {}, "Model calls"), h("dd", {}, count(total.calls)),
      h("dt", {}, "Tokens in"), h("dd", {}, count(total.input_tokens)),
      h("dt", {}, "Tokens out"), h("dd", {}, count(total.output_tokens)),
      h("dt", {}, "Estimated cost"), h("dd", {}, usd(total.estimated_usd)),
      h("dt", {}, "Time in model calls"), h("dd", {}, duration(total.latency_seconds))),
    h("div", { class: "scroll", style: "margin-top:18px" }, h("table", { class: "table" },
      h("thead", {}, h("tr", {}, h("th", {}, "Setup"), h("th", { class: "num" }, "Calls"),
        h("th", { class: "num" }, "Tokens in"), h("th", { class: "num" }, "Tokens out"), h("th", { class: "num" }, "Estimate"))),
      h("tbody", {}, perArm.map(([arm, sum]) => h("tr", {}, h("td", {}, armName(arm)), h("td", { class: "num" }, count(sum.calls)),
        h("td", { class: "num" }, count(sum.input_tokens)), h("td", { class: "num" }, count(sum.output_tokens)),
        h("td", { class: "num" }, usd(sum.estimated_usd))))))),
    h("p", { class: "muted", style: "font-size:13px;max-width:60ch" },
      "Each setup's figures include the calls it depends on, such as memory extraction, so they overlap and don't add up to the bill. Estimates use the prices given to demo.py and ignore cache discounts."));
}

function setupBlock(run) {
  const provider = run.config.provider || {};
  const profile = run.results.profile;
  const rows = [
    ["Provider", provider.provider], ["Model", provider.model],
    ["Profile", typeof profile === "string" ? profile : profile && profile.version],
    ["Thinking mode", provider.thinking_mode],
    ["Temperature", provider.provider && (provider.temperature == null ? "Provider default" : provider.temperature)],
    ["Call limit", provider.max_calls],
    ["Price per million tokens", provider.input_price != null && `$${provider.input_price} in, $${provider.output_price} out`],
    ["Glossary terms", run.glossary.length || null],
    ["Chapters left out by --limit", (run.results.excluded_files || []).length || null],
    ["Configuration hash", run.results.config_sha256 && run.results.config_sha256.slice(0, 12)],
  ].filter(([, value]) => value != null && value !== false && value !== "");
  return h("div", {}, h("h3", {}, "Setup"),
    h("dl", { class: "kv", style: "margin-top:12px" }, rows.map(([label, value]) => [h("dt", {}, label), h("dd", {}, value)])),
    run.results.description && h("p", { class: "muted", style: "font-size:13px;max-width:60ch" }, run.results.description));
}

/* Chapters */
async function chaptersView(run, params) {
  const { results } = run;
  const arms = results.arms;
  const chapter = results.chapters.find((item) => item.id === params.chapter) || results.chapters[0];
  if (!chapter) return h("p", { class: "empty" }, "This run has no chapters.");
  const texts = chapter.texts || {};
  const withText = arms.filter((arm) => texts[arm]);
  const right = arms.includes(params.right) ? params.right : withText[withText.length - 1] || arms[arms.length - 1];
  const fallbackLeft = withText.filter((arm) => arm !== right).pop() || arms[Math.max(0, arms.indexOf(right) - 1)];
  const left = arms.includes(params.left) ? params.left : fallbackLeft;
  const marks = params.marks !== "0";
  const terms = params.terms !== "0";

  const position = results.chapters.indexOf(chapter);
  const step = (offset, label) => {
    const target = results.chapters[position + offset];
    return target ? h("a", { class: "step", href: href({ chapter: target.id }) }, label) : h("span", { class: "step muted" }, label);
  };
  const pills = results.chapters.length <= 10
    ? h("nav", { class: "chapter-tabs", "aria-label": "Chapters" }, results.chapters.map((item) =>
      h("a", { href: href({ chapter: item.id }), "aria-current": item.id === chapter.id ? "page" : null },
        statusTag(item.status, { dot: true }), item.name)))
    : h("nav", { class: "chapter-tabs", "aria-label": "Chapters" }, step(-1, "Previous"),
      h("select", { "aria-label": "Chapter", onchange: (event) => go({ chapter: event.target.value }) },
        results.chapters.map((item) => h("option", { value: item.id, selected: item === chapter },
          `${item.name} (${(STATUS[item.status] || [item.status])[0].toLowerCase()})`))),
      h("span", { class: "step muted" }, `${position + 1} of ${results.chapters.length}`), step(1, "Next"));

  const select = (name, value) => h("select", {
    "aria-label": name === "left" ? "Left column" : "Right column",
    onchange: (event) => go({ [name]: event.target.value }),
  }, arms.map((arm) => h("option", { value: arm, selected: arm === value }, armName(arm) + (texts[arm] ? "" : " (no text)"))));
  const toggle = (name, label, on) => h("label", {}, h("input", {
    type: "checkbox", checked: on, onchange: (event) => go({ [name]: event.target.checked ? null : "0" }),
  }), label);
  const toolbar = h("div", { class: "toolbar" },
    h("label", {}, "Compare", select("left", left)), h("label", {}, "with", select("right", right)),
    toggle("marks", "Mark changes", marks), run.glossary.length ? toggle("terms", "Highlight glossary terms", terms) : null);

  let diff = null;
  if (marks && left !== right && texts[left] && texts[right]) {
    const key = [run.id, chapter.id, left, right].join("\u0000");
    if (!cache.diffs.has(key)) {
      const query = new URLSearchParams({ id: run.id, chapter: chapter.id, a: left, b: right });
      cache.diffs.set(key, await api(`/api/diff?${query}`));
    }
    diff = cache.diffs.get(key);
  }

  const notices = [];
  if (chapter.status === "failed") {
    notices.push(h("p", { class: "notice bad" }, `This chapter stopped with ${chapter.error_type || "an error"}. Setups after that point have no text, and their checks count as failed.`));
  }
  return h("div", {}, pills, notices, toolbar,
    h("div", { class: "chapter-body" },
      reader(run, chapter, left, right, diff, terms ? matchers(run.glossary) : null),
      sidePanel(run, chapter, left, right)));
}

function reader(run, chapter, left, right, diff, match) {
  const texts = chapter.texts || {};
  const src = paragraphs(chapter.source), a = paragraphs(texts[left]), b = paragraphs(texts[right]);
  const english = a.length > 0 && a.length === b.length;
  const all = english && src.length === a.length;
  const changed = diff ? diff.changed : null;
  const head = (column, title, note) => h("div", { class: `col-head c${column}` }, h("strong", {}, title), note && h("span", { class: "muted" }, note));
  const cells = [
    head(1, "Chinese source", plural(src.length, "paragraph")),
    head(2, armName(left), texts[left] ? null : "No text"),
    head(3, armName(right), changed == null ? (texts[right] ? null : "No text")
      : changed ? `${plural(changed, diff.aligned ? "paragraph" : "passage")} edited` : "No edits"),
  ];
  const labels = [null, "Chinese source", armName(left), armName(right)];
  const place = (el, column, row, span) => {
    el.classList.add(`c${column}`);
    el.dataset.label = labels[column];
    el.style.gridColumn = column;
    el.style.gridRow = span ? `${row} / span ${span}` : row;
    return el;
  };
  const zhCell = (text, extra) => h("div", { class: `cell${extra || ""}`, lang: "zh" }, highlight(text, match && match.zh));
  const enCell = (text, extra) => h("div", { class: `cell${extra || ""}`, lang: "en" }, highlight(text, match && match.en));
  const none = (column) => place(h("div", { class: "cell none" }, "No text from this setup for this chapter."), column, 2);
  const diffCell = (ops, extra) => h("div", { class: `cell${extra || ""}`, lang: "en" }, ops.map(([tag, text]) =>
    tag === "equal" ? highlight(text, match && match.en)
      : tag === "delete" ? h("del", {}, text) : h("ins", {}, highlight(text, match && match.en))));

  if (english) {
    src.length && !all && cells.push(place(zhCell(src.join("\n\n"), " whole"), 1, 2, a.length));
    a.forEach((text, i) => {
      const row = i + 2;
      if (all) cells.push(place(zhCell(src[i]), 1, row));
      cells.push(place(enCell(text), 2, row));
      cells.push(place(diff && diff.aligned ? diffCell(diff.paragraphs[i]) : enCell(b[i]), 3, row));
      for (const cell of cells.slice(all ? -3 : -2)) cell.dataset.row = row;
    });
  } else {
    cells.push(src.length ? place(zhCell(src.join("\n\n"), " whole"), 1, 2) : none(1));
    cells.push(a.length ? place(enCell(a.join("\n\n"), " whole"), 2, 2) : none(2));
    cells.push(!b.length ? none(3) : place(diff ? diffCell(diff.paragraphs.flat(), " whole") : enCell(b.join("\n\n"), " whole"), 3, 2));
  }
  const grid = h("div", { class: "reader", onmouseover: linkHover, onmouseout: clearHover }, cells);
  return h("div", { class: "reader-wrap" }, grid);
}

function linkHover(event) {
  const grid = event.currentTarget;
  clearHover({ currentTarget: grid });
  const cell = event.target.closest(".cell[data-row]");
  if (cell) grid.querySelectorAll(`.cell[data-row="${cell.dataset.row}"]`).forEach((el) => el.classList.add("row-hover"));
  const term = event.target.closest("mark.term");
  if (term) grid.querySelectorAll(`mark.term[data-key="${CSS.escape(term.dataset.key)}"]`).forEach((el) => el.classList.add("lit"));
}
function clearHover(event) {
  event.currentTarget.querySelectorAll(".row-hover, .lit").forEach((el) => el.classList.remove("row-hover", "lit"));
}

/* Glossary highlighting: Chinese terms match as substrings; English terms need letter boundaries,
   ignore case, and prefer the longest term, like the harness's own term checks. */
const escapeRegExp = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
function matchers(glossary) {
  if (!glossary.length) return null;
  const zh = new Map(), en = new Map();
  for (const entry of glossary) {
    if (entry.source) zh.set(entry.source, { key: entry.source });
    if (entry.target) en.set(entry.target.toLowerCase(), { key: entry.source });
    for (const variant of entry.blocked_variants || []) en.set(variant.toLowerCase(), { key: entry.source, blocked: true });
  }
  const build = (map, bounded) => {
    const terms = [...map.keys()].sort((x, y) => y.length - x.length).map(escapeRegExp);
    if (!terms.length) return null;
    const body = `(${terms.join("|")})`;
    return { re: new RegExp(bounded ? `(?<![\\p{L}\\p{N}_])${body}(?![\\p{L}\\p{N}_])` : body, bounded ? "giu" : "gu"), map, fold: bounded };
  };
  return { zh: build(zh, false), en: build(en, true) };
}

function highlight(text, matcher) {
  if (!matcher || !text) return text;
  const out = [];
  let last = 0;
  for (const found of text.matchAll(matcher.re)) {
    const info = matcher.map.get(matcher.fold ? found[0].toLowerCase() : found[0]);
    if (!info) continue;
    out.push(text.slice(last, found.index));
    out.push(h("mark", { class: `term${info.blocked ? " blocked" : ""}`, "data-key": info.key,
      title: info.blocked ? `Blocked variant for ${info.key}` : `Glossary term ${info.key}` }, found[0]));
    last = found.index + found[0].length;
  }
  out.push(text.slice(last));
  return out;
}

const EVENTS = {
  term_alignment_requested: (e) => [`Asked the model to align ${plural(e.terms || 0, "glossary term")}`, ""],
  term_alignment_applied: () => ["Kept the term alignment edit", "kept"],
  segment_reviewed: (e) => (e.issues || []).length
    ? [`Review found ${plural(e.issues.length, "issue")} and proposed ${plural(e.patches || 0, "edit")}`, "flag"]
    : ["Review found no issues", ""],
  patch_accepted: (e) => [`Kept an edit for ${(e.issue_ids || []).join(", ") || "a review issue"}`, "kept"],
  patch_rejected: (e) => [`Rejected an edit${e.reason ? `: ${e.reason.replace(/_/g, " ")}` : ""}`, "flag"],
  review_unavailable: (e) => [`Review unavailable${e.reason ? ` (${e.reason.replace(/_/g, " ")})` : ""}`, "flag"],
  qa_findings_informational: (e) => [`${plural(e.count || 0, "informational check finding")}: ${(e.check_ids || []).join(", ").replace(/_/g, " ")}`, ""],
  generation_or_coverage_failed: () => ["Generation or review coverage failed", "flag"],
};

function sidePanel(run, chapter, left, right) {
  const arms = run.results.arms;
  const records = chapter.arms || {};
  const owner = (records[right] || {}).events ? right : arms.find((arm) => (records[arm] || {}).events);
  const events = owner ? records[owner].events : [];
  const warnings = [...(chapter.warnings || []), ...arms.flatMap((arm) => (records[arm] || {}).warnings || [])];
  const sections = [];

  const checked = arms.filter((arm) => ((records[arm] || {}).checks || []).length);
  if (checked.length) {
    sections.push(h("section", {}, h("h4", {}, "Checks in this chapter"), h("table", { class: "table" },
      h("tbody", {}, checked.map((arm) => {
        const checks = records[arm].checks;
        return h("tr", {}, h("td", {}, armName(arm)), h("td", { class: "num" }, `${checks.filter((c) => c.passed).length}/${checks.length}`));
      })))));
  }
  sections.push(h("section", {}, h("h4", {}, owner ? `What the harness did (${armName(owner)})` : "What the harness did"),
    events.length
      ? h("ol", { class: "log" }, events.map((event) => {
        const [label, tone] = (EVENTS[event.kind] || ((e) => [e.kind.replace(/_/g, " "), ""]))(event);
        return h("li", { class: tone }, label, event.segment_id && h("div", { class: "seg" }, `Segment ${event.segment_id}`));
      }))
      : h("p", { class: "muted", style: "font-size:13.5px;margin:0" }, "No repair steps recorded for this chapter.")));
  if (warnings.length) {
    sections.push(h("section", {}, h("h4", {}, "Warnings"), warnings.map((warning) =>
      h("p", { class: "warning" }, warning.message || warning.code, " ", h("code", {}, [warning.code, warning.segment_id].filter(Boolean).join(" "))))));
  }
  const findings = arms.filter((arm) => (records[arm] || {}).deterministic_findings);
  if (findings.length) {
    sections.push(h("section", {}, h("h4", {}, "Automatic text checks"), h("table", { class: "table" }, h("tbody", {},
      findings.map((arm) => {
        const result = records[arm].deterministic_findings;
        const detail = Object.entries(result.by_check || {}).map(([check, n]) => `${check.replace(/_/g, " ")} ${n}`).join(", ");
        return h("tr", {}, h("td", {}, armName(arm)), h("td", { style: "white-space:normal" }, result.total_findings ? detail : "None"));
      })))));
  }
  if (chapter.context && Object.keys(chapter.context).length) {
    sections.push(h("section", {}, h("details", { class: "context" },
      h("summary", {}, "Context sent with this chapter"),
      Object.entries(chapter.context).map(([segment, text]) => [h("div", { class: "seg muted", style: "font-size:12.5px;margin-top:10px" }, `Segment ${segment}`), h("pre", { lang: "zh" }, text)]))));
  }
  return h("aside", { class: "side", "aria-label": "Chapter evidence" }, sections);
}

/* Story memory */
function memoryView(run, params) {
  const entries = run.memory.filter((record) => record.type === "memory_entry" && record.entry);
  const decisions = run.memory.filter((record) => record.type === "terminology_decision" && record.decision);
  const rejections = run.results.memory_rejections || [];
  if (!entries.length && !decisions.length) {
    return h("div", { class: "empty" }, h("strong", {}, "No story memory in this run"),
      "Memory is built by the memory + glossary and full harness setups. Runs that stopped before it, and the repair bench, don't keep any.");
  }
  const claim = (entry) => entry.status === "provisional" || entry.scope === "dialogue_claim";
  const filter = ["claims", "narrated"].includes(params.facts) ? params.facts : "all";
  const shown = entries.filter(({ entry }) => filter === "all" || (filter === "claims") === claim(entry));
  const names = Object.fromEntries(run.results.chapters.map((chapter) => [chapter.id, chapter.name]));
  const claims = entries.filter(({ entry }) => claim(entry)).length;

  const parts = [
    h("p", { class: "headline" }, `${plural(entries.length, "fact")} kept, each tied to an exact quote from the Chinese source.`),
    h("p", { class: "metric-note" }, `${plural(entries.length - claims, "fact")} come from the narration. ${plural(claims, "claim")} by characters ${claims === 1 ? "is" : "are"} kept as provisional, so the story can still contradict ${claims === 1 ? "it" : "them"}. A matching quote shows where a fact came from; it doesn't prove the fact was read correctly.`),
    h("div", { class: "filters", role: "group", "aria-label": "Show facts" },
      [["all", "All facts"], ["narrated", "From the narration"], ["claims", "Characters' claims"]].map(([key, label]) =>
        h("button", { type: "button", "aria-pressed": String(filter === key), onclick: () => go({ facts: key === "all" ? null : key }) }, label))),
  ];
  const byChapter = new Map();
  for (const record of shown) {
    const id = record.chapter_id || record.entry.known_from_chapter;
    if (!byChapter.has(id)) byChapter.set(id, []);
    byChapter.get(id).push(record.entry);
  }
  for (const [id, items] of byChapter) {
    parts.push(h("section", { class: "memory-chapter" },
      h("h3", {}, names[id] ? `${names[id]} (chapter ${id})` : `Chapter ${id}`),
      items.map((entry) => h("article", { class: "fact" },
        h("div", {}, h("div", { class: "subject", lang: "zh" }, entry.subject),
          h("div", { class: "about" }, h("span", {}, entry.kind),
            h("span", { class: claim(entry) ? "provisional" : "" }, claim(entry) ? "Provisional claim" : "Established"))),
        h("div", {}, h("p", { class: "value" }, entry.value),
          (entry.source_evidence || []).map((evidence) => h("blockquote", { class: "cite" },
            h("q", { lang: "zh" }, evidence.exact_excerpt),
            h("span", {}, `Chapter ${evidence.chapter_id}, segment ${evidence.segment_id}`))))))));
  }

  if (decisions.length) {
    const glossary = Object.fromEntries(run.glossary.map((entry) => [entry.source, entry.target]));
    parts.push(h("section", { class: "section" },
      h("h3", {}, "Terms the memory proposed"),
      h("p", {}, "Where the translator's glossary names the same term, the glossary wins and the memory's proposal is struck through."),
      h("div", { class: "scroll" }, h("table", { class: "table term-table" },
        h("thead", {}, h("tr", {}, h("th", {}, "Chapter"), h("th", {}, "Source"), h("th", {}, "Memory proposed"),
          h("th", {}, "Status"), h("th", {}, "Translator glossary"))),
        h("tbody", {}, decisions.map(({ chapter_id, decision }) => {
          const chosen = glossary[decision.source];
          const overridden = chosen != null && chosen !== decision.target;
          return h("tr", {}, h("td", {}, chapter_id), h("td", {}, h("span", { class: "zh", lang: "zh" }, decision.source)),
            h("td", { class: `en ${overridden ? "overridden" : ""}` }, decision.target),
            h("td", {}, [decision.status, decision.outcome].filter(Boolean).join(", ")),
            h("td", { class: "en" }, chosen || "Not in glossary"));
        }))))));
  }
  if (rejections.length) {
    parts.push(h("section", { class: "section" }, h("h3", {}, "Proposals the memory refused"),
      h("p", {}, "Facts and terms the model proposed that failed validation, with the error that stopped each one."),
      h("table", { class: "table" }, h("tbody", {}, rejections.map((row) =>
        h("tr", {}, h("td", {}, `Chapter ${row.chapter_id}`), h("td", {}, row.item), h("td", {}, row.reason)))))));
  }
  return h("div", { class: "measure" }, parts);
}

/* New run: drop chapters, choose a model, start. */
const draft = { chapters: [], glossary: null, sample: false, skipped: 0, rejected: [], error: null, busy: false };

function glossaryTerms(text, name) {
  if (name.toLowerCase().endsWith(".json")) {
    try { const rows = JSON.parse(text); return Array.isArray(rows) ? rows.length : null; } catch { return null; }
  }
  return glossaryLines(text).filter(isGlossaryEntry).length;
}

// The text glossary format parse_glossary_text accepts: "#" lines are comments or "# block:" variants;
// entries are "source -> target" or "source: target, alternative" (full-width colon too). A colon
// entry must name an English target, so a chapter's "林青：走吧。" dialogue line isn't one.
const glossaryLines = (text) => (text || "").split(LINES).map((line) => line.trim()).filter((line) => line && !line.startsWith("#"));
const isGlossaryEntry = (line) => line.includes("->") || /^[^:：]{1,40}[:：][^:：]*[A-Za-z]/.test(line);

// A .txt file is the glossary when at least 80% of its entry lines are glossary entries;
// otherwise it is a chapter. File name alone no longer decides this.
function isGlossaryText(text) {
  const lines = glossaryLines(text);
  return lines.length > 0 && lines.filter(isGlossaryEntry).length / lines.length >= 0.8;
}

// The command line refuses chapters and glossaries that are not valid UTF-8; File.text() would
// silently replace bad bytes with U+FFFD instead, so decode strictly here and strip a leading BOM
// to match the server's utf-8-sig reading.
async function readUtf8(file) {
  const bytes = await file.arrayBuffer();
  const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  return text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;
}

async function addFiles(files) {
  let skipped = 0;
  const rejected = [];
  for (const file of files) {
    const name = file.name, lower = name.toLowerCase();
    if (!lower.endsWith(".json") && !lower.endsWith(".txt")) {
      skipped += 1;
      continue;
    }
    let text;
    try {
      text = await readUtf8(file);
    } catch {
      rejected.push(name);
      continue;
    }
    if (lower.endsWith(".json") || isGlossaryText(text)) {
      draft.glossary = { name, text, terms: glossaryTerms(text, name) };
    } else {
      draft.chapters = draft.chapters.filter((chapter) => chapter.name !== name).concat({ name, text });
    }
  }
  draft.chapters.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  Object.assign(draft, { sample: false, skipped, rejected, error: null });
  if (draft.chapters.length && Number(settings.limit) > draft.chapters.length) settings.limit = draft.chapters.length;
  composer();
}

// Read entries synchronously inside the drop event; the DataTransfer is gone after the first await.
async function droppedFiles(event) {
  const entries = [...event.dataTransfer.items].map((item) => item.webkitGetAsEntry && item.webkitGetAsEntry()).filter(Boolean);
  if (!entries.length) return [...event.dataTransfer.files];
  const fileOf = (entry) => new Promise((resolve, reject) => entry.file(resolve, reject));
  const files = [];
  for (const entry of entries) {
    if (entry.isFile) files.push(await fileOf(entry));
    if (!entry.isDirectory) continue;
    const reader = entry.createReader();
    for (let batch; (batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject))).length;) {
      for (const child of batch) if (child.isFile) files.push(await fileOf(child));
    }
  }
  return files;
}

function field(label, control, hint) {
  return h("label", { class: "field" }, h("span", { class: "label" }, label), control, hint && h("span", { class: "hint" }, hint));
}

function bound(name, attrs = {}) {
  return h("input", { ...attrs, name, value: settings[name] ?? "", oninput: (event) => { settings[name] = event.target.value; remember(); } });
}

async function composer() {
  if (!cache.setup) cache.setup = await api("/api/setup");
  const setup = cache.setup;
  const provider = setup.providers[settings.provider] ? settings.provider : "deepseek";
  // The endpoint is used only with "custom", whose key variable the server reports as TL_API_KEY.
  const { key_env: keyEnv, key_set: keySet } = setup.providers[provider];
  const count = draft.sample ? setup.sample.chapters : draft.chapters.length;
  const pick = (attrs) => h("input", { type: "file", class: "sr", ...attrs, onchange: (event) => addFiles([...event.target.files]) });

  const zone = h("div", {
    class: "dropzone",
    ondragover: (event) => { event.preventDefault(); event.currentTarget.classList.add("over"); },
    ondragleave: (event) => event.currentTarget.classList.remove("over"),
    ondrop: async (event) => {
      event.preventDefault();
      event.currentTarget.classList.remove("over");
      addFiles(await droppedFiles(event));
    },
  },
  h("strong", {}, "Drop chapter files or a folder here"),
  h("p", {}, "One UTF-8 .txt file per chapter, named in reading order. Bring your glossary too: a .json file, or a .txt file of “source -> target” lines."),
  h("div", { class: "pickers" },
    h("label", { class: "button" }, "Choose files", pick({ multiple: true, accept: ".txt,.json" })),
    h("label", { class: "button" }, "Choose a folder", pick({ webkitdirectory: true }))));

  const inputs = [];
  if (draft.sample) {
    inputs.push(h("div", { class: "picked" }, h("div", {},
      h("strong", {}, "Sample story"), h("span", { class: "muted" }, `${setup.sample.chapters} short chapters written for this demo, with a ${setup.sample.terms}-term glossary`)),
      h("button", { type: "button", class: "link", onclick: () => { draft.sample = false; composer(); } }, "Remove")));
  } else if (draft.chapters.length) {
    inputs.push(h("div", { class: "picked" }, h("div", {}, h("strong", {}, plural(draft.chapters.length, "chapter")),
      h("ol", { class: "file-list" }, draft.chapters.map((chapter) => h("li", {}, h("span", {}, chapter.name),
        h("span", { class: "muted" }, plural([...chapter.text].length, "character")),
        h("button", { type: "button", class: "link", "aria-label": `Remove ${chapter.name}`,
          onclick: () => { draft.chapters = draft.chapters.filter((c) => c !== chapter); composer(); } }, "Remove")))))),
    h("div", { class: "picked" }, draft.glossary
      ? [h("div", {}, h("strong", {}, "Glossary"), h("span", { class: "muted" }, `${draft.glossary.name}${draft.glossary.terms != null ? `, ${plural(draft.glossary.terms, "term")}` : ""}`)),
        h("button", { type: "button", class: "link", onclick: () => { draft.glossary = null; composer(); } }, "Remove")]
      : h("div", {}, h("strong", {}, "No glossary"), h("span", { class: "muted" }, "The run still works, but there are no translator terms to hold, so the term ledger stays empty."))));
  }
  if (draft.skipped) inputs.push(h("p", { class: "muted" }, `Skipped ${plural(draft.skipped, "file")} that ${draft.skipped === 1 ? "isn't" : "aren't"} .txt or .json.`));
  if (draft.rejected && draft.rejected.length) {
    const one = draft.rejected.length === 1;
    inputs.push(h("p", { class: "muted" },
      `Not added: ${draft.rejected.join(", ")} ${one ? "isn't" : "aren't"} UTF-8 text. Save ${one ? "it" : "them"} as UTF-8 and add ${one ? "it" : "them"} again.`));
  }

  const replays = setup.replays || [];
  const page = h("div", { class: "measure composer" },
    h("header", { class: "composer-head" }, h("h2", {}, "New run"),
      h("p", { class: "lede" }, "Translate a few chapters with the harness and watch each step as it happens: story memory, every translation setup, term alignment and review. Then compare what changed.")),
    zone,
    !draft.chapters.length && !draft.sample && h("p", { class: "or" }, "No chapters at hand? ",
      h("button", { type: "button", class: "link", onclick: () => { draft.sample = true; draft.error = null; composer(); } }, "Use the sample story"),
      `: ${setup.sample.chapters} short chapters and a ${setup.sample.terms}-term glossary written for this demo.`),
    inputs,
    h("fieldset", {}, h("legend", {}, "Model"),
      h("div", { class: "row" },
        field("Provider", h("select", { onchange: (event) => { settings.provider = event.target.value; remember(); composer(); } },
          Object.entries(PROVIDERS).map(([key, [label]]) => h("option", { value: key, selected: key === provider }, label)))),
        field("Model ID", bound("model", { required: true, placeholder: PROVIDERS[provider][1], autocomplete: "off", spellcheck: "false" })),
        provider === "custom" && field("Endpoint", bound("base_url", { placeholder: "https://…/v1", autocomplete: "off" }))),
      field("API key", h("input", { type: "password", name: "api_key", autocomplete: "off", value: apiKey,
        oninput: (event) => { apiKey = event.target.value; }, placeholder: keySet ? `Using ${keyEnv} from your environment` : "Paste a key" }),
        keySet ? `Leave it empty to use ${keyEnv}.` : "Goes only to this local workbench and on to the provider. It is never written to disk.")),
    h("fieldset", {}, h("legend", {}, "Budget"),
      h("div", { class: "row" },
        field("Input price", bound("input_price", { inputmode: "decimal", placeholder: "USD per million" })),
        field("Output price", bound("output_price", { inputmode: "decimal", placeholder: "USD per million" })),
        field("Spending cap", bound("max_usd", { inputmode: "decimal", placeholder: "USD, optional" }))),
      h("p", { class: "hint" }, "Optional. With both prices the run estimates its cost and can stop at the cap. It never makes more than 120 model calls.")),
    h("fieldset", {}, h("legend", {}, "Run"),
      h("label", { class: "check" }, h("input", { type: "checkbox", checked: settings.compare !== false,
        onchange: (event) => { settings.compare = event.target.checked; remember(); } }),
      h("span", {}, h("strong", {}, "Compare four setups"),
        h("span", { class: "hint" }, "Chapter only, glossary, memory + glossary, then the full harness. Unchecked runs the last two, with fewer calls."))),
      field("Chapters to translate", bound("limit", { type: "number", min: 0, max: count || null, step: 1, class: "short" }),
        `First chapters in reading order${count ? ` (of ${count})` : ""}. 0 translates them all.`)),
    draft.error && h("p", { class: "notice bad", role: "alert" }, draft.error),
    h("div", { class: "start" },
      h("button", { type: "button", class: "primary", disabled: draft.busy || !count, onclick: startRun },
        draft.busy ? "Starting…" : "Start translation"),
      !count && h("span", { class: "muted" }, "Add chapters or use the sample story first.")),
    replays.length > 0 && h("section", { class: "section replay" },
      h("h3", {}, "Replay a recorded run"),
      h("p", {}, "Runs the harness again from a saved run's recorded model calls. No key and no API calls, so its numbers repeat the original."),
      h("div", { class: "row" },
        h("select", { "aria-label": "Recorded run", id: "replay-choice" }, replays.map((run) =>
          h("option", { value: run.id }, [run.name, plural(run.chapters, "chapter"), run.model,
            (STATUS[run.status] || [run.status])[0].toLowerCase()].filter(Boolean).join(", ")))),
        h("button", { type: "button", disabled: draft.busy, onclick: () => startRun({ source: "replay", replay: document.getElementById("replay-choice").value }) }, "Replay"))));
  view.replaceChildren(page);
}

async function startRun(override) {
  const request = override && override.source ? override : {
    ...settings, source: draft.sample ? "sample" : "upload",
    // The endpoint stays remembered for "custom" but is never sent with a named provider.
    base_url: settings.provider === "custom" ? settings.base_url : undefined,
    api_key: apiKey,
    chapters: draft.sample ? undefined : draft.chapters, glossary: draft.sample ? undefined : draft.glossary,
  };
  draft.busy = true;
  draft.error = null;
  composer();
  try {
    const response = await fetch("/api/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request) });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error || `The workbench answered ${response.status}`);
    draft.busy = false;
    await refreshIndex();
    location.hash = href({ job: body.id }, {});
  } catch (error) {
    draft.busy = false;
    draft.error = error.message;
    composer();
  }
}

/* Activity: each model call as the harness makes it, grouped by chapter. */
function stopPolling() {
  if (poller) clearTimeout(poller.timer);
  poller = null;
}

async function activityPage(id) {
  if (poller && poller.id === id) return drawActivity(id, poller);
  stopPolling();
  const state = poller = { id, after: 0, events: [], status: "running", seen: new Map(), clock: null, dom: null, retryDelay: null, failSince: null };
  view.replaceChildren(h("p", { class: "muted" }, "Connecting to the run…"));
  const tick = async () => {
    if (poller !== state) return;
    try {
      const update = await api(`/api/activity?id=${encodeURIComponent(id)}&after=${state.after}`);
      state.events.push(...update.events);
      state.after += update.events.length;
      Object.assign(state, { status: update.status, stopping: update.stopping || state.stopping, error: null, retryDelay: null, failSince: null });
      if (!state.clock) state.clock = Date.now() - 1000 * (state.events.at(-1)?.seconds || 0);
    } catch (error) {
      // Keep the last known status (and the Stop button) while retrying with backoff; give up at once
      // on a definite answer (no such run), or after about two minutes without contact.
      if (!state.failSince) state.failSince = Date.now();
      if ((error.status >= 400 && error.status < 500) || Date.now() - state.failSince >= 120000) {
        Object.assign(state, { status: "unknown", error: `${error.message}. Reload, or open the run from the list.` });
      } else {
        state.error = "Lost contact with the workbench; retrying…";
        state.retryDelay = Math.min((state.retryDelay || 1000) * 2, 15000);
      }
    }
    if (poller !== state) return;
    const nearBottom = innerHeight + scrollY >= document.documentElement.scrollHeight - 120;
    drawActivity(id, state);
    if (nearBottom && state.status === "running") scrollTo({ top: document.documentElement.scrollHeight });
    if (state.status === "running") state.timer = setTimeout(tick, state.retryDelay || 900);
    else await refreshIndex();
  };
  await tick();
}

async function recordedActivity(id) {
  const update = await api(`/api/activity?id=${encodeURIComponent(id)}`);
  return feed({ id, events: update.events, status: "recorded", seen: new Map() }, false);
}

// Failed calls report no usage; the totals cover the calls that did, and say nothing when none did.
function tally(events) {
  const done = events.filter((event) => event.kind === "call_finished");
  const sum = (keys) => {
    const known = done.filter((event) => keys.every((key) => event[key] != null));
    return known.length ? known.reduce((total, event) => total + keys.reduce((n, key) => n + event[key], 0), 0) : null;
  };
  return { calls: done.reduce((n, event) => n + (event.attempts || 0), 0), tokens: sum(["input_tokens", "output_tokens"]), usd: sum(["usd"]) };
}

// The header stays in place between polls so a click on Stop is never swallowed by a redraw.
function drawActivity(id, state) {
  if (!state.dom || !state.dom.head.isConnected) {
    state.dom = { head: h("div", { class: "run-head" }), facts: h("dl", { class: "facts" }), body: h("div", {}) };
    view.replaceChildren(h("header", {}, state.dom.head, state.dom.facts), h("div", { class: "rule" }), state.dom.body);
    state.shown = null;
  }
  const started = state.events.find((event) => event.kind === "run_started") || {};
  const finished = state.events.find((event) => event.kind === "run_finished");
  const failed = state.events.find((event) => event.kind === "run_failed");
  const current = state.events.filter((event) => event.kind === "chapter_started").at(-1);
  const running = state.status === "running";
  const phase = running ? (state.stopping ? "stopping" : "running") : finished ? "finished" : "other";
  if (state.shown !== phase) {
    state.shown = phase;
    state.dom.head.replaceChildren(
      h("div", {}, h("div", { class: "crumb" }, parentPath(id)), h("h2", {}, baseName(id))),
      running ? h("button", { type: "button", class: "stop", disabled: Boolean(state.stopping), onclick: async (event) => {
        event.currentTarget.disabled = true;
        state.stopping = true;
        await fetch("/api/stop", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
        drawActivity(id, state);
      } }, state.stopping ? "Stopping…" : "Stop run")
        : finished && h("nav", { class: "files" }, h("a", { href: href({ run: id, tab: "overview" }, {}) }, "See results"),
          h("a", { href: href({ run: id, tab: "chapters" }, {}) }, "Compare chapters")));
  }
  const totals = tally(state.events);
  const elapsed = running && state.clock ? (Date.now() - state.clock) / 1000 : (state.events.at(-1)?.seconds || 0);
  const chaptersStarted = state.events.some((event) => event.kind === "chapter_started");
  const label = running ? (state.stopping ? ["Stopping after this call", "warn"] : ["Running", "live"])
    : failed ? (chaptersStarted ? ["Failed", "bad"] : ["Didn't start", "bad"]) : finished && finished.stopped ? ["Stopped", "warn"]
      : finished ? STATUS[finished.status] || ["Finished", "good"] : ["Unknown", ""];
  state.dom.facts.replaceChildren(...[
    fact("Status", h("span", { class: `status ${label[1]}` }, label[0])),
    fact("Evidence", modeLabel(started.mode)),
    started.model && fact("Model", `${started.model} (${started.provider})`),
    running && current && fact("Chapter", current.total ? `${current.index} of ${current.total}` : current.index),
    fact("Model calls", totals.calls),
    fact("Tokens", count(totals.tokens)),
    fact("Estimated cost", usd(totals.usd)),
    fact("Elapsed", duration(elapsed))].filter(Boolean));
  state.dom.body.replaceChildren(...[state.error && h("p", { class: "notice bad" }, state.error), feed(state, running)].filter(Boolean));
  clearTimeout(state.redraw);
  if (running) state.redraw = setTimeout(() => poller === state && drawActivity(id, state), 1000);
}

function feed(state, running) {
  const groups = [];
  const byChapter = new Map();
  const calls = new Map();
  for (const event of state.events) {
    if (event.kind === "chapter_started") {
      const group = { ...event, calls: [], summary: null };
      groups.push(group);
      byChapter.set(event.chapter, group);
    } else if (event.kind === "call_started") {
      const call = { ...event, done: null };
      calls.set(event.call, call);
      (byChapter.get(event.chapter) || groups.at(-1) || { calls: [] }).calls.push(call);
      if (!state.seen.has(event.call)) state.seen.set(event.call, Date.now());
    } else if (event.kind === "call_finished" && calls.has(event.call)) {
      calls.get(event.call).done = event;
    } else if (event.kind === "chapter_finished") {
      const group = byChapter.get(event.chapter) || (groups.push({ ...event, calls: [] }), groups.at(-1));
      group.summary = event;
    }
  }
  const finished = state.events.find((event) => event.kind === "run_finished");
  const failed = state.events.find((event) => event.kind === "run_failed");
  const items = groups.map((group) => h("section", { class: "chapter-feed" },
    h("h3", {}, group.name || `Chapter ${group.chapter}`, group.summary && statusTag(group.summary.status)),
    group.calls.length > 0 && h("ol", { class: "feed" }, group.calls.map((call) => {
      const [label, setup] = OPERATIONS[call.operation] || [call.operation.replace(/_/g, " "), null];
      const done = call.done;
      const state_ = !done ? (running ? "active" : "lost") : done.ok ? "ok" : "failed";
      const live = !done && running ? (Date.now() - state.seen.get(call.call)) / 1000 : null;
      return h("li", { class: state_ },
        h("span", { class: "step-label" }, label, setup && h("span", { class: "tag" }, setup)),
        h("span", { class: "step-meta" },
          done ? [done.ok ? null : h("span", { class: "error" }, done.error || "Failed"),
            done.cached ? "recorded" : null,
            done.input_tokens != null ? `${count(done.input_tokens)} in, ${count(done.output_tokens)} out` : null,
            duration(done.duration)].filter(Boolean).map((text) => h("span", {}, text))
            : h("span", {}, running ? `Working, ${duration(live)}` : "No result recorded")));
    })),
    group.summary && chapterLine(group.summary)));
  if (!groups.length && running) items.push(h("p", { class: "muted" }, "Preparing the run…"));
  if (finished) items.push(finishLine(state.id, finished));
  if (!running && !finished && !failed && state.events.length)
    items.push(h("div", { class: "notice" }, h("strong", {}, "This run stopped before it finished. "),
      "Its process ended partway, so calls without a result were never completed. Chapters that finished are on disk."));
  if (failed) {
    const started = state.events.some((event) => event.kind === "chapter_started");
    items.push(h("div", { class: "notice bad" }, h("strong", {}, started ? "The run failed. " : "The run didn't start. "),
      failed.message || failed.error,
      " ", h("a", { href: "#" }, "Back to the new-run form")));
  }
  return h("div", { class: "measure activity" }, items);
}

function chapterLine(summary) {
  if (summary.status === "not_started") return h("p", { class: "chapter-sum muted" }, "Not translated: the run stopped before this chapter.");
  const parts = [];
  if (summary.error) parts.push(`Stopped with ${summary.error}.`);
  if (summary.kept != null) parts.push(`${plural(summary.kept, "edit")} kept, ${summary.rejected} rejected.`);
  if (summary.facts) parts.push(`${plural(summary.facts, "fact")} added to story memory.`);
  const checks = summary.checks || {};
  const arms = Object.keys(checks).filter((arm) => checks[arm][1]);
  if (arms.length) {
    parts.push(`Glossary terms used as specified: ${arms.map((arm) => `${armName(arm).toLowerCase()} ${checks[arm][0]}/${checks[arm][1]}`).join(", ")}.`);
  }
  if (summary.warnings) parts.push(`${plural(summary.warnings, "warning")}.`);
  return h("p", { class: "chapter-sum" }, parts.join(" "));
}

function finishLine(id, finished) {
  const arms = finished.arms || [];
  const first = (finished.scores || {})[arms[0]], last = (finished.scores || {})[arms.at(-1)];
  const resources = finished.resources || {};
  const lines = [finished.stopped ? "Stopped. Chapters translated before the stop are saved."
    : `Finished: ${(STATUS[finished.status] || [finished.status])[0].toLowerCase()}.`];
  const complete = ["delivered", "delivered_with_warnings", "completed"].includes(finished.status) && !finished.stopped;
  if (!complete) {
    lines.push("Some chapters have no text from every setup, so the runs aren't comparable.");
  } else if (first && last && first[1] && arms.length > 1) {
    lines.push(`${first[0]} of ${first[1]} glossary checks passed ${armPhrase(arms[0])}; ${last[0]} of ${last[1]} ${armPhrase(arms.at(-1))}.`);
  }
  lines.push(`${plural(resources.calls || 0, "model call")}${resources.estimated_usd != null ? `, about ${usd(resources.estimated_usd)}` : ""}.`);
  return h("div", { class: "finish" }, h("p", { class: "headline" }, lines.join(" ")),
    h("p", {}, h("a", { class: "button primary", href: href({ run: id, tab: "overview" }, {}) }, "See results"), " ",
      h("a", { class: "button", href: href({ run: id, tab: "chapters" }, {}) }, "Compare chapters")));
}

document.getElementById("rail-toggle").addEventListener("click", (event) => {
  const rail = document.querySelector(".rail");
  const open = rail.classList.toggle("open");
  event.currentTarget.setAttribute("aria-expanded", String(open));
});
window.addEventListener("hashchange", render);
render();
