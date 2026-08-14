const PRONOUNS = new Set([
  "he", "she", "they", "him", "her", "them", "his", "hers", "their", "theirs",
  "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours",
]);
const SENTIMENTS = ["positive", "negative", "mixed", "neutral", "not_about"];
const NAME_RE = /\b[A-Z][a-z]+(?:[ '-][A-Z][a-z]+){0,3}\b/g;
const NAME_STOP = new Set([
  "The", "A", "An", "And", "But", "Or", "For", "In", "On", "At", "To", "From",
  "With", "This", "That", "These", "Those", "There", "Then", "After", "Before",
  "Later", "According", "President", "Prime", "Minister", "Senator", "Washington",
]);

const state = {
  annotator: localStorage.getItem("qg-annotator") || "default",
  batch: [],
  article: null,
  quoteIndex: 0,
  mentionId: null,
  dirty: false,
  saving: false,
  selection: null,
  searchTimer: null,
  searchDraft: {},
};

const $ = (id) => document.getElementById(id);

function inferForm(surface) {
  const token = surface.trim().toLowerCase();
  if (PRONOUNS.has(token)) return "pronoun";
  if (/^(the|a|an)\s/i.test(token)) return "nominal";
  return "proper";
}

function emptySpeaker() {
  return {
    status: "unset",
    start: null,
    end: null,
    surface: "",
    qid: null,
    qid_label: "",
    qid_description: "",
    nil: false,
    form: "proper",
  };
}

function emptyQuotative() {
  return { status: "unset", start: null, end: null, surface: "" };
}

function currentQuote() {
  const quotes = state.article?.quotes || [];
  if (!quotes.length) return null;
  return quotes[Math.min(state.quoteIndex, quotes.length - 1)];
}

function quoteComplete(quote) {
  if (!quote) return false;
  if (quote.status === "pending") return false;
  if (quote.status === "reject") return true;
  const speaker = quote.speaker || {};
  if (speaker.status === "unset") return false;
  if (speaker.status === "identified") {
    if (speaker.start == null || speaker.end == null) return false;
    if (!speaker.nil && !speaker.qid) return false;
  }
  const quotative = quote.quotative || {};
  if (quotative.status !== "present" && quotative.status !== "implicit" && quotative.status !== "none") return false;
  if (quotative.status === "present" && (quotative.start == null || quotative.end == null)) return false;
  for (const mention of quote.mentions || []) {
    if (!SENTIMENTS.includes(mention.sentiment)) return false;
    if (!mention.nil && !mention.qid) return false;
  }
  return true;
}

function setSaveState(kind, label) {
  const el = $("save-state");
  el.dataset.state = kind;
  el.textContent = label;
}

function markDirty() {
  state.dirty = true;
  setSaveState("dirty", "unsaved");
  scheduleSave();
}

let saveTimer = null;
function scheduleSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => saveArticle(), 900);
}

async function api(path, options = {}) {
  const url = new URL(path, window.location.origin);
  url.searchParams.set("annotator", state.annotator);
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || response.statusText);
  }
  if (response.status === 204) return null;
  return response.json();
}

async function loadBatch() {
  state.batch = await api("/api/batch");
  renderBatch();
}

function renderBatch() {
  const list = $("batch-list");
  list.innerHTML = "";
  let done = 0;
  for (const row of state.batch) {
    if (row.complete || row.status === "skipped") done += 1;
    const li = document.createElement("li");
    if (state.article && row.article_id === state.article.article_id) li.classList.add("current");
    li.innerHTML = `<span class="dot ${row.status}"></span><span class="title"></span><span class="meta"></span>`;
    li.querySelector(".title").textContent = row.title || row.article_id;
    li.querySelector(".meta").textContent = `${row.phase ? `phase ${row.phase} · ` : ""}${row.n_done}/${row.n_quotes} quotes · ${row.status}`;
    li.addEventListener("click", () => openArticle(row.article_id));
    list.appendChild(li);
  }
  $("batch-progress").textContent = state.batch.length
    ? `${done}/${state.batch.length}`
    : "no batch.jsonl";
}

async function openArticle(articleId) {
  if (state.dirty && state.article) {
    try {
      await saveArticle();
    } catch (err) {
      setSaveState("error", err.message);
      return;
    }
  }
  state.article = await api(`/api/articles/${encodeURIComponent(articleId)}`);
  state.quoteIndex = 0;
  state.mentionId = null;
  state.searchDraft = {};
  state.dirty = false;
  setSaveState("clean", "saved");
  renderAll();
}

async function saveArticle() {
  if (!state.article || state.saving) return;
  state.saving = true;
  setSaveState("saving", "saving…");
  try {
    const result = await api(`/api/articles/${encodeURIComponent(state.article.article_id)}`, {
      method: "PUT",
      body: JSON.stringify(state.article),
    });
    state.dirty = false;
    setSaveState("clean", result.complete ? "saved · complete" : "saved");
    await loadBatch();
  } catch (err) {
    setSaveState("error", err.message);
    throw err;
  } finally {
    state.saving = false;
  }
}

function renderAll() {
  renderBatch();
  renderArticle();
  renderQuotePane();
}

function renderArticle() {
  const article = state.article;
  if (!article) {
    $("article-title").textContent = "Load a batch to begin";
    $("article-sub").textContent = "";
    $("article-text").textContent = "";
    $("quote-tabs").innerHTML = "";
    $("unmatched-banner").classList.add("hidden");
    return;
  }
  $("article-title").textContent = article.title || article.article_id;
  const bits = [
    article.phase ? `phase ${article.phase}` : "",
    article.date,
    article.source,
    article.text_source || "",
    article.url,
  ].filter(Boolean);
  $("article-sub").textContent = bits.join(" · ");

  const unmatched = article.unmatched_quotes || [];
  const banner = $("unmatched-banner");
  if (unmatched.length) {
    banner.classList.remove("hidden");
    banner.textContent = `${unmatched.length} unpaired quotation mark(s). Select the missing span and Add quote, or Set quote bounds on the candidate.`;
  } else {
    banner.classList.add("hidden");
  }

  const tabs = $("quote-tabs");
  tabs.innerHTML = "";
  (article.quotes || []).forEach((quote, i) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = `Q${i + 1}`;
    btn.title = quote.status;
    if (i === state.quoteIndex) btn.classList.add("current");
    if (quoteComplete(quote)) btn.textContent += " ✓";
    else if (quote.status === "reject") btn.textContent += " ✕";
    btn.addEventListener("click", () => {
      state.quoteIndex = i;
      state.mentionId = null;
      renderAll();
    });
    tabs.appendChild(btn);
  });

  $("article-text").innerHTML = "";
  $("article-text").appendChild(renderHighlightedText(article));
}

function renderHighlightedText(article) {
  const text = article.text || "";
  const quote = currentQuote();
  const layers = [];
  for (const q of article.quotes || []) {
    layers.push({
      start: q.outer_start,
      end: q.outer_end,
      className: "q" + (quote && q.id === quote.id ? " current" : ""),
    });
  }
  if (quote?.speaker?.start != null && quote.speaker.end != null) {
    layers.push({ start: quote.speaker.start, end: quote.speaker.end, className: "spk" });
  }
  if (quote?.quotative?.start != null && quote.quotative.end != null) {
    layers.push({ start: quote.quotative.start, end: quote.quotative.end, className: "qt" });
  }
  for (const mention of quote?.mentions || []) {
    layers.push({
      start: mention.start,
      end: mention.end,
      className: "men" + (mention.id === state.mentionId ? " selected" : ""),
    });
  }
  const cuts = new Set([0, text.length]);
  for (const layer of layers) {
    cuts.add(Math.max(0, Math.min(text.length, layer.start)));
    cuts.add(Math.max(0, Math.min(text.length, layer.end)));
  }
  const points = [...cuts].sort((a, b) => a - b);
  const root = document.createDocumentFragment();
  for (let i = 0; i < points.length - 1; i += 1) {
    const start = points[i];
    const end = points[i + 1];
    if (start === end) continue;
    const span = document.createElement("span");
    span.dataset.start = String(start);
    span.dataset.end = String(end);
    span.textContent = text.slice(start, end);
    const classes = [];
    for (const layer of layers) {
      if (layer.start <= start && end <= layer.end) classes.push(layer.className);
    }
    if (classes.length) span.className = classes.join(" ");
    root.appendChild(span);
  }
  return root;
}

function selectionOffsets() {
  const root = $("article-text");
  const sel = window.getSelection();
  if (!sel || sel.rangeCount === 0 || sel.isCollapsed) return null;
  const range = sel.getRangeAt(0);
  if (!root.contains(range.commonAncestorContainer)) return null;
  const pre = document.createRange();
  pre.selectNodeContents(root);
  pre.setEnd(range.startContainer, range.startOffset);
  const start = pre.toString().length;
  const end = start + range.toString().length;
  if (start === end) return null;
  return { start, end, text: (state.article.text || "").slice(start, end) };
}

function updateSelectionBar() {
  const sel = selectionOffsets();
  state.selection = sel;
  const bar = $("selection-bar");
  if (!sel) {
    bar.classList.add("hidden");
    return;
  }
  bar.classList.remove("hidden");
  $("selection-preview").textContent = `“${sel.text}”`;
}

function splitQuoteSelection(start, end) {
  const slice = (state.article.text || "").slice(start, end);
  const pairs = [
    ['"', '"'],
    ["\u201c", "\u201d"],
    ["\u00ab", "\u00bb"],
    ["``", "''"],
  ];
  for (const [open, close] of pairs) {
    if (slice.startsWith(open) && slice.endsWith(close) && slice.length > open.length + close.length) {
      return {
        outer_start: start,
        outer_end: end,
        inner_start: start + open.length,
        inner_end: end - close.length,
        delimiter: "manual",
      };
    }
  }
  return {
    outer_start: start,
    outer_end: end,
    inner_start: start,
    inner_end: end,
    delimiter: "manual",
  };
}

function nextQuoteId() {
  const ids = (state.article.quotes || []).map((q) => {
    const match = /q(\d+)$/i.exec(q.id || "");
    return match ? Number(match[1]) : -1;
  });
  const n = Math.max(-1, ...ids) + 1;
  return `${state.article.article_id}-q${n}`;
}

function boundaryOk(text, start, end) {
  if (start > 0 && /[A-Za-z0-9']/.test(text[start - 1])) return false;
  if (end < text.length && /[A-Za-z0-9']/.test(text[end])) return false;
  return true;
}

function scanQuotatives(quote) {
  const text = state.article?.text || "";
  const cues = state.article?.quotative_cues || [];
  const windowSize = 160;
  const ranges = [
    [Math.max(0, quote.outer_start - windowSize), quote.inner_start],
    [quote.outer_end, Math.min(text.length, quote.outer_end + windowSize)],
  ];
  const lower = text.toLowerCase();
  const found = [];
  const seen = new Set();
  for (const [left, right] of ranges) {
    for (const cue of cues) {
      let from = left;
      while (from < right) {
        const index = lower.indexOf(cue, from);
        if (index < 0 || index >= right) break;
        const end = index + cue.length;
        from = index + 1;
        if (!boundaryOk(text, index, end)) continue;
        const key = `${index}:${end}`;
        if (seen.has(key)) continue;
        seen.add(key);
        found.push({ start: index, end, surface: text.slice(index, end) });
      }
    }
  }
  return found;
}

function scanNames(quote) {
  const text = state.article?.text || "";
  const center = quote.outer_start || 0;
  const left = Math.max(0, center - 600);
  const right = Math.min(text.length, center + 600);
  const found = [];
  const seen = new Set();
  const slice = text.slice(left, right);
  NAME_RE.lastIndex = 0;
  let match = NAME_RE.exec(slice);
  while (match) {
    const surface = match[0];
    const start = left + match.index;
    const end = start + surface.length;
    match = NAME_RE.exec(slice);
    if (NAME_STOP.has(surface) || PRONOUNS.has(surface.toLowerCase())) continue;
    const key = surface.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    found.push({ start, end, surface, dist: Math.abs(start - center) });
  }
  found.sort((a, b) => a.dist - b.dist || a.start - b.start);
  return found.slice(0, 10).map(({ start, end, surface }) => ({ start, end, surface }));
}

function refreshSuggestions(quote) {
  quote.quotative_candidates = scanQuotatives(quote);
  quote.name_candidates = scanNames(quote);
}

function setSpeakerFromSelection() {
  const sel = state.selection || selectionOffsets();
  const quote = currentQuote();
  if (!sel || !quote) return;
  quote.status = quote.status === "reject" ? "keep" : quote.status === "pending" ? "keep" : quote.status;
  quote.speaker = {
    ...emptySpeaker(),
    status: "identified",
    start: sel.start,
    end: sel.end,
    surface: sel.text,
    form: inferForm(sel.text),
  };
  state.searchDraft.speaker = defaultSearchQuery(quote.speaker);
  markDirty();
  renderAll();
}

function cannotIdentifySpeaker() {
  const quote = currentQuote();
  if (!quote) return;
  quote.status = quote.status === "pending" ? "keep" : quote.status;
  quote.speaker = { ...emptySpeaker(), status: "cannot_identify" };
  state.searchDraft.speaker = "";
  markDirty();
  renderAll();
}

function setQuotativeFromSelection() {
  const sel = state.selection || selectionOffsets();
  const quote = currentQuote();
  if (!sel || !quote) return;
  quote.status = quote.status === "pending" ? "keep" : quote.status;
  quote.quotative = { status: "present", start: sel.start, end: sel.end, surface: sel.text };
  markDirty();
  renderAll();
}

function setQuotativeSpan(span) {
  const quote = currentQuote();
  if (!quote) return;
  quote.status = quote.status === "pending" ? "keep" : quote.status;
  quote.quotative = { status: "present", start: span.start, end: span.end, surface: span.surface };
  markDirty();
  renderAll();
}

function noQuotative() {
  const quote = currentQuote();
  if (!quote) return;
  quote.quotative = { status: "implicit", start: null, end: null, surface: "" };
  markDirty();
  renderAll();
}

function addMentionFromSelection() {
  const sel = state.selection || selectionOffsets();
  const quote = currentQuote();
  if (!sel || !quote) return;
  if (!(quote.inner_start <= sel.start && sel.end <= quote.inner_end)) {
    setSaveState("error", "Mention must lie inside the current quote");
    return;
  }
  quote.status = quote.status === "pending" ? "keep" : quote.status;
  const exists = (quote.mentions || []).some((m) => m.start === sel.start && m.end === sel.end);
  if (exists) return;
  const mention = {
    id: `${quote.id}-m${(quote.mentions || []).length}`,
    start: sel.start,
    end: sel.end,
    surface: sel.text,
    form: inferForm(sel.text),
    qid: null,
    qid_label: "",
    qid_description: "",
    nil: false,
    sentiment: null,
    notes: "",
  };
  quote.mentions = [...(quote.mentions || []), mention];
  state.mentionId = mention.id;
  state.searchDraft[searchDraftKey("mention")] = defaultSearchQuery(mention);
  markDirty();
  renderAll();
}

function addQuoteFromSelection() {
  const sel = state.selection || selectionOffsets();
  if (!sel || !state.article) return;
  const bounds = splitQuoteSelection(sel.start, sel.end);
  const quote = {
    id: nextQuoteId(),
    status: "pending",
    reject_reason: null,
    ...bounds,
    text: state.article.text.slice(bounds.inner_start, bounds.inner_end),
    speaker: emptySpeaker(),
    quotative: emptyQuotative(),
    mentions: [],
    notes: "",
  };
  refreshSuggestions(quote);
  state.article.quotes = [...(state.article.quotes || []), quote]
    .sort((a, b) => a.outer_start - b.outer_start || b.outer_end - a.outer_end);
  state.quoteIndex = state.article.quotes.findIndex((q) => q.id === quote.id);
  markDirty();
  renderAll();
}

function setQuoteBoundsFromSelection() {
  const sel = state.selection || selectionOffsets();
  const quote = currentQuote();
  if (!sel || !quote) return;
  const bounds = splitQuoteSelection(sel.start, sel.end);
  Object.assign(quote, bounds, { text: state.article.text.slice(bounds.inner_start, bounds.inner_end) });
  quote.mentions = (quote.mentions || []).filter(
    (m) => quote.inner_start <= m.start && m.end <= quote.inner_end,
  );
  refreshSuggestions(quote);
  markDirty();
  renderAll();
}

function keepQuote() {
  const quote = currentQuote();
  if (!quote) return;
  quote.status = "keep";
  quote.reject_reason = null;
  markDirty();
  renderQuotePane();
  renderArticle();
}

function rejectQuote() {
  const quote = currentQuote();
  if (!quote) return;
  quote.status = "reject";
  quote.reject_reason = quote.reject_reason || "not_speech";
  markDirty();
  renderQuotePane();
  renderArticle();
}

function renderQuotePane() {
  const quote = currentQuote();
  if (!state.article) {
    $("pane-empty").classList.remove("hidden");
    $("pane-body").classList.add("hidden");
    return;
  }
  $("pane-empty").classList.add("hidden");
  $("pane-body").classList.remove("hidden");
  if (!quote) {
    $("quote-text").textContent = "No quotation-mark spans in this article. Select a span and Add quote if one was missed.";
    $("quote-checklist").textContent = "";
    $("quotative-card").innerHTML = "";
    $("quotative-chips").innerHTML = "";
    $("speaker-card").innerHTML = "";
    $("name-chips").innerHTML = "";
    $("mention-list").innerHTML = "";
    $("reject-wrap").classList.add("hidden");
    return;
  }
  $("quote-text").textContent = quote.text;
  $("btn-keep").classList.toggle("active", quote.status === "keep");
  $("btn-reject").classList.toggle("active", quote.status === "reject");
  $("btn-cannot").classList.toggle("active", quote.speaker?.status === "cannot_identify");
  $("btn-no-quotative").classList.toggle("active", quote.quotative?.status === "implicit" || quote.quotative?.status === "none");
  if (quote.status === "reject") {
    $("reject-wrap").classList.remove("hidden");
    $("reject-reason").value = quote.reject_reason || "not_speech";
  } else {
    $("reject-wrap").classList.add("hidden");
  }
  $("quote-notes").value = quote.notes || "";
  $("article-notes").value = state.article.notes || "";
  $("quote-checklist").innerHTML = checklistHtml(quote);
  renderQuotative(quote);
  renderSpeaker(quote);
  renderMentions(quote);
}

function checklistHtml(quote) {
  const items = [];
  const add = (ok, label) => {
    items.push(`<span class="${ok ? "ok" : "bad"}">${ok ? "✓" : "○"} ${label}</span>`);
  };
  add(quote.status !== "pending", quote.status === "reject" ? "rejected" : "kept or rejected");
  if (quote.status !== "reject") {
    add(quote.quotative?.status === "present" || quote.quotative?.status === "implicit" || quote.quotative?.status === "none", "quotative decided");
    add(quote.speaker?.status === "identified" || quote.speaker?.status === "cannot_identify", "speaker decided");
    if (quote.speaker?.status === "identified") {
      add(Boolean(quote.speaker.nil || quote.speaker.qid), "speaker linked or not in Wikidata");
    }
    const mentions = quote.mentions || [];
    add(
      mentions.every((m) => m.nil || m.qid),
      mentions.length ? "mentions linked" : "no in-quote mentions",
    );
    add(
      mentions.every((m) => SENTIMENTS.includes(m.sentiment)),
      mentions.length ? "sentiment on every mention" : "sentiment n/a",
    );
  }
  return items.join(" · ");
}

function renderQuotative(quote) {
  const card = $("quotative-card");
  const quotative = quote.quotative || emptyQuotative();
  if (quotative.status === "implicit" || quotative.status === "none") {
    card.innerHTML = "<strong>Implicit</strong><p class='hint'>No reporting cue in the text; attribution is still assigned via the speaker.</p>";
  } else if (quotative.status === "present") {
    card.innerHTML = `<strong></strong> <span class="muted">reporting cue</span>`;
    card.querySelector("strong").textContent = quotative.surface;
  } else {
    card.innerHTML = "<span class='muted'>No quotative span yet.</span>";
  }
  const chips = $("quotative-chips");
  chips.innerHTML = "";
  for (const cand of quote.quotative_candidates || []) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = cand.surface;
    if (quotative.status === "present" && quotative.start === cand.start && quotative.end === cand.end) {
      btn.classList.add("active");
    }
    btn.addEventListener("click", () => setQuotativeSpan(cand));
    chips.appendChild(btn);
  }
}

function renderSpeaker(quote) {
  const speaker = quote.speaker || emptySpeaker();
  const card = $("speaker-card");
  if (speaker.status === "cannot_identify") {
    card.innerHTML = "<strong>Cannot identify speaker</strong><p class='hint'>The quote is kept or rejected, but no speaker span is assigned.</p>";
    $("name-chips").innerHTML = "";
    $("speaker-search").innerHTML = "";
    return;
  }
  if (speaker.status !== "identified") {
    card.innerHTML = "<span class='muted'>No speaker span yet.</span>";
    renderNameChips(quote, "speaker");
    $("speaker-search").innerHTML = "";
    return;
  }
  card.innerHTML = `
    <div><strong></strong> <span class="muted">${speaker.form}</span></div>
    <div class="qid"></div>
  `;
  card.querySelector("strong").textContent = speaker.surface;
  const qid = card.querySelector(".qid");
  if (speaker.nil) qid.textContent = "not in Wikidata";
  else if (speaker.qid) qid.textContent = `${speaker.qid} · ${speaker.qid_label}`;
  else qid.textContent = "needs Wikidata (or not in Wikidata)";
  renderNameChips(quote, "speaker");
  renderSearch("speaker", speaker);
}

function renderNameChips(quote, target) {
  const host = $("name-chips");
  host.innerHTML = "";
  const entity = target === "speaker"
    ? quote.speaker
    : (quote.mentions || []).find((m) => m.id === state.mentionId);
  const identified = target === "speaker"
    ? entity?.status === "identified"
    : Boolean(entity);
  if (identified) {
    for (const person of peopleAlreadyLinked()) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = person.qid_label || person.surface;
      btn.title = `${person.qid} (already linked in this article)`;
      btn.addEventListener("click", () => applyLink(target, person));
      host.appendChild(btn);
    }
  }
  const showNames = !identified || entity?.form === "pronoun" || entity?.form === "nominal";
  if (!showNames) return;
  for (const cand of quote.name_candidates || []) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = cand.surface;
    btn.addEventListener("click", () => applyNameCandidate(target, cand));
    host.appendChild(btn);
  }
}

function peopleAlreadyLinked() {
  const seen = new Set();
  const rows = [];
  for (const quote of state.article?.quotes || []) {
    const entities = [quote.speaker, ...(quote.mentions || [])];
    for (const entity of entities) {
      if (!entity?.qid || seen.has(entity.qid)) continue;
      seen.add(entity.qid);
      rows.push({
        qid: entity.qid,
        label: entity.qid_label,
        qid_label: entity.qid_label,
        description: entity.qid_description || "",
        surface: entity.surface,
        is_human: true,
      });
    }
  }
  return rows;
}

function applyNameCandidate(target, cand) {
  const quote = currentQuote();
  if (!quote) return;
  if (target === "speaker") {
    const speaker = quote.speaker || emptySpeaker();
    if (speaker.status !== "identified") {
      quote.status = quote.status === "pending" ? "keep" : quote.status;
      quote.speaker = {
        ...emptySpeaker(),
        status: "identified",
        start: cand.start,
        end: cand.end,
        surface: cand.surface,
        form: inferForm(cand.surface),
      };
    }
    state.searchDraft.speaker = cand.surface;
  } else {
    const mention = (quote.mentions || []).find((m) => m.id === state.mentionId);
    if (!mention) return;
    state.searchDraft[searchDraftKey("mention")] = cand.surface;
  }
  markDirty();
  renderAll();
  searchWikidata(target, cand.surface);
}

function renderMentions(quote) {
  const list = $("mention-list");
  list.innerHTML = "";
  for (const mention of quote.mentions || []) {
    const li = document.createElement("li");
    if (mention.id === state.mentionId) li.classList.add("selected");
    li.innerHTML = `
      <div><strong></strong> <span class="muted">${mention.form}</span>
        <button type="button" class="quiet del">remove</button></div>
      <div class="qid"></div>
      <div class="sentiment"></div>
    `;
    li.querySelector("strong").textContent = mention.surface;
    const qid = li.querySelector(".qid");
    if (mention.nil) qid.textContent = "not in Wikidata";
    else if (mention.qid) qid.textContent = `${mention.qid} · ${mention.qid_label}`;
    else qid.textContent = "needs Wikidata (or not in Wikidata)";
    const sentiment = li.querySelector(".sentiment");
    for (const label of SENTIMENTS) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = label.replace("_", " ");
      if (mention.sentiment === label) btn.classList.add("active");
      btn.addEventListener("click", (event) => {
        event.stopPropagation();
        mention.sentiment = label;
        markDirty();
        renderQuotePane();
      });
      sentiment.appendChild(btn);
    }
    li.querySelector(".del").addEventListener("click", (event) => {
      event.stopPropagation();
      quote.mentions = quote.mentions.filter((m) => m.id !== mention.id);
      if (state.mentionId === mention.id) state.mentionId = null;
      markDirty();
      renderAll();
    });
    li.addEventListener("click", () => {
      state.mentionId = mention.id;
      renderQuotePane();
    });
    list.appendChild(li);
  }
  const selected = (quote.mentions || []).find((m) => m.id === state.mentionId);
  if (selected) {
    renderNameChips(quote, "mention");
    renderSearch("mention", selected);
  } else {
    $("mention-search").innerHTML = "";
  }
}

function searchDraftKey(target) {
  return target === "speaker" ? "speaker" : `mention:${state.mentionId || ""}`;
}

function defaultSearchQuery(entity) {
  const surface = (entity?.surface || "").trim();
  if (!surface || PRONOUNS.has(surface.toLowerCase())) return "";
  return surface;
}

function renderSearch(target, entity) {
  const host = target === "speaker" ? $("speaker-search") : $("mention-search");
  const key = searchDraftKey(target);
  if (state.searchDraft[key] == null) state.searchDraft[key] = defaultSearchQuery(entity);
  const query = state.searchDraft[key] || "";
  const focused = host.contains(document.activeElement) && document.activeElement.tagName === "INPUT";
  if (!focused) {
    host.innerHTML = `
      <div class="search-label">Wikidata candidates — type a name if the right entity is not in the list</div>
      <input type="search" placeholder="Search Wikidata by name">
      <button type="button" class="quiet nil">Not in Wikidata</button>
      <ul class="search-hits"></ul>
    `;
    const input = host.querySelector("input");
    input.value = query;
    host.querySelector(".nil").addEventListener("click", () => applyLink(target, { nil: true }));
    input.addEventListener("input", () => {
      state.searchDraft[key] = input.value;
      clearTimeout(state.searchTimer);
      state.searchTimer = setTimeout(() => searchWikidata(target, input.value), 280);
    });
  }
  const trimmed = query.trim();
  if (trimmed.length >= 2 && !PRONOUNS.has(trimmed.toLowerCase())) searchWikidata(target, query);
  else {
    const hits = host.querySelector(".search-hits");
    if (hits) hits.innerHTML = "<li class='muted'>Type a name to search Wikidata</li>";
  }
}

async function searchWikidata(target, query) {
  const host = target === "speaker" ? $("speaker-search") : $("mention-search");
  const hits = host.querySelector(".search-hits");
  if (!hits) return;
  const trimmed = (query || "").trim();
  if (trimmed.length < 2 || PRONOUNS.has(trimmed.toLowerCase())) {
    hits.innerHTML = "<li class='muted'>Type a name to search Wikidata</li>";
    return;
  }
  hits.innerHTML = "<li class='muted'>searching…</li>";
  try {
    const rows = await api(`/api/wikidata?q=${encodeURIComponent(trimmed)}`);
    hits.innerHTML = "";
    if (!rows.length) {
      hits.innerHTML = "<li class='muted'>No hits — try a different name</li>";
      return;
    }
    for (const row of rows) {
      const li = document.createElement("li");
      const extra = [row.description, row.occupation].filter(Boolean).join(" · ");
      li.innerHTML = `<span class="qid">${row.qid}</span> <strong></strong><div class="muted"></div>`;
      li.querySelector("strong").textContent = row.label + (row.fictional ? " (fictional)" : "");
      li.querySelector(".muted").textContent = extra;
      li.addEventListener("click", () => applyLink(target, row));
      hits.appendChild(li);
    }
  } catch (err) {
    hits.innerHTML = "";
    hits.textContent = err.message;
  }
}

function applyLink(target, row) {
  const quote = currentQuote();
  if (!quote) return;
  const patch = row.nil
    ? { qid: null, qid_label: "", qid_description: "", nil: true }
    : {
        qid: row.qid,
        qid_label: row.label || row.qid_label,
        qid_description: row.description || row.qid_description || "",
        nil: false,
      };
  if (target === "speaker") {
    const prev = quote.speaker || emptySpeaker();
    if (prev.status !== "identified") {
      setSaveState("error", "Set the speaker span first");
      return;
    }
    quote.speaker = { ...prev, ...patch, status: "identified" };
  } else {
    const mention = (quote.mentions || []).find((m) => m.id === state.mentionId);
    if (!mention) return;
    Object.assign(mention, patch);
  }
  markDirty();
  renderQuotePane();
  renderArticle();
}

function bind() {
  $("annotator").value = state.annotator;
  $("annotator").addEventListener("change", async (event) => {
    state.annotator = event.target.value.trim() || "default";
    localStorage.setItem("qg-annotator", state.annotator);
    await loadBatch();
    if (state.article) await openArticle(state.article.article_id);
  });
  $("btn-save").addEventListener("click", () => saveArticle());
  $("btn-help").addEventListener("click", () => $("help").showModal());
  $("btn-keep").addEventListener("click", keepQuote);
  $("btn-reject").addEventListener("click", rejectQuote);
  $("reject-reason").addEventListener("change", (event) => {
    const quote = currentQuote();
    if (!quote) return;
    quote.reject_reason = event.target.value;
    markDirty();
  });
  $("btn-cannot").addEventListener("click", cannotIdentifySpeaker);
  $("btn-clear-speaker").addEventListener("click", () => {
    const quote = currentQuote();
    if (!quote) return;
    quote.speaker = emptySpeaker();
    state.searchDraft.speaker = "";
    markDirty();
    renderAll();
  });
  $("btn-no-quotative").addEventListener("click", noQuotative);
  $("btn-clear-quotative").addEventListener("click", () => {
    const quote = currentQuote();
    if (!quote) return;
    quote.quotative = emptyQuotative();
    markDirty();
    renderAll();
  });
  $("btn-set-speaker").addEventListener("click", setSpeakerFromSelection);
  $("btn-set-quotative").addEventListener("click", setQuotativeFromSelection);
  $("btn-add-mention").addEventListener("click", addMentionFromSelection);
  $("btn-set-bounds").addEventListener("click", setQuoteBoundsFromSelection);
  $("btn-add-quote").addEventListener("click", addQuoteFromSelection);
  $("quote-notes").addEventListener("input", (event) => {
    const quote = currentQuote();
    if (!quote) return;
    quote.notes = event.target.value;
    markDirty();
  });
  $("article-notes").addEventListener("input", (event) => {
    if (!state.article) return;
    state.article.notes = event.target.value;
    markDirty();
  });
  $("btn-skip").addEventListener("click", () => {
    if (!state.article) return;
    const reason = window.prompt("Skip reason (garbled text, not English, …)", state.article.skip_reason || "");
    if (reason == null) return;
    state.article.skipped = true;
    state.article.skip_reason = reason;
    markDirty();
    saveArticle();
  });

  const article = $("article-text");
  article.addEventListener("mouseup", updateSelectionBar);
  article.addEventListener("keyup", updateSelectionBar);
  article.addEventListener("click", (event) => {
    const span = event.target.closest("span[data-start]");
    if (!span || !state.article) return;
    if (window.getSelection() && !window.getSelection().isCollapsed) return;
    const offset = Number(span.dataset.start);
    const quotes = state.article.quotes || [];
    const hits = quotes.filter((q) => q.outer_start <= offset && offset < q.outer_end);
    if (!hits.length) return;
    hits.sort((a, b) => (a.outer_end - a.outer_start) - (b.outer_end - b.outer_start));
    state.quoteIndex = quotes.findIndex((q) => q.id === hits[0].id);
    state.mentionId = null;
    renderAll();
  });

  document.addEventListener("keydown", (event) => {
    const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName) || event.target.isContentEditable;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      saveArticle();
      return;
    }
    if (typing) return;
    if (event.key === "?") {
      $("help").showModal();
      return;
    }
    if (event.key === "k") keepQuote();
    if (event.key === "r") rejectQuote();
    if (event.key === "q") setQuotativeFromSelection();
    if (event.key === "i") noQuotative();
    if (event.key === "s") setSpeakerFromSelection();
    if (event.key === "m") addMentionFromSelection();
    if (event.key === "c") cannotIdentifySpeaker();
    if (event.key === "[" || event.key === "ArrowLeft") {
      state.quoteIndex = Math.max(0, state.quoteIndex - 1);
      renderAll();
    }
    if (event.key === "]" || event.key === "ArrowRight") {
      const n = state.article?.quotes?.length || 1;
      state.quoteIndex = Math.min(n - 1, state.quoteIndex + 1);
      renderAll();
    }
  });

  window.addEventListener("beforeunload", (event) => {
    if (state.dirty) event.preventDefault();
  });
}

async function boot() {
  bind();
  try {
    await loadBatch();
    if (state.batch.length) await openArticle(state.batch[0].article_id);
  } catch (err) {
    $("article-title").textContent = err.message;
  }
}

boot();
