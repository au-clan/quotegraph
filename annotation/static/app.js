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

function emptySpan() {
  return { start: null, end: null, surface: "" };
}

function copySpan(span) {
  if (!span || typeof span !== "object") return emptySpan();
  return {
    start: span.start ?? null,
    end: span.end ?? null,
    surface: span.surface || "",
  };
}

function hasSpan(span) {
  return span != null && span.start != null && span.end != null;
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
    intro_phrase: emptySpan(),
    first_span: emptySpan(),
  };
}

function keepSpeakerAnnotations(prev) {
  const src = prev || emptySpeaker();
  return {
    qid: src.qid ?? null,
    qid_label: src.qid_label || "",
    qid_description: src.qid_description || "",
    nil: Boolean(src.nil),
    intro_phrase: copySpan(src.intro_phrase),
    first_span: copySpan(src.first_span),
  };
}

function ensureSpeaker(quote) {
  quote.speaker = quote.speaker || emptySpeaker();
  quote.speaker.intro_phrase = copySpan(quote.speaker.intro_phrase);
  quote.speaker.first_span = copySpan(quote.speaker.first_span);
  return quote.speaker;
}

function emptyQuotative() {
  return { status: "unset", start: null, end: null, surface: "" };
}

function currentQuote() {
  const quotes = state.article?.quotes || [];
  if (!quotes.length) return null;
  return quotes[Math.min(state.quoteIndex, quotes.length - 1)];
}

function quoteSegments(quote) {
  if (quote?.segments?.length >= 2) return quote.segments;
  if (!quote) return [];
  return [{
    inner_start: quote.inner_start,
    inner_end: quote.inner_end,
    outer_start: quote.outer_start,
    outer_end: quote.outer_end,
    delimiter: quote.delimiter || "",
    text: quote.text || "",
  }];
}

function mentionInsideQuote(quote, start, end) {
  return quoteSegments(quote).some((seg) => seg.inner_start <= start && end <= seg.inner_end);
}

function quoteDisplayText(quote) {
  const segs = quoteSegments(quote);
  if (segs.length <= 1) return quote.text || "";
  return segs.map((seg) => seg.text || "").join(" […] ");
}

function quotesAreMergeable(left, right) {
  const leftEnd = Math.max(...quoteSegments(left).map((seg) => seg.outer_end));
  const rightStart = Math.min(...quoteSegments(right).map((seg) => seg.outer_start));
  return rightStart >= leftEnd;
}

function syncCoveringFields(quote, text) {
  const segs = quoteSegments(quote);
  segs.sort((a, b) => a.outer_start - b.outer_start);
  for (const seg of segs) seg.text = text.slice(seg.inner_start, seg.inner_end);
  quote.outer_start = segs[0].outer_start;
  quote.outer_end = segs[segs.length - 1].outer_end;
  quote.inner_start = segs[0].inner_start;
  quote.inner_end = segs[segs.length - 1].inner_end;
  quote.text = segs.map((seg) => seg.text).join(" […] ");
  if (segs.length >= 2) quote.segments = segs;
  else delete quote.segments;
  return quote;
}

function preferSpeaker(left, right) {
  const rank = { identified: 2, cannot_identify: 1, unset: 0 };
  const ls = left || emptySpeaker();
  const rs = right || emptySpeaker();
  return (rank[rs.status] || 0) > (rank[ls.status] || 0) ? { ...rs } : { ...ls };
}

function preferQuotative(left, right) {
  const rank = { present: 2, implicit: 1, none: 1, unset: 0 };
  const lq = left || emptyQuotative();
  const rq = right || emptyQuotative();
  return (rank[rq.status] || 0) > (rank[lq.status] || 0) ? { ...rq } : { ...lq };
}

function mergeQuoteRecords(left, right, text) {
  const segs = [...quoteSegments(left), ...quoteSegments(right)]
    .map((seg) => ({ ...seg }))
    .sort((a, b) => a.outer_start - b.outer_start);
  const merged = { ...left, segments: segs };
  syncCoveringFields(merged, text);
  const ids = [...(left.merged_from || [left.id]), ...(right.merged_from || [right.id])];
  merged.merged_from = [...new Set(ids)];
  merged.speaker = preferSpeaker(left.speaker, right.speaker);
  merged.quotative = preferQuotative(left.quotative, right.quotative);
  merged.mentions = [...(left.mentions || []), ...(right.mentions || [])]
    .filter((m) => mentionInsideQuote(merged, m.start, m.end));
  if (left.status === "keep" || right.status === "keep") {
    merged.status = "keep";
    merged.reject_reason = null;
  } else if (left.status === "reject" && right.status === "reject") {
    merged.status = "reject";
    merged.reject_reason = left.reject_reason || right.reject_reason || "other";
  } else {
    merged.status = "pending";
    merged.reject_reason = null;
  }
  merged.notes = [left.notes, right.notes].filter(Boolean).join(" ");
  return merged;
}

function spanOwnerIndex(quote, start, end) {
  if (start == null || end == null) return 0;
  const segs = quoteSegments(quote);
  for (let i = 0; i < segs.length; i += 1) {
    const seg = segs[i];
    if (seg.outer_start <= start && end <= seg.outer_end) return i;
    if (i + 1 < segs.length && seg.outer_end <= start && end <= segs[i + 1].outer_start) return i;
  }
  if (end <= segs[0].outer_start) return 0;
  return segs.length - 1;
}

function unmergeQuoteRecord(quote, text) {
  const segs = quoteSegments(quote);
  if (segs.length < 2) return [quote];
  const ids = quote.merged_from || [];
  const speaker = quote.speaker || emptySpeaker();
  const quotative = quote.quotative || emptyQuotative();
  const speakerI = speaker.status === "identified" ? spanOwnerIndex(quote, speaker.start, speaker.end) : 0;
  const quotativeI = quotative.status === "present" ? spanOwnerIndex(quote, quotative.start, quotative.end) : 0;
  return segs.map((seg, i) => {
    const piece = {
      id: ids[i] || `${quote.id}-p${i}`,
      status: quote.status || "pending",
      reject_reason: quote.reject_reason || null,
      inner_start: seg.inner_start,
      inner_end: seg.inner_end,
      outer_start: seg.outer_start,
      outer_end: seg.outer_end,
      delimiter: seg.delimiter || quote.delimiter || "",
      text: text.slice(seg.inner_start, seg.inner_end),
      speaker: emptySpeaker(),
      quotative: emptyQuotative(),
      mentions: (quote.mentions || []).filter(
        (m) => seg.inner_start <= m.start && m.end <= seg.inner_end,
      ),
      notes: "",
    };
    if (i === speakerI) piece.speaker = { ...speaker };
    if (i === quotativeI) piece.quotative = { ...quotative };
    else if ((quotative.status === "implicit" || quotative.status === "none") && i === 0) {
      piece.quotative = { ...quotative };
    }
    return piece;
  });
}

function offsetInQuote(quote, offset) {
  const segs = quoteSegments(quote);
  if (segs.some((seg) => seg.outer_start <= offset && offset < seg.outer_end)) return true;
  if (segs.length > 1 && quote.outer_start <= offset && offset < quote.outer_end) return true;
  return false;
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
    if (!hasSpan(speaker.first_span)) return false;
  }
  const quotative = quote.quotative || {};
  if (quotative.status !== "present" && quotative.status !== "implicit" && quotative.status !== "none") return false;
  if (quotative.status === "present" && (quotative.start == null || quotative.end == null)) return false;
  for (const mention of quote.mentions || []) {
    if (!SENTIMENTS.includes(mention.sentiment)) return false;
    if (!mention.nil && !mention.qid) return false;
    if (!hasSpan(mention.first_span)) return false;
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
  state.selection = null;
  clearSelectionEchoes();
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
    clearSelectionEchoes();
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
    const nSeg = quoteSegments(quote).length;
    if (nSeg > 1) btn.textContent += `×${nSeg}`;
    btn.title = nSeg > 1 ? `${quote.status} · ${nSeg} spans` : quote.status;
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
  applySelectionEchoes();
}

function renderHighlightedText(article) {
  const text = article.text || "";
  const quote = currentQuote();
  const layers = [];
  for (const q of article.quotes || []) {
    for (const seg of quoteSegments(q)) {
      layers.push({
        start: seg.outer_start,
        end: seg.outer_end,
        className: "q" + (quote && q.id === quote.id ? " current" : ""),
      });
    }
  }
  if (quote?.speaker?.start != null && quote.speaker.end != null) {
    layers.push({ start: quote.speaker.start, end: quote.speaker.end, className: "spk" });
  }
  const selectedMention = (quote?.mentions || []).find((m) => m.id === state.mentionId);
  const auxEntity = selectedMention || quote?.speaker;
  if (hasSpan(auxEntity?.intro_phrase)) {
    layers.push({ start: auxEntity.intro_phrase.start, end: auxEntity.intro_phrase.end, className: "spk-intro" });
  }
  if (hasSpan(auxEntity?.first_span)) {
    layers.push({ start: auxEntity.first_span.start, end: auxEntity.first_span.end, className: "spk-first" });
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
  const slice = (state.article.text || "").slice(start, end);
  return { start, end, text: slice, tokenBounded: boundaryOk(state.article.text || "", start, end) };
}

function updateSelectionBar() {
  const sel = selectionOffsets();
  state.selection = sel;
  const bar = $("selection-bar");
  if (!sel) {
    bar.classList.add("hidden");
    clearSelectionEchoes();
    return;
  }
  bar.classList.remove("hidden");
  const others = findEchoOffsets(state.article?.text || "", sel.text, sel.start, sel.end, sel.tokenBounded);
  $("selection-preview").textContent = others.length
    ? `“${sel.text}” · ${others.length} other`
    : `“${sel.text}”`;
  applySelectionEchoes();
}

function findEchoOffsets(text, needle, skipStart, skipEnd, requireBoundary) {
  const hits = [];
  if (!text || !needle || !needle.trim()) return hits;
  if (requireBoundary == null) {
    requireBoundary = skipStart >= 0 && skipEnd > skipStart && boundaryOk(text, skipStart, skipEnd);
  }
  if (needle.length === 1 && !requireBoundary) return hits;
  let from = 0;
  while (from <= text.length - needle.length) {
    const index = text.indexOf(needle, from);
    if (index < 0) break;
    const end = index + needle.length;
    from = end;
    if (index === skipStart && end === skipEnd) continue;
    if (requireBoundary && !boundaryOk(text, index, end)) continue;
    hits.push({ start: index, end });
    if (hits.length >= 250) break;
  }
  return hits;
}

function offsetsToDomRange(root, start, end) {
  let startNode = null;
  let startOff = 0;
  let endNode = null;
  let endOff = 0;
  for (const el of root.querySelectorAll("span[data-start]")) {
    const a = Number(el.dataset.start);
    const b = Number(el.dataset.end);
    const textNode = el.firstChild;
    if (!textNode || textNode.nodeType !== Node.TEXT_NODE || a === b) continue;
    if (startNode == null && start >= a && start < b) {
      startNode = textNode;
      startOff = start - a;
    }
    if (end > a && end <= b) {
      endNode = textNode;
      endOff = end - a;
    }
  }
  if (!startNode || !endNode) return null;
  const range = document.createRange();
  range.setStart(startNode, Math.min(startOff, startNode.length));
  range.setEnd(endNode, Math.min(endOff, endNode.length));
  return range;
}

function clearSelectionEchoes() {
  if (globalThis.CSS?.highlights) CSS.highlights.delete("sel-echo");
}

function applySelectionEchoes() {
  clearSelectionEchoes();
  const sel = state.selection;
  const root = $("article-text");
  const text = state.article?.text || "";
  if (!sel || !root || !text || !globalThis.CSS?.highlights) return;
  const live = selectionOffsets();
  const skipStart = live ? sel.start : -1;
  const skipEnd = live ? sel.end : -1;
  const hits = findEchoOffsets(text, sel.text, skipStart, skipEnd, sel.tokenBounded);
  if (!hits.length) return;
  const ranges = [];
  for (const hit of hits) {
    const range = offsetsToDomRange(root, hit.start, hit.end);
    if (range) ranges.push(range);
  }
  if (ranges.length) CSS.highlights.set("sel-echo", new Highlight(...ranges));
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
  const segs = quoteSegments(quote).slice().sort((a, b) => a.outer_start - b.outer_start);
  const ranges = [];
  if (segs.length >= 2) {
    ranges.push([Math.max(0, segs[0].outer_start - windowSize), segs[0].inner_start]);
    for (let i = 0; i < segs.length - 1; i += 1) {
      ranges.push([segs[i].outer_end, segs[i + 1].outer_start]);
    }
    const last = segs[segs.length - 1];
    ranges.push([last.outer_end, Math.min(text.length, last.outer_end + windowSize)]);
  } else {
    ranges.push(
      [Math.max(0, quote.outer_start - windowSize), quote.inner_start],
      [quote.outer_end, Math.min(text.length, quote.outer_end + windowSize)],
    );
  }
  const inners = segs.map((seg) => [seg.inner_start, seg.inner_end]);
  const lower = text.toLowerCase();
  const found = [];
  const seen = new Set();
  for (const [left, right] of ranges) {
    if (right <= left) continue;
    for (const cue of cues) {
      let from = left;
      while (from < right) {
        const index = lower.indexOf(cue, from);
        if (index < 0 || index >= right) break;
        const end = index + cue.length;
        from = index + 1;
        if (!boundaryOk(text, index, end)) continue;
        if (inners.some(([innerStart, innerEnd]) => innerStart <= index && end <= innerEnd)) continue;
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
  const center = quoteSegments(quote).length >= 2
    ? quoteSegments(quote)[0].outer_end
    : (quote.outer_start || 0);
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
    ...keepSpeakerAnnotations(quote.speaker),
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

function setAuxFromSelection(field) {
  const sel = state.selection || selectionOffsets();
  const quote = currentQuote();
  if (!sel || !quote) return;
  const target = auxTarget();
  const entity = currentAuxEntity(quote, target);
  if (!entity) {
    setSaveState("error", target === "mention" ? "Select a mention first" : "Set the speaker span first");
    return;
  }
  if (target === "speaker" && entity.status === "cannot_identify") {
    setSaveState("error", "Clear cannot-identify before setting intro or first span");
    return;
  }
  ensureEntityAux(entity);
  entity[field] = { start: sel.start, end: sel.end, surface: sel.text };
  markDirty();
  renderAll();
}

function clearAux(field, target, entity) {
  const quote = currentQuote();
  if (!quote) return;
  const dest = entity || currentAuxEntity(quote, target || auxTarget());
  if (!dest) return;
  ensureEntityAux(dest);
  dest[field] = emptySpan();
  markDirty();
  renderAll();
}

function auxTarget() {
  return state.mentionId ? "mention" : "speaker";
}

function currentAuxEntity(quote, target) {
  if (target === "mention") {
    const mention = (quote.mentions || []).find((m) => m.id === state.mentionId);
    if (mention) ensureEntityAux(mention);
    return mention || null;
  }
  if ((quote.speaker || {}).status === "cannot_identify") return quote.speaker;
  return ensureSpeaker(quote);
}

function ensureEntityAux(entity) {
  entity.intro_phrase = copySpan(entity.intro_phrase);
  entity.first_span = copySpan(entity.first_span);
  return entity;
}

function applyAuxSpan(target, field, span, source, entity) {
  const quote = currentQuote();
  if (!quote) return;
  const dest = entity || currentAuxEntity(quote, target);
  if (!dest) return;
  ensureEntityAux(dest);
  dest[field] = copySpan(span);
  copyAuxIfEmpty(dest, source || {});
  if (source?.qid && !dest.qid && !dest.nil) {
    dest.qid = source.qid;
    dest.qid_label = source.qid_label || source.label || "";
    dest.qid_description = source.qid_description || source.description || "";
  }
  markDirty();
  renderAll();
}

function copyAuxIfEmpty(entity, source) {
  ensureEntityAux(entity);
  let intro = hasSpan(source?.intro_phrase) ? copySpan(source.intro_phrase) : emptySpan();
  let first = hasSpan(source?.first_span) ? copySpan(source.first_span) : emptySpan();
  if (source?.qid) {
    const fromQid = auxFromQid(source.qid);
    if (!hasSpan(intro)) intro = fromQid.intro_phrase;
    if (!hasSpan(first)) first = fromQid.first_span;
  }
  if (!hasSpan(entity.intro_phrase) && hasSpan(intro)) entity.intro_phrase = intro;
  if (!hasSpan(entity.first_span) && hasSpan(first)) entity.first_span = first;
}

function auxFromQid(qid) {
  const out = { intro_phrase: emptySpan(), first_span: emptySpan() };
  if (!qid) return out;
  for (const quote of state.article?.quotes || []) {
    for (const entity of [quote.speaker, ...(quote.mentions || [])]) {
      if (entity?.qid !== qid) continue;
      if (!hasSpan(out.intro_phrase) && hasSpan(entity.intro_phrase)) out.intro_phrase = copySpan(entity.intro_phrase);
      if (!hasSpan(out.first_span) && hasSpan(entity.first_span)) out.first_span = copySpan(entity.first_span);
    }
  }
  return out;
}

function knownAuxSpans(field) {
  const seen = new Set();
  const rows = [];
  for (const quote of state.article?.quotes || []) {
    for (const entity of [quote.speaker, ...(quote.mentions || [])]) {
      const span = entity?.[field];
      if (!hasSpan(span)) continue;
      const key = `${span.start}:${span.end}`;
      if (seen.has(key)) continue;
      seen.add(key);
      rows.push({
        ...copySpan(span),
        qid: entity.qid || null,
        qid_label: entity.qid_label || "",
        qid_description: entity.qid_description || "",
        label: entity.qid_label || entity.surface,
        intro_phrase: copySpan(entity.intro_phrase),
        first_span: copySpan(entity.first_span),
      });
    }
  }
  return rows;
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
  if (!mentionInsideQuote(quote, sel.start, sel.end)) {
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
    intro_phrase: emptySpan(),
    first_span: emptySpan(),
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
  delete quote.segments;
  delete quote.merged_from;
  quote.mentions = (quote.mentions || []).filter(
    (m) => mentionInsideQuote(quote, m.start, m.end),
  );
  refreshSuggestions(quote);
  markDirty();
  renderAll();
}

function mergeQuoteWithNeighbor(direction) {
  const quotes = state.article?.quotes || [];
  const i = state.quoteIndex;
  const j = direction === "next" ? i + 1 : i - 1;
  if (!state.article || j < 0 || j >= quotes.length) {
    setSaveState("error", direction === "next" ? "No next quote to merge" : "No previous quote to merge");
    return;
  }
  const leftIndex = Math.min(i, j);
  const rightIndex = Math.max(i, j);
  if (!quotesAreMergeable(quotes[leftIndex], quotes[rightIndex])) {
    setSaveState("error", "Can only merge adjacent, non-overlapping spans");
    return;
  }
  const merged = mergeQuoteRecords(quotes[leftIndex], quotes[rightIndex], state.article.text);
  refreshSuggestions(merged);
  quotes.splice(leftIndex, 2, merged);
  state.quoteIndex = leftIndex;
  state.mentionId = null;
  markDirty();
  renderAll();
}

function unmergeCurrentQuote() {
  const quote = currentQuote();
  if (!quote || !state.article || quoteSegments(quote).length < 2) {
    setSaveState("error", "This quote is not a merged quotation");
    return;
  }
  const pieces = unmergeQuoteRecord(quote, state.article.text);
  for (const piece of pieces) refreshSuggestions(piece);
  const quotes = state.article.quotes;
  quotes.splice(state.quoteIndex, 1, ...pieces);
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
    $("btn-merge-next").disabled = true;
    $("btn-merge-prev").disabled = true;
    $("btn-unmerge").disabled = true;
    $("btn-unmerge").classList.add("hidden");
    return;
  }
  $("quote-text").textContent = quoteDisplayText(quote);
  const nSeg = quoteSegments(quote).length;
  const quotes = state.article.quotes || [];
  const i = state.quoteIndex;
  $("btn-merge-next").disabled = i >= quotes.length - 1 || !quotesAreMergeable(quotes[i], quotes[i + 1]);
  $("btn-merge-prev").disabled = i <= 0 || !quotesAreMergeable(quotes[i - 1], quotes[i]);
  $("btn-unmerge").disabled = nSeg < 2;
  $("btn-unmerge").classList.toggle("hidden", nSeg < 2);
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
  const onMention = Boolean(state.mentionId);
  $("btn-set-intro").textContent = onMention ? "Intro phrase (mention)" : "Intro phrase";
  $("btn-set-first").textContent = onMention ? "First span (mention)" : "First span";
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
      add(hasSpan(quote.speaker.first_span), "speaker first span");
    }
    const mentions = quote.mentions || [];
    add(
      mentions.every((m) => m.nil || m.qid),
      mentions.length ? "mentions linked" : "no in-quote mentions",
    );
    add(
      mentions.every((m) => hasSpan(m.first_span)),
      mentions.length ? "mention first spans" : "mention first span n/a",
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
  card.onclick = () => {
    if (!state.mentionId) return;
    state.mentionId = null;
    renderAll();
  };
  if (speaker.status === "cannot_identify") {
    card.innerHTML = "<strong>Cannot identify speaker</strong><p class='hint'>The quote is kept or rejected, but no speaker span is assigned.</p>";
    $("name-chips").innerHTML = "";
    $("speaker-search").innerHTML = "";
    return;
  }
  if (speaker.status !== "identified") {
    card.innerHTML = "<span class='muted'>No speaker span yet.</span>";
    card.appendChild(entityAuxBlock(speaker, "speaker"));
    renderNameChips(quote, "speaker");
    $("speaker-search").innerHTML = "";
    return;
  }
  card.innerHTML = `
    <div><strong></strong> <span class="muted">${speaker.form}</span></div>
    <div class="qid"></div>
  `;
  card.querySelector("strong").textContent = speaker.surface;
  fillQidLine(card.querySelector(".qid"), speaker);
  card.appendChild(entityAuxBlock(speaker, "speaker"));
  renderNameChips(quote, "speaker");
  renderSearch("speaker", speaker);
}

function entityAuxBlock(entity, target) {
  const wrap = document.createElement("div");
  wrap.className = "speaker-aux-list";
  wrap.appendChild(entityAuxRow(entity, target, "intro_phrase", "Intro phrase", "Role or description that introduces this entity (the president of the US)."));
  wrap.appendChild(entityAuxRow(entity, target, "first_span", "First span (required)", "First mention of this entity in the article."));
  return wrap;
}

function entityAuxRow(entity, target, field, label, hint) {
  const row = document.createElement("div");
  row.className = "speaker-aux";
  const head = document.createElement("div");
  head.className = "speaker-aux-head";
  const title = document.createElement("span");
  title.className = "muted";
  title.textContent = label;
  head.appendChild(title);
  const span = entity[field];
  if (hasSpan(span)) {
    const clear = document.createElement("button");
    clear.type = "button";
    clear.className = "quiet";
    clear.textContent = "Clear";
    clear.addEventListener("click", (event) => {
      event.stopPropagation();
      if (target === "mention") state.mentionId = entity.id;
      clearAux(field, target, entity);
    });
    head.appendChild(clear);
  }
  row.appendChild(head);
  const surface = document.createElement("div");
  if (hasSpan(span)) {
    surface.className = field === "intro_phrase" ? "spk-intro-label" : "spk-first-label";
    surface.textContent = span.surface;
  } else {
    surface.className = "muted";
    surface.textContent = hint;
  }
  row.appendChild(surface);
  const reuse = knownAuxSpans(field);
  const chips = document.createElement("div");
  chips.className = "chips";
  for (const item of reuse) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = item.surface;
    btn.title = item.qid_label ? `${item.qid_label} — reuse this span` : "Reuse this span";
    if (hasSpan(span) && span.start === item.start && span.end === item.end) btn.classList.add("active");
    btn.addEventListener("click", (event) => {
      event.stopPropagation();
      if (target === "mention") state.mentionId = entity.id;
      applyAuxSpan(target, field, item, item, entity);
    });
    chips.appendChild(btn);
  }
  if (chips.childNodes.length) row.appendChild(chips);
  return row;
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
  const byQid = new Map();
  for (const quote of state.article?.quotes || []) {
    for (const entity of [quote.speaker, ...(quote.mentions || [])]) {
      if (!entity?.qid) continue;
      const prev = byQid.get(entity.qid) || {
        qid: entity.qid,
        label: entity.qid_label,
        qid_label: entity.qid_label,
        description: entity.qid_description || "",
        qid_description: entity.qid_description || "",
        surface: entity.surface,
        is_human: true,
        intro_phrase: emptySpan(),
        first_span: emptySpan(),
      };
      if (!prev.qid_label && entity.qid_label) {
        prev.qid_label = entity.qid_label;
        prev.label = entity.qid_label;
      }
      if (!hasSpan(prev.intro_phrase) && hasSpan(entity.intro_phrase)) prev.intro_phrase = copySpan(entity.intro_phrase);
      if (!hasSpan(prev.first_span) && hasSpan(entity.first_span)) prev.first_span = copySpan(entity.first_span);
      byQid.set(entity.qid, prev);
    }
  }
  return [...byQid.values()];
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
        ...keepSpeakerAnnotations(speaker),
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
    fillQidLine(li.querySelector(".qid"), mention);
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
    li.appendChild(entityAuxBlock(mention, "mention"));
    li.addEventListener("click", () => {
      state.mentionId = mention.id;
      renderAll();
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

function wikidataItemUrl(qid) {
  return `https://www.wikidata.org/wiki/${encodeURIComponent(qid)}`;
}

function googleSearchUrl(query) {
  return `https://www.google.com/search?q=${encodeURIComponent(query)}`;
}

function wikidataSearchUrl(query) {
  return `https://www.wikidata.org/w/index.php?search=${encodeURIComponent(query)}`;
}

function ICON_GOOGLE() {
  return `<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>`;
}

function ICON_WIKIDATA() {
  return `<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><rect x="1" y="4" width="1.6" height="8" fill="#990000"/><rect x="3.4" y="2.5" width="1.6" height="11" fill="#339966"/><rect x="5.8" y="5" width="1.6" height="6" fill="#006699"/><rect x="8.2" y="3" width="1.6" height="10" fill="#990000"/><rect x="10.6" y="4.5" width="1.6" height="7" fill="#339966"/><rect x="13" y="2.5" width="1.6" height="11" fill="#006699"/></svg>`;
}

function fillQidLine(el, entity) {
  el.replaceChildren();
  if (entity?.nil) {
    el.textContent = "not in Wikidata";
    return;
  }
  if (entity?.qid) {
    const link = document.createElement("a");
    link.href = wikidataItemUrl(entity.qid);
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = `${entity.qid} · ${entity.qid_label || ""}`.trim();
    link.title = "Open on Wikidata";
    link.addEventListener("click", (event) => event.stopPropagation());
    el.appendChild(link);
    return;
  }
  el.textContent = "needs Wikidata (or not in Wikidata)";
}

function searchQueryFor(target, entity) {
  const draft = (state.searchDraft[searchDraftKey(target)] || "").trim();
  if (draft && !PRONOUNS.has(draft.toLowerCase())) return draft;
  return defaultSearchQuery(entity);
}

function bindLookupLinks(host, query) {
  const google = host.querySelector("a.js-google");
  const wikidata = host.querySelector("a.js-wikidata-search");
  if (google) {
    google.href = query ? googleSearchUrl(query) : "#";
    google.classList.toggle("disabled", !query);
  }
  if (wikidata) {
    wikidata.href = query ? wikidataSearchUrl(query) : "#";
    wikidata.classList.toggle("disabled", !query);
  }
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
      <div class="search-actions">
        <button type="button" class="quiet nil" title="N">Not in Wikidata</button>
        <a class="icon-btn js-google" target="_blank" rel="noopener noreferrer" title="Google this name">${ICON_GOOGLE()}</a>
        <a class="icon-btn js-wikidata-search" target="_blank" rel="noopener noreferrer" title="Search Wikidata for this name">${ICON_WIKIDATA()}</a>
      </div>
      <ul class="search-hits"></ul>
    `;
    const input = host.querySelector("input");
    input.value = query;
    host.querySelector(".nil").addEventListener("click", () => applyLink(target, { nil: true }));
    input.addEventListener("input", () => {
      state.searchDraft[key] = input.value;
      bindLookupLinks(host, searchQueryFor(target, entity));
      clearTimeout(state.searchTimer);
      state.searchTimer = setTimeout(() => searchWikidata(target, input.value), 280);
    });
  }
  bindLookupLinks(host, searchQueryFor(target, entity));
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
      li.innerHTML = `<a class="qid-link" target="_blank" rel="noopener noreferrer"></a> <strong></strong><div class="muted"></div>`;
      const qidLink = li.querySelector(".qid-link");
      qidLink.href = wikidataItemUrl(row.qid);
      qidLink.textContent = row.qid;
      qidLink.title = "Open on Wikidata";
      qidLink.addEventListener("click", (event) => event.stopPropagation());
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

function markNotInWikidata() {
  if (state.mentionId) applyLink("mention", { nil: true });
  else applyLink("speaker", { nil: true });
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
    copyAuxIfEmpty(quote.speaker, row);
  } else {
    const mention = (quote.mentions || []).find((m) => m.id === state.mentionId);
    if (!mention) return;
    Object.assign(mention, patch);
    copyAuxIfEmpty(mention, row);
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
  $("btn-set-intro").addEventListener("click", () => setAuxFromSelection("intro_phrase"));
  $("btn-set-first").addEventListener("click", () => setAuxFromSelection("first_span"));
  $("btn-set-quotative").addEventListener("click", setQuotativeFromSelection);
  $("btn-add-mention").addEventListener("click", addMentionFromSelection);
  $("btn-set-bounds").addEventListener("click", setQuoteBoundsFromSelection);
  $("btn-add-quote").addEventListener("click", addQuoteFromSelection);
  $("btn-merge-next").addEventListener("click", () => mergeQuoteWithNeighbor("next"));
  $("btn-merge-prev").addEventListener("click", () => mergeQuoteWithNeighbor("prev"));
  $("btn-unmerge").addEventListener("click", unmergeCurrentQuote);
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
  document.addEventListener("selectionchange", () => {
    if (selectionOffsets()) updateSelectionBar();
  });
  article.addEventListener("click", (event) => {
    const span = event.target.closest("span[data-start]");
    if (!span || !state.article) return;
    if (window.getSelection() && !window.getSelection().isCollapsed) return;
    const offset = Number(span.dataset.start);
    const quotes = state.article.quotes || [];
    const hits = quotes.filter((q) => offsetInQuote(q, offset));
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
    if (event.key === "-") rejectQuote();
    if (event.key === "q") setQuotativeFromSelection();
    if (event.key === "i") noQuotative();
    if (event.key === "s") setSpeakerFromSelection();
    if (event.key === "p") setAuxFromSelection("intro_phrase");
    if (event.key === "f") setAuxFromSelection("first_span");
    if (event.key === "m") addMentionFromSelection();
    if (event.key === "c") cannotIdentifySpeaker();
    if (event.key === "n") markNotInWikidata();
    if (event.key === "g" || event.key === "G") mergeQuoteWithNeighbor(event.shiftKey ? "prev" : "next");
    if (event.key === "u" || event.key === "U") unmergeCurrentQuote();
    if (!event.altKey && !event.ctrlKey && !event.metaKey) {
      if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
        event.preventDefault();
        state.quoteIndex = Math.max(0, state.quoteIndex - 1);
        renderAll();
      }
      if (event.key === "ArrowRight" || event.key === "ArrowDown") {
        event.preventDefault();
        const n = state.article?.quotes?.length || 1;
        state.quoteIndex = Math.min(n - 1, state.quoteIndex + 1);
        renderAll();
      }
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
