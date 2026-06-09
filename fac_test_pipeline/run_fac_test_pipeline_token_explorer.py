import argparse
import csv
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import tqdm


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAE_PRETRAIN_DIR = os.path.join(ROOT_DIR, "sae_pretrain")
if SAE_PRETRAIN_DIR not in sys.path:
    sys.path.insert(0, SAE_PRETRAIN_DIR)


TOP_K = 20
TOP_K_PER_TOKEN = 20
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_BASELINE_TSV = os.path.join(_SCRIPT_DIR, "feature_activation_baseline_agg.tsv")

try:
    import torch as tc
    from autoencoder import TopKSAE  # noqa: E402
    from generator_uni import UnifiedGenerator  # noqa: E402
    from llm_surgery_uni import mount_function, switch_mode  # noqa: E402
    tc.manual_seed(42)
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared utilities (mirrored from run_fac_test_pipeline_token_matrix.py)
# ---------------------------------------------------------------------------

def _default_cache_dir() -> str:
    return os.environ.get("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface"))


def load_baseline_p95(tsv_path: str) -> Dict[int, float]:
    baseline: Dict[int, float] = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                fid = int(row["feature_id"])
                p95 = float(row["p95"])
                baseline[fid] = p95
            except (KeyError, ValueError):
                continue
    return baseline


def _parse_word_spans(words_text: str) -> List[str]:
    parts = re.split(r"(?:^|\n)Span \d+:\s*", words_text)
    return [p.strip() for p in parts if p.strip()]


def load_feature_descriptions(tsv_path: str, feature_ids: set) -> Dict[int, List[str]]:
    descriptions: Dict[int, List[str]] = {fid: [] for fid in feature_ids}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                fid = int(row["FeatureID"])
            except (KeyError, ValueError):
                continue
            if fid not in feature_ids:
                continue
            words = row.get("Words", "").strip()
            if words:
                descriptions[fid] = _parse_word_spans(words)
    return descriptions


def resolve_sae_checkpoint(local_path: Optional[str]) -> str:
    if local_path:
        local_path = os.path.abspath(local_path)
        if os.path.exists(local_path):
            return local_path
        raise FileNotFoundError(f"Local SAE checkpoint not found: {local_path}")
    raise ValueError("--sae-ckpt-path is required. Online SAE download is disabled.")


def load_prompts(prompts_json: str) -> List[str]:
    with open(prompts_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    prompts: List[str] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                prompts.append(item)
            elif isinstance(item, dict):
                prompt = item.get("prompt")
                if isinstance(prompt, str):
                    prompts.append(prompt)
    elif isinstance(payload, dict):
        items = payload.get("prompts", [])
        if isinstance(items, list):
            for item in items:
                if isinstance(item, str):
                    prompts.append(item)
                elif isinstance(item, dict):
                    prompt = item.get("prompt")
                    if isinstance(prompt, str):
                        prompts.append(prompt)
    if not prompts:
        raise ValueError("No prompts found in prompts JSON.")
    return prompts


def _encode_prompt_tokens(prompt: str, model) -> Tuple:
    messages = model.build_messages(user_text=prompt)
    enc = model._tokenizer.apply_chat_template(  # noqa: SLF001
        messages, tokenize=True, add_generation_prompt=False, return_tensors="pt",
    )
    if isinstance(enc, dict):
        input_ids = enc["input_ids"]
    else:
        input_ids = enc
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(model._device)
    token_ids = input_ids[0].tolist()
    tokens = model._tokenizer.convert_ids_to_tokens(token_ids)  # noqa: SLF001
    return input_ids, token_ids, tokens


def _special_token_id_set(tokenizer) -> set:
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    special_ids.update(range(128000, 128256))
    return special_ids


def _is_newline_only_token(token: str) -> bool:
    stripped = token.replace("Ċ", "").replace("\n", "")
    return len(stripped) == 0 and len(token) > 0


def _find_user_content_start(token_ids: Sequence[int], tokens: Sequence[str], tokenizer) -> int:
    marker_tokens = ("<|start_header_id|>", "user", "<|end_header_id|>")
    marker_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in marker_tokens]
    user_header_end = -1
    if all(isinstance(tid, int) and tid >= 0 for tid in marker_ids):
        marker_len = len(marker_ids)
        for idx in range(0, len(token_ids) - marker_len + 1):
            if list(token_ids[idx: idx + marker_len]) == marker_ids:
                user_header_end = idx + marker_len
                break
    if user_header_end < 0:
        marker_len = len(marker_tokens)
        for idx in range(0, len(tokens) - marker_len + 1):
            if tuple(tokens[idx: idx + marker_len]) == marker_tokens:
                user_header_end = idx + marker_len
                break
    if user_header_end < 0:
        return 0
    content_start = user_header_end
    while content_start < len(tokens) and _is_newline_only_token(tokens[content_start]):
        content_start += 1
    return content_start


def _estimate_content_start_from_tokens(tokens: List[str]) -> int:
    """Estimate content start from token strings alone (no tokenizer needed).

    Used as fallback when reading pre-computed JSONL records that lack
    a stored content_start field.
    """
    for i in range(len(tokens) - 2):
        if tokens[i] == "<|start_header_id|>" and tokens[i + 1] == "user":
            for j in range(i + 2, min(i + 6, len(tokens))):
                if tokens[j] == "<|end_header_id|>":
                    k = j + 1
                    while k < len(tokens) and _is_newline_only_token(tokens[k]):
                        k += 1
                    return k
    return 0


# ---------------------------------------------------------------------------
# Collector + extraction (adds content_start to the record)
# ---------------------------------------------------------------------------

class Collector:
    def __init__(self, layer: int):
        self.layer = layer
        if _TORCH_AVAILABLE:
            switch_mode(self, "monitor")
        self.early_stop = False
        self.cache = None

    def monitor(self, x):
        self.cache = x


def extract_token_feature_matrix_for_prompt(
    prompt: str,
    model,
    collector: "Collector",
    sae,
    baseline_p95: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    collector.cache = None
    _, token_ids, tokens = _encode_prompt_tokens(prompt, model)

    try:
        model.get_activates(prompt)
    except RuntimeError:
        pass

    if collector.cache is None:
        raise RuntimeError("Collector cache is empty. Hook may not be mounted correctly.")

    hidden = collector.cache.to(tc.float32)
    if hidden.dim() != 3:
        raise RuntimeError(f"Expected [batch, seq, hidden], got {tuple(hidden.shape)}")

    hidden_seq = hidden[0]
    if hidden_seq.shape[0] != len(tokens):
        seq_len = min(hidden_seq.shape[0], len(tokens))
        hidden_seq = hidden_seq[:seq_len]
        token_ids = token_ids[:seq_len]
        tokens = tokens[:seq_len]

    sparse_features = sae.encode(hidden_seq).detach().cpu()
    top_k = int(getattr(sae, "topk", TOP_K))
    special_ids = _special_token_id_set(model._tokenizer)  # noqa: SLF001
    content_start = _find_user_content_start(token_ids, tokens, model._tokenizer)  # noqa: SLF001
    token_mask = tc.tensor(
        [idx >= content_start and tid not in special_ids for idx, tid in enumerate(token_ids)],
        device=sparse_features.device,
    )
    sparse_features = sparse_features * token_mask.to(sparse_features.dtype).unsqueeze(-1)
    content_features = sparse_features[token_mask]
    if content_features.numel() == 0:
        raise ValueError("No content tokens found after masking.")

    rel_feature_tensor = tc.unique((content_features > 0).nonzero(as_tuple=False)[:, 1])
    rel_feature_tensor, _ = tc.sort(rel_feature_tensor)

    excluded_features: List[int] = []
    if baseline_p95 is not None:
        fids = rel_feature_tensor.tolist()
        keep_mask = tc.tensor(
            [fid in baseline_p95 for fid in fids], dtype=tc.bool, device=rel_feature_tensor.device
        )
        excluded_features = [int(fid) for fid in fids if fid not in baseline_p95]
        rel_feature_tensor = rel_feature_tensor[keep_mask]

    if rel_feature_tensor.numel() > 0:
        feature_sub = sparse_features[:, rel_feature_tensor]
        if baseline_p95 is not None:
            p95_vals = tc.tensor(
                [baseline_p95[int(fid)] for fid in rel_feature_tensor.tolist()],
                dtype=tc.float32, device=feature_sub.device,
            ).clamp(min=1e-9)
            feature_sub = feature_sub / p95_vals.unsqueeze(0)
    else:
        feature_sub = tc.zeros(
            sparse_features.shape[0], 0, dtype=tc.float32, device=sparse_features.device
        )

    return {
        "prompt": prompt,
        "token_ids": token_ids,
        "tokens": list(tokens),
        "layer": int(collector.layer),
        "encoding_mode": "dense_token_matrix",
        "top_k": int(top_k),
        "content_start": content_start,
        "rel_features": [int(i) for i in rel_feature_tensor.tolist()],
        "matrix_shape": [int(sparse_features.shape[0]), int(rel_feature_tensor.numel())],
        "feature_matrix": feature_sub.tolist(),
        "excluded_features": excluded_features,
        "normalized": baseline_p95 is not None,
    }


# ---------------------------------------------------------------------------
# HTML token-explorer generation
# ---------------------------------------------------------------------------

_HTML_TOKEN_TEMPLATE = """\
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #ffffff; --bg-secondary: #f5f4ef;
    --text-primary: #1a1a1a; --text-secondary: #6b6b6b; --text-tertiary: #9a9a9a;
    --border: rgba(0,0,0,0.12); --border-strong: rgba(0,0,0,0.25);
    --accent: #2D6A4F;
    --accent2: #6B46C1;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #1e1e1e; --bg-secondary: #2a2a2a;
      --text-primary: #f0f0f0; --text-secondary: #b0b0b0; --text-tertiary: #808080;
      --border: rgba(255,255,255,0.15); --border-strong: rgba(255,255,255,0.3);
    }
  }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    background: var(--bg); color: var(--text-primary); margin: 0; padding: 24px; line-height: 1.6; }
  .container { max-width: 1200px; margin: 0 auto; }
  h1 { font-size: 22px; font-weight: 500; margin: 0 0 6px; }
  .subtitle { color: var(--text-secondary); margin: 0 0 20px; font-size: 14px; }
  .prompt-box { background: var(--bg-secondary); border-radius: 8px; padding: 12px 16px;
    font-size: 12px; color: var(--text-secondary); margin-bottom: 20px;
    max-height: 100px; overflow-y: auto; white-space: pre-wrap; }
  .stats { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 24px; }
  .stat { background: var(--bg-secondary); border-radius: 8px; padding: 10px 14px; min-width: 90px; }
  .stat-label { font-size: 12px; color: var(--text-secondary); }
  .stat-value { font-size: 20px; font-weight: 500; margin-top: 2px; }

  /* Token Explorer */
  .explorer-header { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }
  .explorer-title { font-size: 16px; font-weight: 500; flex: 1; min-width: 140px; }
  .search-input { padding: 7px 12px; border-radius: 8px; border: 1px solid var(--border-strong);
    background: var(--bg-secondary); color: var(--text-primary); font-size: 13px; outline: none; width: 260px; }
  .search-input:focus { border-color: var(--accent); }
  .tok-count-badge { font-size: 12px; color: var(--text-secondary); white-space: nowrap; }

  .tok-list { border: 0.5px solid var(--border); border-radius: 12px; overflow: hidden; }
  .tok-card { border-bottom: 0.5px solid var(--border); }
  .tok-card:last-child { border-bottom: none; }

  .tok-header-row { display: flex; align-items: center; gap: 10px; padding: 9px 14px;
    cursor: pointer; user-select: none; }
  .tok-header-row:hover { background: var(--bg-secondary); }
  .tok-toggle { width: 14px; color: var(--text-tertiary); font-size: 11px; flex-shrink: 0; }
  .tok-idx { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 11px;
    color: var(--text-tertiary); min-width: 34px; flex-shrink: 0; }
  .tok-display { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 13px;
    color: var(--accent); font-weight: 500; min-width: 120px; max-width: 180px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex-shrink: 0; }
  .tok-bar-wrap { width: 60px; height: 5px; background: var(--bg-secondary);
    border-radius: 3px; overflow: hidden; flex-shrink: 0; }
  .tok-bar { height: 100%; background: var(--accent); border-radius: 3px; }
  .tok-max-val { font-size: 11px; min-width: 38px; text-align: right;
    font-variant-numeric: tabular-nums; color: var(--text-secondary); flex-shrink: 0; }
  .tok-feat-count { font-size: 12px; color: var(--text-secondary); flex-shrink: 0; }

  /* Expanded token detail */
  .tok-detail { display: none; padding: 10px 14px 14px 38px; }
  .tok-detail.open { display: block; }
  .tok-no-feat { font-size: 12px; color: var(--text-tertiary); font-style: italic; padding: 4px 0; }

  /* Feature entry within expanded token */
  .feat-entry { margin-bottom: 10px; border: 0.5px solid var(--border); border-radius: 8px; overflow: hidden; }
  .feat-entry-hdr { display: flex; align-items: center; gap: 8px; padding: 7px 10px;
    background: var(--bg-secondary); }
  .feat-id { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px;
    color: var(--accent2); font-weight: 500; min-width: 68px; flex-shrink: 0; }
  .feat-bar-wrap2 { width: 60px; height: 4px; background: rgba(107,70,193,0.15);
    border-radius: 2px; overflow: hidden; flex-shrink: 0; }
  .feat-bar2 { height: 100%; background: var(--accent2); border-radius: 2px; }
  .feat-act-val { font-size: 11px; min-width: 38px; text-align: right;
    font-variant-numeric: tabular-nums; color: var(--text-secondary); flex-shrink: 0; }
  .feat-desc-preview { font-size: 12px; color: var(--text-secondary);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; min-width: 0; }

  .feat-body { padding: 8px 10px 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 700px) { .feat-body { grid-template-columns: 1fr; } }
  .feat-section h4 { font-size: 11px; font-weight: 600; color: var(--text-tertiary);
    text-transform: uppercase; letter-spacing: 0.06em; margin: 0 0 7px; }
  .span-list { display: flex; flex-direction: column; gap: 4px; max-height: 180px; overflow-y: auto; }
  .span-item { font-size: 12px; background: var(--bg-secondary);
    border-radius: 6px; padding: 5px 9px; line-height: 1.5; }
  .no-data { font-size: 12px; color: var(--text-tertiary); font-style: italic; padding: 2px 0; }

  /* Cross-token activation rows */
  .cross-tok-row { display: flex; align-items: center; gap: 8px; font-size: 12px; margin-bottom: 3px; }
  .cross-tok-text { font-family: ui-monospace, "SF Mono", Menlo, monospace;
    min-width: 110px; max-width: 110px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .cross-tok-text.is-current { color: var(--accent); font-weight: 600; }
  .cross-bar-wrap { flex: 1; height: 4px; background: var(--bg-secondary); border-radius: 2px; }
  .cross-bar { height: 100%; background: var(--accent2); border-radius: 2px; }
  .cross-tok-val { min-width: 38px; text-align: right; color: var(--text-secondary);
    font-variant-numeric: tabular-nums; font-size: 11px; }
  .current-marker { font-size: 10px; color: var(--accent); margin-left: 2px; }

  .empty-state { padding: 32px; text-align: center; color: var(--text-tertiary); font-size: 13px; }
</style>
</head>
<body>
<div class="container">
  <h1>__TITLE__</h1>
  <p class="subtitle">Top-__TOPK_PER_TOKEN__ Features je Token &middot; Layer __LAYER____NORM_NOTE__</p>
  <div class="prompt-box">__PROMPT_HTML__</div>

  <div class="stats">
    <div class="stat"><div class="stat-label">Tokens gesamt</div><div class="stat-value">__NTOK__</div></div>
    <div class="stat"><div class="stat-label">Content-Tokens</div><div class="stat-value">__N_CONTENT__</div></div>
    <div class="stat"><div class="stat-label">Features</div><div class="stat-value">__NFEAT__</div></div>
    <div class="stat"><div class="stat-label">Layer</div><div class="stat-value">__LAYER__</div></div>
    <div class="stat"><div class="stat-label">Top-k</div><div class="stat-value">__TOPK_PER_TOKEN__</div></div>
    <div class="stat"><div class="stat-label">Max-Aktivierung</div><div class="stat-value">__MAX2__</div></div>
    <div class="stat"><div class="stat-label">Sparsity</div><div class="stat-value">__SPARSE__%</div></div>
  </div>

  <div class="explorer-header">
    <span class="explorer-title">Token Explorer</span>
    <input class="search-input" id="tokSearch" type="text"
      placeholder="Token-Text filtern&hellip;" />
    <span class="tok-count-badge" id="tokCount"></span>
  </div>
  <div class="tok-list" id="tokList"></div>
</div>

<script>
const TOKENS = __TOKENS__;
const CONTENT_START = __CONTENT_START__;
const REL_FEATURES = __FEATS__;
const MATRIX = __MATRIX__;
const MAX_VAL = __MAXVAL__;
const FEATURE_DESCRIPTIONS = __FEAT_DESCS__;
const TOP_K_PER_TOKEN = __TOPK_PER_TOKEN__;

const displayTok = t => t
  .replace(/Ġ/g, '▁')
  .replace(/Ċ/g, '⏎')
  .replace(/âĢĻ/g, "'")
  .replace(/âĢĽ/g, '"')
  .replace(/âĢļ/g, '"');

function escH(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// Build per-content-token stats once (only tokens from CONTENT_START onward).
const tokenStats = [];
for (let i = CONTENT_START; i < TOKENS.length; i++) {
  const row = MATRIX[i] || [];
  const feats = [];
  for (let j = 0; j < REL_FEATURES.length; j++) {
    const v = row[j] || 0;
    if (v > 0) feats.push({fid: REL_FEATURES[j], j: j, v: v});
  }
  feats.sort((a, b) => b.v - a.v);
  tokenStats.push({
    i: i,
    tok: TOKENS[i],
    topFeats: feats.slice(0, TOP_K_PER_TOKEN),
    totalActive: feats.length,
    maxVal: feats.length > 0 ? feats[0].v : 0
  });
}

// Cache cross-token activations per feature column index (computed lazily).
const crossCache = {};
function getFeatureActivations(j) {
  if (crossCache[j] !== undefined) return crossCache[j];
  const acts = [];
  for (let i = 0; i < TOKENS.length; i++) {
    const v = (MATRIX[i] || [])[j] || 0;
    if (v > 0) acts.push({i: i, tok: TOKENS[i], v: v});
  }
  acts.sort((a, b) => b.v - a.v);
  crossCache[j] = acts.slice(0, 15);
  return crossCache[j];
}

let currentFilter = '';
const expanded = new Set();

function tokMatches(ts, q) {
  if (!q) return true;
  const ql = q.toLowerCase();
  return displayTok(ts.tok).toLowerCase().includes(ql)
    || ts.tok.toLowerCase().includes(ql)
    || String(ts.i).includes(ql);
}

function buildFeatEntry(currentTokIdx, feat) {
  const spans = FEATURE_DESCRIPTIONS[String(feat.fid)] || [];
  const preview = spans.length > 0 ? spans[0].substring(0, 80) : '';
  const pct = MAX_VAL > 0 ? (feat.v / MAX_VAL * 100).toFixed(1) : 0;

  const previewHtml = preview
    ? escH(preview) + (spans[0] && spans[0].length > 80 ? '&hellip;' : '')
    : '<span style="font-style:italic;color:var(--text-tertiary);">keine Beschreibung</span>';

  // Description spans (up to 8)
  const spansSlice = spans.slice(0, 8);
  const descHtml = spansSlice.length > 0
    ? spansSlice.map(s => '<div class="span-item">' + escH(s) + '</div>').join('')
    : '<div class="no-data">Keine Beschreibungen verfügbar.</div>';
  const moreDesc = spans.length > 8
    ? '<div class="no-data" style="margin-top:4px;">' + (spans.length - 8) + ' weitere&hellip;</div>'
    : '';

  // Cross-token activations for this feature
  const crossActs = getFeatureActivations(feat.j);
  const crossMax = crossActs.length > 0 ? crossActs[0].v : 1;
  const crossHtml = crossActs.length > 0
    ? crossActs.map(ct => {
        const cp = (ct.v / crossMax * 100).toFixed(0);
        const isCur = ct.i === currentTokIdx;
        return '<div class="cross-tok-row">' +
          '<span class="cross-tok-text' + (isCur ? ' is-current' : '') +
            '" title="Index ' + ct.i + ': ' + escH(ct.tok) + '">' +
            '[' + ct.i + '] ' + escH(displayTok(ct.tok)) +
          '</span>' +
          '<div class="cross-bar-wrap"><div class="cross-bar" style="width:' + cp + '%;"></div></div>' +
          '<span class="cross-tok-val">' + ct.v.toFixed(3) + '</span>' +
          (isCur ? '<span class="current-marker">&#9654;</span>' : '') +
          '</div>';
      }).join('')
    : '<div class="no-data">Keine Aktivierungen.</div>';

  return '<div class="feat-entry">' +
    '<div class="feat-entry-hdr">' +
      '<span class="feat-id">#' + feat.fid + '</span>' +
      '<div class="feat-bar-wrap2"><div class="feat-bar2" style="width:' + pct + '%;"></div></div>' +
      '<span class="feat-act-val">' + feat.v.toFixed(3) + '</span>' +
      '<span class="feat-desc-preview">' + previewHtml + '</span>' +
    '</div>' +
    '<div class="feat-body">' +
      '<div class="feat-section">' +
        '<h4>Beschreibungen (' + spans.length + ')</h4>' +
        '<div class="span-list">' + descHtml + moreDesc + '</div>' +
      '</div>' +
      '<div class="feat-section">' +
        '<h4>Sequenz-Aktivierungen (' + crossActs.length + ')</h4>' +
        crossHtml +
      '</div>' +
    '</div>' +
    '</div>';
}

function renderList() {
  const filtered = tokenStats.filter(ts => tokMatches(ts, currentFilter));
  document.getElementById('tokCount').textContent =
    filtered.length + ' / ' + tokenStats.length + ' Tokens';

  const list = document.getElementById('tokList');
  list.innerHTML = '';

  if (filtered.length === 0) {
    list.innerHTML = '<div class="empty-state">Keine Tokens gefunden.</div>';
    return;
  }

  filtered.forEach(ts => {
    const isOpen = expanded.has(ts.i);
    const pct = MAX_VAL > 0 ? (ts.maxVal / MAX_VAL * 100).toFixed(1) : 0;

    const card = document.createElement('div');
    card.className = 'tok-card';

    const hdr = document.createElement('div');
    hdr.className = 'tok-header-row';
    hdr.innerHTML =
      '<span class="tok-toggle">' + (isOpen ? '&#9660;' : '&#9654;') + '</span>' +
      '<span class="tok-idx">[' + ts.i + ']</span>' +
      '<span class="tok-display" title="' + escH(ts.tok) + '">' + escH(displayTok(ts.tok)) + '</span>' +
      '<div class="tok-bar-wrap"><div class="tok-bar" style="width:' + pct + '%;"></div></div>' +
      '<span class="tok-max-val">' + ts.maxVal.toFixed(2) + '</span>' +
      '<span class="tok-feat-count">' + ts.totalActive + ' Features</span>';

    const detail = document.createElement('div');
    detail.className = 'tok-detail' + (isOpen ? ' open' : '');

    if (isOpen) {
      if (ts.topFeats.length === 0) {
        detail.innerHTML = '<div class="tok-no-feat">Keine aktiven Features für diesen Token.</div>';
      } else {
        detail.innerHTML = ts.topFeats.map(f => buildFeatEntry(ts.i, f)).join('');
      }
    }

    hdr.addEventListener('click', () => {
      if (expanded.has(ts.i)) expanded.delete(ts.i);
      else expanded.add(ts.i);
      renderList();
    });

    card.appendChild(hdr);
    card.appendChild(detail);
    list.appendChild(card);
  });
}

document.getElementById('tokSearch').addEventListener('input', e => {
  currentFilter = e.target.value.trim();
  renderList();
});

renderList();
</script>
</body>
</html>
"""


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def generate_token_explorer_html(
    record: Dict[str, Any],
    output_path: str,
    title: Optional[str] = None,
    feature_descriptions: Optional[Dict[int, List[str]]] = None,
) -> None:
    """Render a token-centric interactive HTML explorer for a single record."""
    tokens: List[str] = record["tokens"]
    rel_feats: List[int] = record["rel_features"]
    matrix: List[List[float]] = record["feature_matrix"]
    layer: int = int(record.get("layer", 0))
    prompt: str = record.get("prompt", "")
    top_k: int = int(record.get("top_k", TOP_K))
    is_normalized: bool = bool(record.get("normalized", False))

    # content_start: prefer stored value, fall back to estimation from token strings.
    content_start: int = record.get("content_start") or _estimate_content_start_from_tokens(tokens)

    num_tok = len(tokens)
    num_feat = len(rel_feats)
    n_content = num_tok - content_start

    max_val = 0.0
    total_cells = num_tok * max(num_feat, 1)
    zero_cells = 0
    for row in matrix:
        for v in row:
            if v == 0:
                zero_cells += 1
            elif v > max_val:
                max_val = float(v)
    sparsity_pct = round((zero_cells / total_cells) * 100) if total_cells else 0

    norm_note = " · normalisiert (&divide; p95 Baseline)" if is_normalized else ""
    title = title or f"SAE Token Explorer · Layer {layer}"

    feat_descs_json: Dict[str, List[str]] = {}
    if feature_descriptions is not None:
        for fid in rel_feats:
            feat_descs_json[str(fid)] = feature_descriptions.get(fid, [])
    else:
        for fid in rel_feats:
            feat_descs_json[str(fid)] = []

    rounded = [[round(float(v), 3) for v in row] for row in matrix]
    prompt_preview = prompt if len(prompt) <= 1200 else prompt[:1200] + "…"

    replacements = {
        "__TITLE__": _html_escape(title),
        "__NTOK__": str(num_tok),
        "__N_CONTENT__": str(n_content),
        "__NFEAT__": str(num_feat),
        "__LAYER__": str(layer),
        "__MAX2__": f"{max_val:.2f}",
        "__SPARSE__": str(sparsity_pct),
        "__PROMPT_HTML__": _html_escape(prompt_preview),
        "__TOKENS__": json.dumps(tokens, ensure_ascii=False),
        "__CONTENT_START__": str(content_start),
        "__FEATS__": json.dumps(rel_feats),
        "__MATRIX__": json.dumps(rounded),
        "__MAXVAL__": json.dumps(max_val if max_val > 0 else 1.0),
        "__TOPK_PER_TOKEN__": str(TOP_K_PER_TOKEN),
        "__NORM_NOTE__": norm_note,
        "__FEAT_DESCS__": json.dumps(feat_descs_json, ensure_ascii=False),
    }

    html = _HTML_TOKEN_TEMPLATE
    for needle, value in replacements.items():
        html = html.replace(needle, value)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def _html_path_for_record(html_dir: str, base_name: str, idx: int) -> str:
    return os.path.join(html_dir, f"{base_name}_{idx:04d}.html")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Token-centric SAE feature explorer: search by token, see top-k SAE features "
            "with descriptions and cross-token activations. Parallel to "
            "run_fac_test_pipeline_token_matrix.py — same computation, inverted HTML view."
        )
    )

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--device-id", type=str, default="0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument(
        "--model-name", type=str, default="",
        help="Local path to model directory. Required unless --input-jsonl is used.",
    )
    parser.add_argument("--layer", type=int, default=16, help="1-based decoder layer index.")
    parser.add_argument("--sae-ckpt-path", type=str, default="", help="Local path to SAE checkpoint .pth")

    parser.add_argument(
        "--input-jsonl", type=str, default="",
        help=(
            "Path to a pre-computed JSONL file (output of run_fac_test_pipeline_token_matrix.py). "
            "If provided, model loading is skipped entirely."
        ),
    )
    parser.add_argument(
        "--prompts-json", type=str, default="",
        help="Path to prompts JSON. Required when --input-jsonl is not used.",
    )
    parser.add_argument("--output-jsonl", type=str, default="", help="Optional: write records to JSONL.")
    parser.add_argument("--max-prompts", type=int, default=0, help="0 = all")
    parser.add_argument("--hf-cache-dir", type=str, default=_default_cache_dir())

    parser.add_argument(
        "--html-dir", type=str, default="",
        help="Output directory for HTML files. Defaults to the directory of --output-jsonl or --input-jsonl.",
    )
    parser.add_argument(
        "--baseline-tsv", type=str, default=_DEFAULT_BASELINE_TSV,
        help="Path to feature_activation_baseline_agg.tsv for p95 normalization.",
    )
    parser.add_argument(
        "--feature-descriptions-tsv", type=str, default="",
        help="Path to a TSV with columns FeatureID and Words (feature descriptions).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Path A: read pre-computed JSONL records (no model/SAE needed)
    # ------------------------------------------------------------------
    if args.input_jsonl:
        if not os.path.isfile(args.input_jsonl):
            raise FileNotFoundError(f"Input JSONL not found: {args.input_jsonl}")
        with open(args.input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        print(f"Loaded {len(records)} records from {args.input_jsonl}")
        if args.max_prompts > 0:
            records = records[: args.max_prompts]

    # ------------------------------------------------------------------
    # Path B: run the full model + SAE pipeline
    # ------------------------------------------------------------------
    else:
        if not args.model_name:
            raise ValueError("--model-name is required when --input-jsonl is not provided.")
        if not args.prompts_json:
            raise ValueError("--prompts-json is required when --input-jsonl is not provided.")
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "torch / autoencoder / generator_uni are not importable. "
                "Use --input-jsonl to read pre-computed records instead."
            )

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_CACHE"] = args.hf_cache_dir
        os.makedirs(args.hf_cache_dir, exist_ok=True)

        if args.device == "cuda":
            os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

        model_path = os.path.abspath(args.model_name)
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Local model directory not found: {model_path}")

        prompts = load_prompts(args.prompts_json)
        if args.max_prompts > 0:
            prompts = prompts[: args.max_prompts]

        sae_ckpt = resolve_sae_checkpoint(local_path=args.sae_ckpt_path or None)

        model = UnifiedGenerator(
            model_path, device=args.device, dtype=args.dtype,
            cache_dir=args.hf_cache_dir, local_files_only=True, strict_local_paths=True,
        )
        collector = Collector(args.layer)
        mount_function(model._model, "llama", args.layer, collector)
        collector.early_stop = True

        sae = TopKSAE.from_disk(sae_ckpt, device=args.device)
        sae.topk = TOP_K
        sae.eval()

        baseline_p95: Optional[Dict[int, float]] = None
        if args.baseline_tsv:
            if not os.path.isfile(args.baseline_tsv):
                raise FileNotFoundError(f"Baseline TSV not found: {args.baseline_tsv}")
            baseline_p95 = load_baseline_p95(args.baseline_tsv)
            print(f"Loaded p95 baseline for {len(baseline_p95)} features from {args.baseline_tsv}")

        with tc.no_grad():
            for idx, prompt in enumerate(tqdm.tqdm(prompts, desc="Processing prompts")):
                record = extract_token_feature_matrix_for_prompt(
                    prompt, model, collector, sae, baseline_p95=baseline_p95
                )
                record["index"] = idx
                record["num_tokens"] = len(record["tokens"])
                record["num_features"] = len(record["rel_features"])
                records.append(record)

        if args.output_jsonl:
            out_dir = os.path.dirname(os.path.abspath(args.output_jsonl))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.output_jsonl, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"Wrote {len(records)} records to {args.output_jsonl}")

    # ------------------------------------------------------------------
    # HTML generation (always performed)
    # ------------------------------------------------------------------
    html_dir = args.html_dir
    if not html_dir:
        if args.output_jsonl:
            html_dir = os.path.dirname(os.path.abspath(args.output_jsonl))
        elif args.input_jsonl:
            html_dir = os.path.dirname(os.path.abspath(args.input_jsonl))
        else:
            html_dir = os.getcwd()
    os.makedirs(html_dir, exist_ok=True)

    # Derive a base name for the HTML files.
    if args.input_jsonl:
        base_name = os.path.splitext(os.path.basename(args.input_jsonl))[0] + "_token_explorer"
    elif args.output_jsonl:
        base_name = os.path.splitext(os.path.basename(args.output_jsonl))[0] + "_token_explorer"
    else:
        base_name = "token_explorer"

    for idx, record in enumerate(tqdm.tqdm(records, desc="Generating HTML")):
        feature_descriptions: Optional[Dict[int, List[str]]] = None
        if args.feature_descriptions_tsv:
            if not os.path.isfile(args.feature_descriptions_tsv):
                raise FileNotFoundError(
                    f"Feature descriptions TSV not found: {args.feature_descriptions_tsv}"
                )
            feature_descriptions = load_feature_descriptions(
                args.feature_descriptions_tsv,
                set(record["rel_features"]),
            )

        record_idx = record.get("index", idx)
        html_path = _html_path_for_record(html_dir, base_name, record_idx)
        title = (
            f"SAE Token Explorer · Prompt {record_idx} "
            f"· Layer {record.get('layer', '?')}"
        )
        generate_token_explorer_html(
            record, html_path, title=title, feature_descriptions=feature_descriptions
        )

    print(f"Wrote {len(records)} HTML token explorers to {html_dir}")


if __name__ == "__main__":
    main()
