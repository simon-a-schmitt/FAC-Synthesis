import argparse
import csv
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch as tc
import tqdm


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAE_PRETRAIN_DIR = os.path.join(ROOT_DIR, "sae_pretrain")
if SAE_PRETRAIN_DIR not in sys.path:
    sys.path.insert(0, SAE_PRETRAIN_DIR)

from autoencoder import TopKSAE  # noqa: E402
from generator_uni import UnifiedGenerator  # noqa: E402
from llm_surgery_uni import mount_function, switch_mode  # noqa: E402


tc.manual_seed(42)


TOP_K = 20
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_BASELINE_TSV = os.path.join(_SCRIPT_DIR, "feature_activation_baseline_agg.tsv")


class Collector:
    def __init__(self, layer: int):
        self.layer = layer
        switch_mode(self, "monitor")
        self.early_stop = False
        self.cache = None

    def monitor(self, x):
        self.cache = x


def _default_cache_dir() -> str:
    return os.environ.get("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface"))


def load_baseline_p95(tsv_path: str) -> Dict[int, float]:
    """Load feature_id -> p95 normalization values from the baseline TSV."""
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
    """Split 'Span 1: text\\nSpan 2: text\\n...' into individual span strings."""
    parts = re.split(r"(?:^|\n)Span \d+:\s*", words_text)
    return [p.strip() for p in parts if p.strip()]


def load_feature_descriptions(tsv_path: str, feature_ids: set) -> Dict[int, List[str]]:
    """Load text spans for the given feature IDs from a (large) TSV.

    Expects columns FeatureID (string int) and Words (all spans concatenated as
    'Span 1: text\\nSpan 2: text\\n...').  Only rows whose FeatureID is in
    feature_ids are loaded — safe for very large files.
    Returns {fid: [span1, span2, ...]} for every fid in feature_ids.
    """
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

    raise ValueError("--sae-ckpt-path is required. Online SAE download is disabled in this pipeline.")


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

    if len(prompts) == 0:
        raise ValueError("No prompts found in prompts JSON. Expected list[str] or list[{\"prompt\": ...}].")
    return prompts


def _encode_prompt_tokens(prompt: str, model: UnifiedGenerator) -> Tuple[tc.Tensor, List[int], List[str]]:
    messages = model.build_messages(user_text=prompt)
    enc = model._tokenizer.apply_chat_template(  # noqa: SLF001
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
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


def _special_token_id_set(tokenizer) -> set[int]:
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    special_ids.update(range(128000, 128256))
    return special_ids


def _is_newline_only_token(token: str) -> bool:
    stripped = token.replace("Ċ", "").replace("\n", "")
    return len(stripped) == 0 and len(token) > 0


def _find_user_content_start(token_ids: Sequence[int], tokens: Sequence[str], tokenizer) -> int:
    # Chat template user header: <|start_header_id|> user <|end_header_id|>
    marker_tokens = ("<|start_header_id|>", "user", "<|end_header_id|>")
    marker_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in marker_tokens]

    user_header_end = -1
    if all(isinstance(tok_id, int) and tok_id >= 0 for tok_id in marker_ids):
        marker_len = len(marker_ids)
        for idx in range(0, len(token_ids) - marker_len + 1):
            if list(token_ids[idx : idx + marker_len]) == marker_ids:
                user_header_end = idx + marker_len
                break

    if user_header_end < 0:
        marker_len = len(marker_tokens)
        for idx in range(0, len(tokens) - marker_len + 1):
            if tuple(tokens[idx : idx + marker_len]) == marker_tokens:
                user_header_end = idx + marker_len
                break

    if user_header_end < 0:
        return 0

    content_start = user_header_end
    while content_start < len(tokens) and _is_newline_only_token(tokens[content_start]):
        content_start += 1
    return content_start


def extract_token_feature_matrix_for_prompt(
    prompt: str,
    model: UnifiedGenerator,
    collector: Collector,
    sae: TopKSAE,
    baseline_p95: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    collector.cache = None
    _, token_ids, tokens = _encode_prompt_tokens(prompt, model)

    try:
        model.get_activates(prompt)
    except RuntimeError:
        # Expected when collector.early_stop is true.
        pass

    if collector.cache is None:
        raise RuntimeError("Collector cache is empty. Hook may not be mounted correctly.")

    hidden = collector.cache.to(tc.float32)
    if hidden.dim() != 3:
        raise RuntimeError(f"Expected hidden states with shape [batch, seq, hidden], got {tuple(hidden.shape)}")

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
        [idx >= content_start and token_id not in special_ids for idx, token_id in enumerate(token_ids)],
      device=sparse_features.device,
    )
    sparse_features = sparse_features * token_mask.to(sparse_features.dtype).unsqueeze(-1)
    content_features = sparse_features[token_mask]
    if content_features.numel() == 0:
      raise ValueError("No content tokens found after masking; cannot build a top-k feature union.")

    rel_feature_tensor = tc.unique((content_features > 0).nonzero(as_tuple=False)[:, 1])
    rel_feature_tensor, _ = tc.sort(rel_feature_tensor)

    # Filter features to those present in the baseline; track excluded ones.
    excluded_features: List[int] = []
    if baseline_p95 is not None:
        fids = rel_feature_tensor.tolist()
        keep_mask = tc.tensor(
            [fid in baseline_p95 for fid in fids], dtype=tc.bool, device=rel_feature_tensor.device
        )
        excluded_features = [int(fid) for fid in fids if fid not in baseline_p95]
        rel_feature_tensor = rel_feature_tensor[keep_mask]

    # Build sub-matrix, normalize by p95, and zero out values <= 1.
    if rel_feature_tensor.numel() > 0:
        feature_sub = sparse_features[:, rel_feature_tensor]
        if baseline_p95 is not None:
            p95_vals = tc.tensor(
                [baseline_p95[int(fid)] for fid in rel_feature_tensor.tolist()],
                dtype=tc.float32,
                device=feature_sub.device,
            ).clamp(min=1e-9)
            feature_sub = feature_sub / p95_vals.unsqueeze(0)
            feature_sub = tc.where(feature_sub > 0.6, feature_sub, tc.zeros_like(feature_sub))
            # Drop columns that are entirely zero after thresholding.
            active_col_mask = (feature_sub > 0).any(dim=0)
            feature_sub = feature_sub[:, active_col_mask]
            rel_feature_tensor = rel_feature_tensor[active_col_mask]
    else:
        feature_sub = tc.zeros(
            sparse_features.shape[0], 0, dtype=tc.float32, device=sparse_features.device
        )

    feature_matrix = feature_sub.tolist()

    return {
        "prompt": prompt,
        "token_ids": token_ids,
        "tokens": tokens,
        "layer": int(collector.layer),
        "encoding_mode": "dense_token_matrix",
        "top_k": int(top_k),
        "rel_features": [int(index) for index in rel_feature_tensor.tolist()],
        "matrix_shape": [int(sparse_features.shape[0]), int(rel_feature_tensor.numel())],
        "feature_matrix": feature_matrix,
        "excluded_features": excluded_features,
        "normalized": baseline_p95 is not None,
    }


def write_jsonl(records: List[Dict[str, Any]], output_jsonl: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(output_jsonl))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# HTML feature-explorer generation
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
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
    --accent: #6B46C1;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #1e1e1e; --bg-secondary: #2a2a2a;
      --text-primary: #f0f0f0; --text-secondary: #b0b0b0; --text-tertiary: #808080;
      --border: rgba(255,255,255,0.15); --border-strong: rgba(255,255,255,0.3);
    }
  }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    background: var(--bg); color: var(--text-primary);
    margin: 0; padding: 24px; line-height: 1.6; }
  .container { max-width: 1200px; margin: 0 auto; }
  h1 { font-size: 22px; font-weight: 500; margin: 0 0 6px; }
  .subtitle { color: var(--text-secondary); margin: 0 0 20px; font-size: 14px; }
  .prompt-box { background: var(--bg-secondary); border-radius: 8px;
    padding: 12px 16px; font-size: 12px; color: var(--text-secondary);
    margin-bottom: 20px; max-height: 100px; overflow-y: auto; white-space: pre-wrap; }
  .stats { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 24px; }
  .stat { background: var(--bg-secondary); border-radius: 8px; padding: 10px 14px; min-width: 90px; }
  .stat-label { font-size: 12px; color: var(--text-secondary); }
  .stat-value { font-size: 20px; font-weight: 500; margin-top: 2px; }
  /* Feature Explorer */
  .explorer-header { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }
  .explorer-title { font-size: 16px; font-weight: 500; flex: 1; min-width: 140px; }
  .search-input { padding: 7px 12px; border-radius: 8px; border: 1px solid var(--border-strong);
    background: var(--bg-secondary); color: var(--text-primary); font-size: 13px; outline: none; width: 260px; }
  .search-input:focus { border-color: var(--accent); }
  .feat-count-badge { font-size: 12px; color: var(--text-secondary); white-space: nowrap; }
  .feat-list { border: 0.5px solid var(--border); border-radius: 12px; overflow: hidden; }
  .feat-card { border-bottom: 0.5px solid var(--border); }
  .feat-card:last-child { border-bottom: none; }
  .feat-header-row { display: flex; align-items: center; gap: 10px; padding: 9px 14px;
    cursor: pointer; user-select: none; }
  .feat-header-row:hover { background: var(--bg-secondary); }
  .feat-toggle { width: 14px; color: var(--text-tertiary); font-size: 11px; flex-shrink: 0; }
  .feat-id-badge { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px;
    color: var(--accent); min-width: 70px; flex-shrink: 0; }
  .feat-bar-wrap { width: 72px; height: 5px; background: var(--bg-secondary);
    border-radius: 3px; overflow: hidden; flex-shrink: 0; }
  .feat-bar { height: 100%; background: var(--accent); border-radius: 3px; }
  .feat-max-val { font-size: 11px; min-width: 38px; text-align: right;
    font-variant-numeric: tabular-nums; color: var(--text-secondary); flex-shrink: 0; }
  .feat-preview { font-size: 12px; color: var(--text-secondary);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; min-width: 0; }
  .feat-detail { display: none; padding: 0 14px 14px 38px; }
  .feat-detail.open { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 700px) { .feat-detail.open { grid-template-columns: 1fr; } }
  .detail-section h4 { font-size: 11px; font-weight: 600; color: var(--text-tertiary);
    text-transform: uppercase; letter-spacing: 0.06em; margin: 0 0 8px; }
  .token-row { display: flex; align-items: center; gap: 8px; font-size: 12px; margin-bottom: 3px; }
  .token-text { font-family: ui-monospace, "SF Mono", Menlo, monospace;
    min-width: 110px; max-width: 110px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .token-mini-bar-wrap { flex: 1; height: 4px; background: var(--bg-secondary); border-radius: 2px; }
  .token-mini-bar { height: 100%; background: var(--accent); border-radius: 2px; }
  .token-val { min-width: 38px; text-align: right; color: var(--text-secondary);
    font-variant-numeric: tabular-nums; font-size: 11px; }
  .span-list { display: flex; flex-direction: column; gap: 5px; max-height: 240px; overflow-y: auto; }
  .span-item { font-size: 12px; background: var(--bg-secondary);
    border-radius: 6px; padding: 5px 9px; line-height: 1.5; }
  .no-data { font-size: 12px; color: var(--text-tertiary); font-style: italic; padding: 2px 0; }
  .empty-state { padding: 32px; text-align: center; color: var(--text-tertiary); font-size: 13px; }
  .excluded-note { margin-top: 20px; font-size: 12px; color: var(--text-tertiary); }
</style>
</head>
<body>
<div class="container">
  <h1>__TITLE__</h1>
  <p class="subtitle">Top-__TOPK__ Features je Token &middot; Layer __LAYER____NORM_NOTE__</p>
  <div class="prompt-box">__PROMPT_HTML__</div>

  <div class="stats">
    <div class="stat"><div class="stat-label">Tokens</div><div class="stat-value">__NTOK__</div></div>
    <div class="stat"><div class="stat-label">Features</div><div class="stat-value">__NFEAT__</div></div>
    <div class="stat"><div class="stat-label">Layer</div><div class="stat-value">__LAYER__</div></div>
    <div class="stat"><div class="stat-label">Top-k</div><div class="stat-value">__TOPK__</div></div>
    <div class="stat"><div class="stat-label">Max-Aktivierung</div><div class="stat-value">__MAX2__</div></div>
    <div class="stat"><div class="stat-label">Sparsity</div><div class="stat-value">__SPARSE__%</div></div>
    <div class="stat"><div class="stat-label">Aktive Tokens</div><div class="stat-value">__ACTIVE__ / __NTOK__</div></div>
  </div>

  <div class="explorer-header">
    <span class="explorer-title">Feature Explorer</span>
    <input class="search-input" id="featSearch" type="text"
      placeholder="Feature-ID oder Beschreibung filtern&hellip;" />
    <span class="feat-count-badge" id="featCount"></span>
  </div>
  <div class="feat-list" id="featList"></div>
  __EXCLUDED_NOTE__
</div>

<script>
const TOKENS = __TOKENS__;
const REL_FEATURES = __FEATS__;
const MATRIX = __MATRIX__;
const MAX_VAL = __MAXVAL__;
const FEATURE_DESCRIPTIONS = __FEAT_DESCS__;

const displayTok = t => t
  .replace(/\\u0120/g, '\\u2581')
  .replace(/\\u010a/g, '\\u23ce')
  .replace(/\\u00e2\\u0122\\u013b/g, "'")
  .replace(/\\u00e2\\u0122\\u013d/g, '"')
  .replace(/\\u00e2\\u0122\\u013c/g, '"');

function escH(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// Build per-feature stats once.
const featStats = REL_FEATURES.map((fid, j) => {
  let maxVal = 0;
  const activations = [];
  MATRIX.forEach((row, i) => {
    const v = row[j];
    if (v > 0) {
      activations.push({i: i, tok: TOKENS[i], v: v});
      if (v > maxVal) maxVal = v;
    }
  });
  activations.sort((a, b) => b.v - a.v);
  const spans = FEATURE_DESCRIPTIONS[String(fid)] || [];
  const preview = spans.length > 0 ? spans[0].substring(0, 90) : '';
  return {fid: fid, j: j, maxVal: maxVal, activations: activations, spans: spans, preview: preview};
});
featStats.sort((a, b) => b.maxVal - a.maxVal);

let currentFilter = '';
const expanded = new Set();

function matches(fs, q) {
  if (!q) return true;
  const ql = q.toLowerCase();
  if (String(fs.fid).includes(ql)) return true;
  return fs.spans.some(s => s.toLowerCase().includes(ql));
}

function renderList() {
  const filtered = featStats.filter(fs => matches(fs, currentFilter));
  document.getElementById('featCount').textContent =
    filtered.length + ' / ' + featStats.length + ' Features';

  const list = document.getElementById('featList');
  list.innerHTML = '';

  if (filtered.length === 0) {
    list.innerHTML = '<div class="empty-state">Keine Features gefunden.</div>';
    return;
  }

  filtered.forEach(fs => {
    const isOpen = expanded.has(fs.fid);
    const pct = MAX_VAL > 0 ? (fs.maxVal / MAX_VAL * 100).toFixed(1) : 0;

    const card = document.createElement('div');
    card.className = 'feat-card';

    // Header row
    const hdr = document.createElement('div');
    hdr.className = 'feat-header-row';
    const previewHtml = fs.preview
      ? escH(fs.preview) + (fs.spans[0] && fs.spans[0].length > 90 ? '&hellip;' : '')
      : '<span style="font-style:italic;color:var(--text-tertiary);">keine Beschreibung</span>';
    hdr.innerHTML =
      '<span class="feat-toggle">' + (isOpen ? '&#9660;' : '&#9654;') + '</span>' +
      '<span class="feat-id-badge">#' + fs.fid + '</span>' +
      '<div class="feat-bar-wrap"><div class="feat-bar" style="width:' + pct + '%;"></div></div>' +
      '<span class="feat-max-val">' + fs.maxVal.toFixed(2) + '</span>' +
      '<span class="feat-preview">' + previewHtml + '</span>';

    // Detail panel
    const detail = document.createElement('div');
    detail.className = 'feat-detail' + (isOpen ? ' open' : '');

    // Top tokens section
    const topToks = fs.activations.slice(0, 10);
    const tokMax = topToks.length > 0 ? topToks[0].v : 1;
    const tokHtml = topToks.length > 0
      ? topToks.map(t => {
          const tp = (t.v / tokMax * 100).toFixed(0);
          return '<div class="token-row">' +
            '<span class="token-text" title="Index ' + t.i + ': ' + escH(t.tok) + '">' +
              '[' + t.i + ']\\u00a0' + escH(displayTok(t.tok)) +
            '</span>' +
            '<div class="token-mini-bar-wrap"><div class="token-mini-bar" style="width:' + tp + '%;"></div></div>' +
            '<span class="token-val">' + t.v.toFixed(3) + '</span>' +
            '</div>';
        }).join('')
      : '<div class="no-data">Nicht aktiv in diesem Prompt.</div>';

    // Spans section
    const spansSlice = fs.spans.slice(0, 20);
    const spansHtml = spansSlice.length > 0
      ? spansSlice.map(s => '<div class="span-item">' + escH(s) + '</div>').join('')
      : '<div class="no-data">Keine Beschreibungen verf\\u00fcgbar.</div>';
    const moreNote = fs.spans.length > 20
      ? '<div class="no-data" style="margin-top:4px;">' + (fs.spans.length - 20) + ' weitere&hellip;</div>'
      : '';

    detail.innerHTML =
      '<div class="detail-section">' +
        '<h4>Top Tokens (' + fs.activations.length + ' aktiv)</h4>' +
        '<div>' + tokHtml + '</div>' +
      '</div>' +
      '<div class="detail-section">' +
        '<h4>Beschreibungen (' + fs.spans.length + ')</h4>' +
        '<div class="span-list">' + spansHtml + moreNote + '</div>' +
      '</div>';

    hdr.addEventListener('click', () => {
      if (expanded.has(fs.fid)) expanded.delete(fs.fid);
      else expanded.add(fs.fid);
      renderList();
    });

    card.appendChild(hdr);
    card.appendChild(detail);
    list.appendChild(card);
  });
}

document.getElementById('featSearch').addEventListener('input', e => {
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


def generate_heatmap_html(
    record: Dict[str, Any],
    output_path: str,
    title: Optional[str] = None,
    feature_descriptions: Optional[Dict[int, List[str]]] = None,
) -> None:
    """Render an interactive HTML feature explorer for a single extraction record."""
    tokens: List[str] = record["tokens"]
    rel_feats: List[int] = record["rel_features"]
    matrix: List[List[float]] = record["feature_matrix"]
    layer: int = int(record.get("layer", 0))
    prompt: str = record.get("prompt", "")
    top_k: int = int(record.get("top_k", 20))

    num_tok = len(tokens)
    num_feat = len(rel_feats)
    total_cells = num_tok * max(num_feat, 1)

    max_val = 0.0
    zero_cells = 0
    active_tokens = 0
    for row in matrix:
        row_has_value = False
        for v in row:
            if v == 0:
                zero_cells += 1
            else:
                if v > max_val:
                    max_val = float(v)
                row_has_value = True
        if row_has_value:
            active_tokens += 1

    sparsity_pct = round((zero_cells / total_cells) * 100) if total_cells else 0
    title = title or f"SAE Feature Activation Explorer · Layer {layer}"

    is_normalized = bool(record.get("normalized", False))
    excluded_feats: List[int] = record.get("excluded_features", [])

    norm_note = " · normalisiert (&divide; p95 Baseline)" if is_normalized else ""

    if excluded_feats:
        excluded_note = (
            '<p class="excluded-note">'
            f"<b>{len(excluded_feats)} Features ausgeschlossen</b> (nicht in Baseline-Datei): "
            + ", ".join(f"#{fid}" for fid in sorted(excluded_feats)[:30])
            + ("…" if len(excluded_feats) > 30 else "")
            + "</p>"
        )
    else:
        excluded_note = ""

    # Build {str(fid): [spans]} for embedding — empty list for features without descriptions.
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
        "__NFEAT__": str(num_feat),
        "__LAYER__": str(layer),
        "__MAX2__": f"{max_val:.2f}",
        "__SPARSE__": str(sparsity_pct),
        "__ACTIVE__": str(active_tokens),
        "__PROMPT_HTML__": _html_escape(prompt_preview),
        "__TOKENS__": json.dumps(tokens, ensure_ascii=False),
        "__FEATS__": json.dumps(rel_feats),
        "__MATRIX__": json.dumps(rounded),
        "__MAXVAL__": json.dumps(max_val if max_val > 0 else 1.0),
        "__TOPK__": str(top_k),
        "__NORM_NOTE__": norm_note,
        "__FEAT_DESCS__": json.dumps(feat_descs_json, ensure_ascii=False),
        "__EXCLUDED_NOTE__": excluded_note,
    }

    html = _HTML_TEMPLATE
    for needle, value in replacements.items():
        html = html.replace(needle, value)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def _html_path_for_record(html_dir: str, output_jsonl: str, idx: int) -> str:
    base = os.path.splitext(os.path.basename(output_jsonl))[0] or "heatmap"
    return os.path.join(html_dir, f"{base}_{idx:04d}.html")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FAC token-matrix test pipeline: prompt -> token-level SAE features using the union of each token's Top-k activations."
    )

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--device-id", type=str, default="0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])

    parser.add_argument("--model-name", type=str, required=True, help="Local path to model directory")
    parser.add_argument("--layer", type=int, default=16, help="1-based decoder layer index for activation extraction.")

    parser.add_argument("--sae-ckpt-path", type=str, default="", help="Local path to SAE checkpoint .pth")

    parser.add_argument("--prompts-json", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--max-prompts", type=int, default=0, help="0 means all prompts")

    parser.add_argument("--hf-cache-dir", type=str, default=_default_cache_dir())

    # HTML feature-explorer output
    parser.add_argument(
        "--emit-html",
        action="store_true",
        help="If set, additionally write an interactive HTML feature explorer per prompt.",
    )
    parser.add_argument(
        "--html-dir",
        type=str,
        default="",
        help="Directory for HTML outputs. Defaults to the directory of --output-jsonl.",
    )
    parser.add_argument(
        "--baseline-tsv",
        type=str,
        default=_DEFAULT_BASELINE_TSV,
        help="Path to feature_activation_baseline_agg.tsv for p95 normalization. "
             "Features absent from this file are excluded from analysis.",
    )
    parser.add_argument(
        "--feature-descriptions-tsv",
        type=str,
        default="",
        help="Path to a TSV file with columns feature_id and text (one span per row). "
             "Only rows for active features are loaded. Optional — explorer works without it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # This pipeline is intentionally strict local-only. Runtime network downloads are disabled.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    # Keep cache dir explicit so all loaders resolve to the same local location.
    os.environ["TRANSFORMERS_CACHE"] = args.hf_cache_dir
    os.makedirs(args.hf_cache_dir, exist_ok=True)

    if args.device == "cuda":
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    model_path = os.path.abspath(args.model_name)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Local model directory not found: {model_path}. "
            "Provide a local model snapshot path via --model-name."
        )

    prompts = load_prompts(args.prompts_json)
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    sae_ckpt = resolve_sae_checkpoint(local_path=args.sae_ckpt_path or None)

    model = UnifiedGenerator(
        model_path,
        device=args.device,
        dtype=args.dtype,
        cache_dir=args.hf_cache_dir,
        local_files_only=True,
        strict_local_paths=True,
    )
    collector = Collector(args.layer)
    mount_function(model._model, "llama", args.layer, collector)
    collector.early_stop = True

    sae = TopKSAE.from_disk(sae_ckpt, device=args.device)
    sae.topk = TOP_K
    sae.eval()

    # Load p95 baseline for normalization.
    baseline_p95: Optional[Dict[int, float]] = None
    if args.baseline_tsv:
        if not os.path.isfile(args.baseline_tsv):
            raise FileNotFoundError(f"Baseline TSV not found: {args.baseline_tsv}")
        baseline_p95 = load_baseline_p95(args.baseline_tsv)
        print(f"Loaded p95 baseline for {len(baseline_p95)} features from {args.baseline_tsv}")

    # Resolve HTML output directory once.
    html_dir = ""
    if args.emit_html:
        html_dir = args.html_dir or os.path.dirname(os.path.abspath(args.output_jsonl))
        if html_dir:
            os.makedirs(html_dir, exist_ok=True)

    records: List[Dict[str, Any]] = []
    with tc.no_grad():
        for idx, prompt in enumerate(tqdm.tqdm(prompts, desc="Processing prompts")):
            record = extract_token_feature_matrix_for_prompt(prompt, model, collector, sae, baseline_p95=baseline_p95)
            record["index"] = idx
            record["num_tokens"] = len(record["tokens"])
            record["num_features"] = len(record["rel_features"])
            records.append(record)

            if args.emit_html:
                # Load descriptions only for the features active in this prompt.
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

                html_path = _html_path_for_record(html_dir, args.output_jsonl, idx)
                title = f"SAE Feature Explorer · Prompt {idx} · Layer {record['layer']}"
                generate_heatmap_html(
                    record, html_path, title=title, feature_descriptions=feature_descriptions
                )

    # write_jsonl(records, args.output_jsonl)
    # print(f"Wrote {len(records)} records to {args.output_jsonl}")
    if args.emit_html:
        print(f"Wrote {len(records)} HTML feature explorers to {html_dir}")


if __name__ == "__main__":
    main()
