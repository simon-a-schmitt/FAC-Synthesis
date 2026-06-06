import argparse
import json
import os
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

    feature_matrix = sparse_features[:, rel_feature_tensor].tolist()
 
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
    }
 
 
def write_jsonl(records: List[Dict[str, Any]], output_jsonl: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(output_jsonl))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
 
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
 
 
# ---------------------------------------------------------------------------
# HTML heatmap generation
# ---------------------------------------------------------------------------
 
_HTML_TEMPLATE = r"""<!DOCTYPE html>
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
  .stats { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
  .stat { background: var(--bg-secondary); border-radius: 8px;
    padding: 10px 14px; min-width: 90px; }
  .stat-label { font-size: 12px; color: var(--text-secondary); }
  .stat-value { font-size: 20px; font-weight: 500; margin-top: 2px; }
  .controls { display: flex; align-items: center; gap: 16px; margin-bottom: 12px; font-size: 13px; }
  .toggle { display: flex; align-items: center; gap: 6px; cursor: pointer; user-select: none; }
  .legend { display: flex; align-items: center; gap: 14px;
    font-size: 12px; color: var(--text-secondary); margin-bottom: 12px; }
  .legend-bar { width: 200px; height: 10px; border-radius: 5px;
    background: linear-gradient(to right, #EFE7FA, #6B46C1, #2D1B5E); }
  .heatmap-wrap { border: 0.5px solid var(--border); border-radius: 12px;
    padding: 12px; background: var(--bg); overflow: auto; max-height: 75vh; }
  #heatmap { display: grid; gap: 1px; font-size: 11px; }
  .feat-header { font-size: 10px; color: var(--text-secondary);
    writing-mode: vertical-rl; transform: rotate(180deg);
    padding: 4px 0; white-space: nowrap; align-self: end;
    text-align: center; position: sticky; top: 0;
    background: var(--bg); z-index: 5; }
  .header-corner { position: sticky; top: 0; left: 0; background: var(--bg); z-index: 10; }
  .header-label { position: sticky; top: 0; background: var(--bg); z-index: 9;
    font-size: 10px; color: var(--text-tertiary); align-self: end; padding-bottom: 4px; }
  .tok-idx { font-size: 10px; color: var(--text-tertiary);
    text-align: right; padding-right: 6px; align-self: center;
    position: sticky; left: 0; background: var(--bg); z-index: 3; }
  .tok-label { font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 11px; padding: 2px 8px 2px 4px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    align-self: center;
    position: sticky; left: 32px; background: var(--bg); z-index: 3; }
  .tok-label.special { color: var(--text-tertiary); }
  .tok-label.zero-row { opacity: 0.4; }
  .cell { border-radius: 2px; cursor: crosshair; }
  .cell:hover { outline: 1.5px solid var(--text-primary); outline-offset: -1px; z-index: 2; position: relative; }
  #tooltip { position: fixed; pointer-events: none;
    background: var(--bg); border: 0.5px solid var(--border-strong);
    border-radius: 8px; padding: 8px 10px; font-size: 12px;
    opacity: 0; transition: opacity .1s; z-index: 1000;
    box-shadow: 0 2px 12px rgba(0,0,0,0.15); max-width: 280px; }
  .top-section { margin-top: 32px; }
  .top-section h2 { font-size: 16px; font-weight: 500; margin: 0 0 12px; }
  .top-grid { display: grid;
    grid-template-columns: auto 1fr auto auto;
    gap: 8px 16px; font-size: 13px; align-items: center; }
  .top-grid .head { color: var(--text-tertiary); font-size: 11px; }
  .top-bar-wrap { width: 100px; height: 6px; background: var(--bg-secondary);
    border-radius: 3px; overflow: hidden; }
  .top-bar { height: 100%; background: var(--accent); }
  .top-tok { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px; }
  .feat-id { color: var(--text-secondary); }
</style>
</head>
<body>
<div class="container">
  <h1>__TITLE__</h1>
  <p class="subtitle">Vereinigungsmenge der Token-Top-__TOPK__ Features über __NTOK__ Tokens · Layer __LAYER__</p>
  <div class="prompt-box">__PROMPT_HTML__</div>
 
  <div class="stats">
    <div class="stat"><div class="stat-label">Tokens</div><div class="stat-value">__NTOK__</div></div>
    <div class="stat"><div class="stat-label">Features</div><div class="stat-value">__NFEAT__</div></div>
    <div class="stat"><div class="stat-label">Layer</div><div class="stat-value">__LAYER__</div></div>
    <div class="stat"><div class="stat-label">Top-k pro Token</div><div class="stat-value">__TOPK__</div></div>
    <div class="stat"><div class="stat-label">Max-Aktivierung</div><div class="stat-value">__MAX2__</div></div>
    <div class="stat"><div class="stat-label">Sparsity</div><div class="stat-value">__SPARSE__%</div></div>
    <div class="stat"><div class="stat-label">Aktive Tokens</div><div class="stat-value">__ACTIVE__ / __NTOK__</div></div>
  </div>
 
  <div class="controls">
    <label class="toggle">
      <input type="checkbox" id="hideZero" checked />
      Leere Tokens ausblenden
    </label>
  </div>
 
  <div class="legend">
    <span>0</span>
    <div class="legend-bar"></div>
    <span>__MAX1__</span>
    <span style="margin-left:auto;">Maus über Zelle für Details · Header bleiben beim Scrollen sichtbar</span>
  </div>
 
  <div class="heatmap-wrap">
    <div id="heatmap"></div>
  </div>
 
  <div id="tooltip"></div>
 
  <div class="top-section">
    <h2>Top 15 stärkste Aktivierungen</h2>
    <div id="top-list" class="top-grid"></div>
  </div>
</div>
 
<script>
const TOKENS = __TOKENS__;
const REL_FEATURES = __FEATS__;
const MATRIX = __MATRIX__;
const MAX_VAL = __MAXVAL__;
 
const displayTok = t => t
  .replace(/\u0120/g, '\u2581')
  .replace(/\u010a/g, '\u23ce')
  .replace(/\u00e2\u0122\u013b/g, "'")
  .replace(/\u00e2\u0122\u013d/g, '"')
  .replace(/\u00e2\u0122\u013c/g, '"');
 
function colorFor(v) {
  if (v === 0) return 'var(--bg-secondary)';
  const t = Math.min(1, v / MAX_VAL);
  const stops = [[239,231,250],[107,70,193],[45,27,94]];
  let c;
  if (t < 0.5) {
    const k = t / 0.5;
    c = stops[0].map((a,i)=> Math.round(a + (stops[1][i]-a)*k));
  } else {
    const k = (t-0.5)/0.5;
    c = stops[1].map((a,i)=> Math.round(a + (stops[2][i]-a)*k));
  }
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
 
const grid = document.getElementById('heatmap');
const tooltip = document.getElementById('tooltip');
const N_FEAT = REL_FEATURES.length;
const CELL = 18;
const LABEL_W = 140;
const TOK_IDX_W = 32;
 
function render(hideZero) {
  grid.innerHTML = '';
  grid.style.gridTemplateColumns = `${TOK_IDX_W}px ${LABEL_W}px repeat(${N_FEAT}, ${CELL}px)`;
 
  const corner = document.createElement('div');
  corner.className = 'header-corner';
  grid.appendChild(corner);
  const hdrLabel = document.createElement('div');
  hdrLabel.className = 'header-label';
  hdrLabel.textContent = 'Token';
  grid.appendChild(hdrLabel);
  REL_FEATURES.forEach((fid, j) => {
    const cell = document.createElement('div');
    cell.className = 'feat-header';
    cell.textContent = '#' + fid;
    cell.title = `Feature ${fid} (Spalte ${j})`;
    grid.appendChild(cell);
  });
 
  TOKENS.forEach((tok, i) => {
    const row = MATRIX[i];
    const isZeroRow = row.every(v => v === 0);
    if (hideZero && isZeroRow) return;
 
    const idxCell = document.createElement('div');
    idxCell.className = 'tok-idx';
    idxCell.textContent = i;
    grid.appendChild(idxCell);
 
    const labelCell = document.createElement('div');
    const isSpecial = tok.startsWith('<|');
    labelCell.className = 'tok-label' + (isSpecial ? ' special' : '') + (isZeroRow ? ' zero-row' : '');
    labelCell.textContent = displayTok(tok);
    labelCell.title = tok;
    grid.appendChild(labelCell);
 
    row.forEach((v, j) => {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.style.cssText = `height:${CELL-1}px; background:${colorFor(v)};`;
      cell.addEventListener('mousemove', (e) => {
        tooltip.style.opacity = '1';
        let left = e.clientX + 14;
        if (left + 280 > window.innerWidth) left = e.clientX - 290;
        tooltip.style.left = left + 'px';
        tooltip.style.top = (e.clientY + 14) + 'px';
        tooltip.innerHTML = `
          <div style="font-weight:500; margin-bottom:4px;">Aktivierung: ${v.toFixed(3)}</div>
          <div style="color:var(--text-secondary); margin-bottom:2px;">Token [${i}]: <span style="font-family:ui-monospace,monospace;">${displayTok(tok)}</span></div>
          <div style="color:var(--text-secondary);">Feature #${REL_FEATURES[j]} (Spalte ${j})</div>
        `;
      });
      cell.addEventListener('mouseleave', () => { tooltip.style.opacity = '0'; });
      grid.appendChild(cell);
    });
  });
}
 
document.getElementById('hideZero').addEventListener('change', (e) => render(e.target.checked));
render(true);
 
const flat = [];
MATRIX.forEach((row, i) => row.forEach((v, j) => {
  if (v > 0) flat.push({ v, i, j });
}));
flat.sort((a,b) => b.v - a.v);
 
const topList = document.getElementById('top-list');
topList.innerHTML = `
  <div class="head">#</div>
  <div class="head">Token</div>
  <div class="head">Feature</div>
  <div class="head">Aktivierung</div>
`;
if (flat.length === 0) {
  topList.insertAdjacentHTML('beforeend',
    `<div style="grid-column:1/-1; color:var(--text-tertiary);">Keine Aktivierungen &gt; 0.</div>`);
} else {
  flat.slice(0, 15).forEach((d, k) => {
    const pct = (d.v / Math.max(MAX_VAL, 1e-9) * 100).toFixed(0);
    topList.insertAdjacentHTML('beforeend', `
      <div style="color:var(--text-tertiary);">${k+1}</div>
      <div class="top-tok">[${d.i}] ${displayTok(TOKENS[d.i])}</div>
      <div class="feat-id">#${REL_FEATURES[d.j]}</div>
      <div style="display:flex; align-items:center; gap:10px;">
        <div class="top-bar-wrap"><div class="top-bar" style="width:${pct}%;"></div></div>
        <span style="font-variant-numeric: tabular-nums; min-width:42px; text-align:right;">${d.v.toFixed(2)}</span>
      </div>
    `);
  });
}
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
 
 
def generate_heatmap_html(record: Dict[str, Any], output_path: str, title: Optional[str] = None) -> None:
    """Render an interactive HTML heatmap for a single extraction record.
 
    The record must contain: tokens, rel_features, feature_matrix, layer, prompt.
    """
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
    title = title or f"SAE Feature Activation Heatmap · Layer {layer}"
 
    # Round matrix values for compact JSON; preserve zeros explicitly.
    rounded = [[round(float(v), 3) for v in row] for row in matrix]
 
    prompt_preview = prompt if len(prompt) <= 1200 else prompt[:1200] + "…"
    replacements = {
        "__TITLE__": _html_escape(title),
        "__NTOK__": str(num_tok),
        "__NFEAT__": str(num_feat),
        "__LAYER__": str(layer),
        "__MAX1__": f"{max_val:.1f}",
        "__MAX2__": f"{max_val:.2f}",
        "__SPARSE__": str(sparsity_pct),
        "__ACTIVE__": str(active_tokens),
        "__PROMPT_HTML__": _html_escape(prompt_preview),
        "__TOKENS__": json.dumps(tokens, ensure_ascii=False),
        "__FEATS__": json.dumps(rel_feats),
        "__MATRIX__": json.dumps(rounded),
        "__MAXVAL__": json.dumps(max_val if max_val > 0 else 1.0),
        "__TOPK__": str(top_k),
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
 
    # HTML heatmap output
    parser.add_argument(
        "--emit-html",
        action="store_true",
        help="If set, additionally write an interactive HTML heatmap per prompt.",
    )
    parser.add_argument(
        "--html-dir",
        type=str,
        default="",
        help="Directory for HTML heatmaps. Defaults to the directory of --output-jsonl.",
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
 
    # Resolve HTML output directory once.
    html_dir = ""
    if args.emit_html:
        html_dir = args.html_dir or os.path.dirname(os.path.abspath(args.output_jsonl))
        if html_dir:
            os.makedirs(html_dir, exist_ok=True)
 
    records: List[Dict[str, Any]] = []
    with tc.no_grad():
        for idx, prompt in enumerate(tqdm.tqdm(prompts, desc="Processing prompts")):
            record = extract_token_feature_matrix_for_prompt(prompt, model, collector, sae)
            record["index"] = idx
            record["num_tokens"] = len(record["tokens"])
            record["num_features"] = len(record["rel_features"])
            records.append(record)
 
            if args.emit_html:
                html_path = _html_path_for_record(html_dir, args.output_jsonl, idx)
                title = f"SAE Feature Activation Heatmap · Prompt {idx} · Layer {record['layer']}"
                generate_heatmap_html(record, html_path, title=title)
 
    write_jsonl(records, args.output_jsonl)
    print(f"Wrote {len(records)} records to {args.output_jsonl}")
    if args.emit_html:
        print(f"Wrote {len(records)} HTML heatmaps to {html_dir}")
 
 
if __name__ == "__main__":
    main()