#!/usr/bin/env python3
"""
Prior-Matched Sampling: waehlt aus einem Pool synthetischer CTI-VSP-Beispiele
eine Teilmenge fester Groesse, deren 8 CVSS-Metrik-Marginalen einer
Zielverteilung moeglichst nahe kommen (Minimierung der summierten
Total-Variation-Distanz), geloest als exaktes ILP.

Optional:
  --max-per-product K   hoechstens K Beispiele pro Produkt/Vendor (Dedup)
  --seed S              zufaellige Tie-Break-Perturbation -> Replikate

Bewusst NICHT enthalten: jeder Bezug auf SAE-Feature-Coverage. Coverage wird
ausschliesslich post hoc als Outcome berichtet, nie als Constraint - sonst
wird genau der Confounder reintroduziert, den das Experiment isolieren soll.

Input : TSV ohne Header, Spalte 0 = Beschreibung, Spalte 1 = CVSS-Vektor
Output: TSV der gewaehlten Zeilen + JSON-Report
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_POOL = os.path.join(_SCRIPT_DIR, "input", "cti_vsp_bb_1025.tsv")
_DEFAULT_TARGET = os.path.join(_SCRIPT_DIR, "input", "cti_vsp_ft_200.tsv")

METRICS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]
CLASSES = {
    "AV": ["N", "A", "L", "P"],
    "AC": ["L", "H"],
    "PR": ["N", "L", "H"],
    "UI": ["N", "R"],
    "S":  ["U", "C"],
    "C":  ["H", "L", "N"],
    "I":  ["H", "L", "N"],
    "A":  ["H", "L", "N"],
}
SLOTS = [(m, c) for m in METRICS for c in CLASSES[m]]


# ---------------------------------------------------------------- parsing

def parse_vector(vec):
    """'CVSS:3.1/AV:N/AC:L/...' -> {'AV':'N', 'AC':'L', ...}"""
    out = {}
    for part in vec.strip().split("/"):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        if k in CLASSES:
            if v not in CLASSES[k]:
                raise ValueError(f"unbekannte Klasse {k}:{v} in {vec!r}")
            out[k] = v
    missing = [m for m in METRICS if m not in out]
    if missing:
        raise ValueError(f"fehlende Metriken {missing} in {vec!r}")
    return out


def load_tsv(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(f"{path}:{ln}: erwarte 2 TSV-Spalten")
            rows.append({"line": line, "text": parts[0],
                         "vec": parts[1].strip(),
                         "lab": parse_vector(parts[1])})
    return rows


def marginals(rows):
    out = {}
    n = len(rows)
    for m in METRICS:
        cnt = Counter(r["lab"][m] for r in rows)
        out[m] = {c: cnt.get(c, 0) / n for c in CLASSES[m]}
    return out


def tv_per_metric(p, q):
    return {m: 0.5 * sum(abs(p[m].get(c, 0.0) - q[m].get(c, 0.0))
                         for c in CLASSES[m]) for m in METRICS}


def entropy(p):
    return {m: -sum(v * np.log2(v) for v in p[m].values() if v > 0)
            for m in METRICS}


# ------------------------------------------------- integer target counts

def largest_remainder(props, n):
    """Reelle Anteile -> ganzzahlige Zielzahlen, die exakt auf n summieren."""
    raw = {k: v * n for k, v in props.items()}
    base = {k: int(np.floor(v)) for k, v in raw.items()}
    rest = n - sum(base.values())
    order = sorted(raw, key=lambda k: raw[k] - base[k], reverse=True)
    for k in order[:rest]:
        base[k] += 1
    return base


def target_counts(target_props, n):
    return {m: largest_remainder(
        {c: target_props[m].get(c, 0.0) for c in CLASSES[m]}, n)
        for m in METRICS}


# ------------------------------------------------------- product key (dedup)

def product_key(text):
    """Grobe Vendor-/Produkt-Signatur: erste 2 kapitalisierte bzw. Backtick-Tokens.
    Heuristik - nur fuer die Dedup-Obergrenze, nicht fuer die Auswertung."""
    toks = re.findall(r"`([^`]+)`|\b([A-Z][A-Za-z0-9\-]{2,})\b", text)
    flat = [a or b for a, b in toks]
    return " ".join(flat[:2]).lower() if flat else "<none>"


# ------------------------------------------------------------------ solve

def solve(rows, target_props, n_select, max_per_product=None,
          seed=None, class_balanced=False, time_limit=120.0):
    N = len(rows)
    tgt = target_counts(target_props, n_select)

    # Indikatormatrix A[i, s] = 1 wenn Beispiel i in Slot s faellt
    A = np.zeros((N, len(SLOTS)))
    for i, r in enumerate(rows):
        for s, (m, c) in enumerate(SLOTS):
            if r["lab"][m] == c:
                A[i, s] = 1.0

    # Gewichte: gleichgewichtet (konsistent zu Macro-F1 ueber 8 Metriken)
    # class_balanced=True -> seltene Zielklassen staerker gewichten
    w = np.ones(len(SLOTS))
    if class_balanced:
        for s, (m, c) in enumerate(SLOTS):
            p = max(target_props[m].get(c, 0.0), 1e-3)
            w[s] = 1.0 / p
        w = w / w.mean()

    n_aux = len(SLOTS)
    n_var = N + 2 * n_aux

    c_obj = np.concatenate([np.zeros(N), np.repeat(w, 2)])
    if seed is not None:
        rng = np.random.default_rng(seed)
        # winzige Perturbation nur auf den Auswahlvariablen -> zufaelliger
        # Tie-Break unter gleichwertigen Optima, ohne das Optimum zu aendern
        c_obj[:N] = rng.random(N) * 1e-4

    cons = []

    # (1) genau n_select Beispiele
    row = np.zeros(n_var); row[:N] = 1.0
    cons.append(LinearConstraint(row, n_select, n_select))

    # (2) Abweichungs-Constraints: sum_i x_i a_is - d+_s + d-_s = tgt_s
    M = np.zeros((len(SLOTS), n_var))
    lb = np.zeros(len(SLOTS))
    for s, (m, c) in enumerate(SLOTS):
        M[s, :N] = A[:, s]
        M[s, N + 2 * s] = -1.0      # d+
        M[s, N + 2 * s + 1] = 1.0   # d-
        lb[s] = tgt[m][c]
    cons.append(LinearConstraint(M, lb, lb))

    # (3) optional: Obergrenze pro Produkt
    if max_per_product:
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            groups[product_key(r["text"])].append(i)
        big = [g for g in groups.values() if len(g) > max_per_product]
        if big:
            G = np.zeros((len(big), n_var))
            for j, idxs in enumerate(big):
                G[j, idxs] = 1.0
            cons.append(LinearConstraint(G, -np.inf, max_per_product))

    integrality = np.concatenate([np.ones(N), np.zeros(2 * n_aux)])
    bounds = Bounds(lb=np.zeros(n_var),
                    ub=np.concatenate([np.ones(N), np.full(2 * n_aux, np.inf)]))

    res = milp(c=c_obj, constraints=cons, integrality=integrality,
               bounds=bounds, options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if not res.success:
        raise RuntimeError(f"ILP nicht geloest: {res.message}")

    pick = np.where(res.x[:N] > 0.5)[0]
    return [rows[i] for i in pick], sorted(pick.tolist())


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=_DEFAULT_POOL,
                    help="TSV mit dem Kandidatenpool")
    ap.add_argument("--target", default=_DEFAULT_TARGET,
                    help="TSV, dessen Marginalen als Ziel dienen (z.B. Gold-200)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="",
                    help="Ziel-TSV. Default: "
                    "feature_coverage_filter/output/<pool-stem>_prior_matched_n<n>[_seed<seed>].tsv")
    ap.add_argument("--report", default="",
                    help="Ziel-JSON fuer den Report. Default: "
                    "feature_coverage_filter/output/<pool-stem>_prior_matched_n<n>[_seed<seed>]_report.json")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-per-product", type=int, default=None)
    ap.add_argument("--class-balanced", action="store_true",
                    help="seltene Zielklassen hoeher gewichten (Arm B)")
    args = ap.parse_args()

    if not os.path.isfile(args.pool):
        raise FileNotFoundError(f"Pool-TSV nicht gefunden: {args.pool}")
    if not os.path.isfile(args.target):
        raise FileNotFoundError(f"Target-TSV nicht gefunden: {args.target}")

    suffix = f"_n{args.n}" + (f"_seed{args.seed}" if args.seed is not None else "")
    stem = os.path.splitext(os.path.basename(args.pool))[0]
    if not args.out:
        args.out = os.path.join(_SCRIPT_DIR, "output", f"{stem}_prior_matched{suffix}.tsv")
    if not args.report:
        args.report = os.path.join(_SCRIPT_DIR, "output", f"{stem}_prior_matched{suffix}_report.json")

    pool = load_tsv(args.pool)
    target = load_tsv(args.target)
    tp = marginals(target)

    # Machbarkeitspruefung: kann der Pool die Zielzahlen ueberhaupt liefern?
    pool_counts = {m: Counter(r["lab"][m] for r in pool) for m in METRICS}
    tgt = target_counts(tp, args.n)
    infeasible = []
    for m in METRICS:
        for c in CLASSES[m]:
            have, need = pool_counts[m].get(c, 0), tgt[m][c]
            if have < need:
                infeasible.append({"metric": m, "class": c,
                                   "need": need, "have": have})
    if infeasible:
        print("[WARN] Pool kann folgende Zielzahlen nicht erreichen:", file=sys.stderr)
        for d in infeasible:
            print(f"  {d['metric']}:{d['class']}  brauche {d['need']}, "
                  f"habe {d['have']}", file=sys.stderr)
        print("[WARN] Das ILP minimiert weiterhin die TV, das Restresiduum "
              "ist strukturell und MUSS berichtet werden.", file=sys.stderr)

    sel, idx = solve(pool, tp, args.n, args.max_per_product,
                     seed=args.seed, class_balanced=args.class_balanced)

    sm = marginals(sel)
    rm = marginals(pool)
    tv_sel = tv_per_metric(sm, tp)
    tv_pool = tv_per_metric(rm, tp)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in sel:
            fh.write(r["line"] + "\n")

    report = {
        "pool": args.pool, "target": args.target, "n_selected": len(sel),
        "seed": args.seed, "max_per_product": args.max_per_product,
        "class_balanced": args.class_balanced,
        "selected_indices": idx,
        "infeasible_slots": infeasible,
        "tv_per_metric_selected": tv_sel,
        "tv_mean_selected": sum(tv_sel.values()) / 8,
        "tv_per_metric_pool": tv_pool,
        "tv_mean_pool": sum(tv_pool.values()) / 8,
        "entropy_selected": entropy(sm),
        "entropy_target": entropy(tp),
        "marginals_selected": sm,
        "marginals_target": tp,
        "num_unique_vectors_selected": len({r["vec"] for r in sel}),
        "num_unique_vectors_target": len({r["vec"] for r in target}),
        "max_product_multiplicity": max(
            Counter(product_key(r["text"]) for r in sel).values()),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"n            : {len(sel)}")
    print(f"TV_mean pool : {report['tv_mean_pool']:.4f}")
    print(f"TV_mean sel  : {report['tv_mean_selected']:.4f}")
    print("TV je Metrik :", {k: round(v, 4) for k, v in tv_sel.items()})
    print(f"unique vecs  : {report['num_unique_vectors_selected']} "
          f"(Ziel {report['num_unique_vectors_target']})")
    print(f"max je Produkt: {report['max_product_multiplicity']}")
    print(f"out          : {args.out}")
    print(f"report       : {args.report}")


if __name__ == "__main__":
    main()