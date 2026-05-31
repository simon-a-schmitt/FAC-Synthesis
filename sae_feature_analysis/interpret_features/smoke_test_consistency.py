"""
Smoke-Test fuer collect_spans.py-Konsistenz.
 
Schickt EINEN bekannten Prompt (das GEN-CaseHold-Beispiel) durch exakt dieselbe
Lade-/Mount-/Aktivierungs-Logik wie collect_spans.py und prueft drei Dinge:
 
  1. Tokenisierung liefert genau 1x BOS  -> kein BOS-Doubling
  2. sae.actvs hat die volle (ungekuerzte) Sequenzlaenge -> Masking-Indexraum stimmt
  3. Feature #94 hat an Position 356 ("cases") den dichten Wert ~1.0265
     und dort auch sein Argmax -> Layer + Hook identisch zum Token-Matrix-Pfad
 
Aufruf (gleiche Argumente wie der echte collect_spans-Run):
  python smoke_test_consistency.py --model-key llama --sae-path <PFAD_ZUR_SAE.pth> --device cuda
  optional: --sae-layer <N>   (nur falls dein echter Run das auch setzt)
"""
import argparse
import torch as tc
 
from llm_surgery import switch_mode, mount_function
from generator import Generator
from autoencoder import load_pretrained
 
# Index-Helfer und die geprueften Funktionen direkt aus collect_spans wiederverwenden,
# damit der Test exakt denselben Code-Pfad nutzt:
from collect_spans import _find_user_content_start, _special_token_id_set
 
# Der GEN-Prompt aus dem Chat (identischer String -> identische Tokenisierung).
GEN_PROMPT = (
    "Identify the single correct legal holding statement from options A-E to fill "
    "the <HOLDING> placeholder in the citation and output only the corresponding letter. "
    "The plaintiff, a former employee of the defendant company, alleged that her termination "
    "was a result of retaliation for filing a workers' compensation claim. However, the court "
    "found that the defendant had presented evidence of a legitimate, non-retaliatory reason "
    "for the termination, namely, the plaintiff's consistent failure to meet sales quotas. "
    "See Russell v. McKinney Hosp. Venture, 235 F.3d 219, 223 (5th Cir.2000). The distinction "
    "between this case and others of its kind lies in the fact that the plaintiff had been warned "
    "repeatedly about her performance, whereas in similar cases, such as Johnson v. McDonnell "
    "Douglas Corp., 871 F.2d 1501, 1504 (8th Cir.1989), the employees had not received prior "
    "warnings. This difference sets the present case apart from others in which the courts have "
    "found retaliatory discharge. The court's decision was based on the principle that an "
    "employer's actions are not retaliatory if they are motivated by a legitimate business reason, "
    "as opposed to cases where the employer's actions are taken in response to an employee's "
    "exercise of a protected right. See, e.g., Williams v. Gen. Motors Corp., 901 F.2d 1508, 1512 "
    "(8th Cir.1990) (<HOLDING>). In this case, the defendant's decision to terminate the plaintiff "
    "was based on her poor performance, which distinguishes it from cases where the termination was "
    "based on retaliatory motives. A. holding that an employer's termination of an employee based "
    "on a protected characteristic, such as age or disability, is permissible if the employer can "
    "demonstrate a legitimate business necessity for the termination. B. holding that the filing of "
    "a workers' compensation claim constitutes a waiver of an employee's right to pursue a "
    "retaliation claim under federal law. C. holding that an employer's termination of an employee "
    "based on poor performance, where the employee has received prior warnings, is not retaliatory "
    "if the employer's actions are motivated by a legitimate business reason rather than the "
    "employee's exercise of a protected right. D. holding that an employee's failure to exhaust "
    "administrative remedies prior to filing a lawsuit bars their claim for retaliatory discharge, "
    "regardless of the merits of the claim. E. holding that the existence of a collective bargaining "
    "agreement between the employer and the employee's union precludes an individual employee from "
    "bringing a retaliation claim under federal or state law."
)
 
TARGET_FEATURE = 94
EXPECTED_POS = 356
EXPECTED_VAL = 1.0265
EXPECTED_SEQLEN = 554
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True, choices=["mistral", "llama", "qwen"])
    ap.add_argument("--sae-path", required=True, help="GENAU der Pfad aus dem collect_spans-Slurm")
    ap.add_argument("--sae-layer", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
 
    model_map = {"mistral": "mistral-7b", "llama": "llama3-8b", "qwen": "qwen2.5-7b"}
 
    # --- identische Lade-/Mount-Sequenz wie collect_spans.main() ---
    name, layer, sae = load_pretrained(args.sae_path, device=args.device)
    print(f"[load_pretrained] name={name}  layer(aus Dateiname)={layer}")
    if args.sae_layer is not None:
        print(f"[override] layer {layer} -> {args.sae_layer}")
        layer = args.sae_layer
 
    dtype = "float32" if args.device == "cpu" else "bfloat16"
    generator = Generator(model_map[args.model_key], device=args.device, dtype=dtype)
    tokenizer = generator._tokenizer
 
    print(f"[mount] mounting layer_idx={int(layer)} ...")
    mount_function(generator._model, args.model_key, int(layer), sae)  # druckt 'Mounted hook at ...'
 
    # --- identische Aktivierungsberechnung wie activations() ---
    switch_mode(sae, "train")
    sae.topk = 65536  # dichte Aktivierungen
 
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": GEN_PROMPT}],
        tokenize=True, add_generation_prompt=False, return_tensors="pt",
    )
    input_ids = enc["input_ids"] if isinstance(enc, dict) else enc
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    ids = input_ids[0].to(generator._device)
    token_ids = ids.tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
 
    try:
        with tc.no_grad():
            generator.get_activates(ids)
    except RuntimeError:
        pass
 
    A = sae.actvs.squeeze()  # erwartet [seq, n_features]
 
    print("\n" + "=" * 60)
    print("ERGEBNISSE")
    print("=" * 60)
 
    # Check 1: BOS-Doubling
    n_bos = int((ids == tokenizer.bos_token_id).sum())
    print(f"[1] BOS-Count           = {n_bos}   (erwartet: 1)")
 
    # Check 2: actvs-Laenge == ids-Laenge == 554
    print(f"[2] len(ids)            = {len(token_ids)}   (erwartet: {EXPECTED_SEQLEN})")
    print(f"    actvs.shape         = {tuple(A.shape)}   (dim0 muss == len(ids) sein)")
 
    # content_start zur Info
    cs = _find_user_content_start(token_ids, tokens, tokenizer)
    print(f"    content_start       = {cs}   (erwartet: 30)")
    print(f"    tokens[content_start] = {tokens[cs]!r}   (erwartet: 'Ident' o. 'ĠIdent')")
 
    # Check 3: Feature #94 @ 356
    if A.dim() == 2 and A.shape[0] > EXPECTED_POS:
        val_at = float(A[EXPECTED_POS, TARGET_FEATURE])
        argmax = int(A[:, TARGET_FEATURE].argmax())
        maxval = float(A[:, TARGET_FEATURE].max())
        print(f"[3] #94 @ idx{EXPECTED_POS}      = {val_at:.4f}   (erwartet: ~{EXPECTED_VAL})")
        print(f"    #94 argmax pos      = {argmax}   (erwartet: {EXPECTED_POS})")
        print(f"    #94 max value       = {maxval:.4f}")
        if tokens and argmax < len(tokens):
            print(f"    token@argmax        = {tokens[argmax]!r}   (erwartet: 'Ġcases')")
    else:
        print(f"[3] actvs hat unerwartete Form {tuple(A.shape)} - kann #94 nicht lesen!")
 
    print("=" * 60)
    print("Interpretation:")
    print("  Check1 != 1            -> BOS-Doubling, Offsets um 1 verschoben")
    print("  Check2 len != actvs    -> Hook kuerzt Sequenz, Masking-Indexraum kaputt")
    print("  Check3 weit daneben    -> falscher Layer oder Hook (Layer-Inkonsistenz)")
    print("  alle drei OK           -> Run sauber startbar")
 
 
if __name__ == "__main__":
    main()