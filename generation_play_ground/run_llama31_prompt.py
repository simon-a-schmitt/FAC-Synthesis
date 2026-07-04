import sys
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark_play_ground.model_wrapper import LocalModel


MODEL_PATH = Path("/pfs/work9/workspace/scratch/ka_ai3967-master_thesis_exp/models/llama-3.1-70b")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "bfloat16"


SYSTEM_PROMPT = """
# Rolle:
Du bist mein persönlicher Data Sample Generator! Du bist sehr erfahren und kompetent 
darin auf Basis eines Ziel-Features für einen gegebenen Ziel-Task neue synthetische
Beispiele zu generieren!

# Kontext:
Das Ziel ist es für einen Ziel-Task neue synthetische Beispiele (Task + Musterlösung) 
zu generieren. Die synthetischen Beispiele sollen das Ziel-Feature bei der Verarbeitung
durch die Neuronen eines LLMs gezielt und signifikant aktivieren.

# Aufgabe:
Du erhältst eine textuelle Beschreibung des Ziel-Features. Außerdem erhältst du eine
Liste von Text Spans aus einem allgemeinen Text-Corpus die das Feature bei der 
Verarbeitung durch ein LLM stark aktivieren.
Zudem erhälst du eine kurze textuelle Beschreibung des Ziel-Tasks und ein Beispiel dieses
Tasks inkl. Musterlösung.
Du sollst verstehen worum es bei dem Ziel-Task konzeptionell geht und was die 
inhaltlichen Rahmenbedingungen sind. Anschließend sollst du 3 synthetische Beispiele als
Instanzen des Ziel-Tasks generieren. Diese sollen folgende Eigenschaften erfüllen:
- Das Ziel-Feature soll durch diese Beispiele jeweils signifikant und präzise aktivert werden
- Die Beispiele sollen dem Stil und Format des Ziel-Tasks (definiert durch Beschreibung & Beispiel) entsprechen z.B. im Bezug auf Wortwahl, Schwierigkeitsgard und Textlänge
- Die Beispiele dürfen auf keinen Fall dem gegebenen Beispiel inhaltlich ähneln oder auch nur Teile/Passagen daraus übernehmen!
- Die Beispiele sollen sich untereinander inhaltlich unterscheiden
- Die zugehörige Musterlösung soll korrekt und präzise sein
- Falls das Task-Formt Multiple-Choice ist soll der Buchstabe der korrekten Antwort zwischen den generierten Beispielen variieren

# Output:
Gebe die neuen synthetischen Beispiele im folgenden Format aus:
{
"task": [Synthetisches Beispiel]
"label": [Zugehörige Musterlösung]
"feature_context": [Kurze Erläuterung wo und wie das Ziel-Feature hier aktiviert wird]
}
"""


TARGET_TASK_DESCRIPTION = "Die Aufgabe besteht aus Multiple-Choice-Fragen mit Textausschnitten aus einer Gerichtsentscheidung und mehreren potenziellen Leitsätzen (Holdings), von denen einer korrekt ist und zitiert werden könnte. Leitsätze sind von zentraler Bedeutung für das Case-Law-System (Fallrecht). Sie repräsentieren die maßgebliche Rechtsregel, wenn das Recht auf einen bestimmten Sachverhalt angewendet wird. Sie stellen das Präzedenzfallrecht dar und sind das, worauf sich Prozessbeteiligte in nachfolgenden Verfahren berufen können. Die aus dem Datensatz abgeleitete „CaseHOLD“-Aufgabe ist eine Multiple-Choice-Fragebeantwortung mit fünf potenziellen Leitsätzen (einer richtig, vier falsch) für jeden zitierenden Kontext."

TARGET_TASK_EXAMPLE_PROMPT = "Identify the single correct legal holding statement from options A-E to fill the <HOLDING> placeholder in the citation and output only the corresponding letter. to agree with the district court’s construction of the function of this means-plus-function claim, and disagrees only with its finding that the specification did not disclose a corresponding structure. Because this means-plus-function term is a computer-implemented one, the patent must disclose more than a general purpose processor; it must also include an algorithm to perform the function. See Aristocrat, 521 F.3d at 1333 (“In cases involving a computer-implemented invention in which the inventor has invoked means- plus-function claiming, this court has consistently required that the structure disclosed in the specification be more than simply a general purpose computer or microprocessor.”); see also Ergo Licensing, LLC v. CareFusion 303, Inc., 673 F.3d 1361, 1365 (Fed. Cir. 2012) (<HOLDING>). The district court found that the ’616 patent A. holding that the error must be egregious or othexnvise constitute a manifest miscarriage of justice citation omitted B. recognizing that dismissal is warranted only in extreme cases citation omitted C. holding that prejudice may result when the alien is prevented from  reasonably presenting his or her case  citation omitted emphasis added D. holding that the weight and credibility to be given to the opinions of expert witnesses is uniquely within the province of the fact finder  in this instance the trial court citation omitted E. holding that however an algorithm is expressed it must be a stepbystep procedure for accomplishing a given result citation omitted"
TARGET_TASK_EXAMPLE_SOLUTION = "E"

TARGET_FEATURE_DESCRIPTION = "Sprachliche Muster und Ausdrücke, die dazu dienen, Entitäten, Konzepte oder Fälle voneinander abzugrenzen, Unterschiede hervorzuheben oder Alleinstellungsmerkmale zu identifizieren (wie \"distinguishes itself from\", \"set Elkhorn apart from\" oder \"How is it different from\")."
TARGET_FEATURE_TEXT_SPANS = "Span 1: nRainbow River in Florida is different from\nSpan 2: nThe Saint Lambert March distinguishes itself from\nSpan 3:  architectural style distinguishes Ghadamès from\nSpan 4: nOne aspect that set Elkhorn apart from\nSpan 5: cks on their surface. Lighter in color than\nSpan 6: azzano.\nSo what sets this apart from\nSpan 7: : How is the Chevy Volt different than\nSpan 8:  are a unique feature that sets it apart from many\nSpan 9:  Richard Nelson sees NASA as significantly different when compared to\nSpan 10:  in the world.\nHow is it different from"

EXAMPLE_1 = "If the clause says the agreement ends immediately on breach, classify it as termination for cause."
EXAMPLE_2 = "If the clause states payment must be made within 15 days of invoice, classify it as a payment term."
EXAMPLE_3 = "If the clause requires written consent before assignment, classify it as an assignment restriction."

EXAMPLES_BLOCK = f"""Example 1: {EXAMPLE_1}
Example 2: {EXAMPLE_2}
Example 3: {EXAMPLE_3}"""


TASK_PROMPT = f"""
# Input: 
## Ziel-Task:
### Beschreibung:
{TARGET_TASK_DESCRIPTION}
# Beispiel:
#### Prompt:
{TARGET_TASK_EXAMPLE_PROMPT}
#### Lösung:
{TARGET_TASK_EXAMPLE_SOLUTION}
## Ziel-Feature:
### Beschreibung:
{TARGET_FEATURE_DESCRIPTION}
### Text Spans:
{TARGET_FEATURE_TEXT_SPANS}
"""


def generate_response(task_prompt: str, max_new_tokens: int, temperature: float) -> str:
    model = LocalModel(model_path=str(MODEL_PATH), device=DEVICE, dtype=DTYPE)
    return model.generate(
        prompt=task_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        system=SYSTEM_PROMPT,
    ).strip()


def main() -> None:
    print(TASK_PROMPT, flush=True)
    print("Loading model...", flush=True)
    response = generate_response(
        task_prompt=TASK_PROMPT,
        max_new_tokens=512,
        temperature=0.5,
    )
    print("=== RESPONSE ===", flush=True)
    print(response, flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()