import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark_play_ground.model_wrapper import LocalModel
from ma_synthesis.feature_guided.prompt_step1 import STEP_1_PROMPT_TEMPLATE
from ma_synthesis.common.prompt_variables import SYSTEM_PROMPT


STEP_1_MAX_NEW_TOKENS = 2048
STEP_1_TEMPERATURE = 0.5

TASK_BESCHREIBUNG = (
    "Die Aufgabe besteht darin, einen realistischen Textausschnitt aus einer"
    " US-Gerichtsentscheidung (einen sogenannten \"citing context\") zu generieren."
    " Dieser Text beschreibt einen rechtlichen Sachverhalt oder die Argumentation eines"
    " Gerichts. Der Text muss an einer strategischen Stelle abbrechen – genau dort, wo"
    " ein verbindlicher rechtlicher Leitsatz (eine \"Holding\") eines anderen Falls"
    " zitiert wird."
)

TASK_BEISPIEL_PROMPT = (
    "was unable to establish a foundation for his federal claims because he could not"
    " demonstrate Officer Ahlm's conduct violated a constitutional right. See Grubbs v."
    " Bailes, 445 F.3d 1275, 1278 (10th Cir.2006). The conclusion was threefold. First,"
    " Mr. Titus's malicious prosecution claim failed because \"Officer Ahlm possessed"
    " probable cause to believe that Mr. Titus had been operating his vehicle while"
    " intoxicated to the slightest degree,\" even after determining Mr. Titus had only a"
    " .02% breath alcohol concentration (BAC). Aplt. App. at 131. Second, Mr. Titus's"
    " retaliatory prosecution claim failed because Mr. Titus did not plead and prove the"
    " absence of probable cause for charging him with DWI. See id. at 137; Hartman v."
    " Moore, 547 U.S. 250, 265-66, 126 S.Ct. 1695, 164 L.Ed.2d 441 (2006) (<HOLDING>)"
)

WEITERE_ANWEISUNG = (
    "Fuege an der Stelle im Text, an der das Holding stehen muss, das Tag (<HOLDING>) ein."
)


def _render_template(template: str, **values: str) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def run_step1(
    model: LocalModel,
    feature_description: str,
    feature_text_spans: str,
) -> str:
    prompt = _render_template(
        STEP_1_PROMPT_TEMPLATE,
        TASK_BESCHREIBUNG=TASK_BESCHREIBUNG,
        FEATURE_BESCHREIBUNG=feature_description,
        TEXT_SPANS=feature_text_spans,
        TASK_BEISPIEL=f"Prompt:\n{TASK_BEISPIEL_PROMPT}",
        WEITERE_ANWEISUNG=WEITERE_ANWEISUNG,
    )
    return model.generate(
        prompt=prompt,
        max_new_tokens=STEP_1_MAX_NEW_TOKENS,
        do_sample=STEP_1_TEMPERATURE > 0,
        temperature=STEP_1_TEMPERATURE,
        system=SYSTEM_PROMPT,
    ).strip()
