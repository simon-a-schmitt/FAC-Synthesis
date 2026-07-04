STEP_4_MAX_NEW_TOKENS = 2048
STEP_4_TEMPERATURE = 0.0

# TODO: Refine this template once the final output format is confirmed.
STEP_4_PROMPT_TEMPLATE = """
Du erhältst alle Bestandteile fuer ein CaseHOLD Multiple-Choice-Trainingsbeispiel.
Deine Aufgabe: Formatiere das finale Trainingsbeispiel exakt nach dem vorgegebenen Schema.

Der korrekte Leitsatz soll zufaellig einer der Optionen A–E zugeordnet werden.
Die restlichen Positionen werden mit den Distraktoren gefuellt.
Merke: Der korrekte Buchstabe muss im Feld "label" stehen.

# Citing Context (mit <HOLDING> Platzhalter):
{{CITING_CONTEXT}}

# Korrekter Leitsatz:
{{CORRECT_HOLDING}}

# Distraktoren (nummeriert 1–4):
{{DISTRACTORS}}

# Korrekte Antwort-Position (A/B/C/D/E):
{{CORRECT_POSITION}}

# Ausgabeformat (exakt so ausgeben, kein anderer Text):
{
  "prompt": "[CITING_CONTEXT] A. [Option A] B. [Option B] C. [Option C] D. [Option D] E. [Option E]",
  "label": "[CORRECT_POSITION]"
}
""".strip()
