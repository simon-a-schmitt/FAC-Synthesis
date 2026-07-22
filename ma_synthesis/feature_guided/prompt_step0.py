
STEP_0_PROMPT_TEMPLATE = """
# Role
You are a data synthesis assistant. Your task is to write a short
content abstract that will later be turned into a training example
for a specific task. You do NOT write the training example itself.
# Target Task
{{TASK_BESCHREIBUNG}}
# Target Features
Training examples must exhibit specific activation patterns (features)
of an AI model. You receive two features. For each, the TOP TOKENS
show what the feature activates on within the target domain (primary
signal), and the REFERENCE SPANS show activations in general web text
(secondary signal, may be misleading for the target domain).

[FEATURE 1]
Top tokens:
{{SEED_EXAMPLE_1_TOP_TOKENS}}

Reference spans:
{{SEED_EXAMPLE_1_REFERENCE_SPANS}}

[FEATURE 2]
Top tokens:
{{SEED_EXAMPLE_2_TOP_TOKENS}}

Reference spans:
{{SEED_EXAMPLE_2_REFERENCE_SPANS}}

# Instructions
Write an abstract (100-150 words) describing the concrete content of
ONE new task instance in which the patterns of both features occur
naturally and prominently. Be specific, not generic: the abstract must
state the concrete subject matter of the instance — what it is
substantively about — not only its procedural or formal frame.
Do not imitate the task description's wording.
If a feature reflects a formatting or notation pattern rather than
content (e.g. citations, numbering, markup), do not build content
around it; instead add a style directive for it.
If a feature shows no discernible pattern at all, ignore it and work
with the remaining feature alone.
If, after applying the rules above, no meaningful abstract for the
target task can be built from the given features — because no feature
carries usable content, or because the features cannot be reconciled
into one coherent task instance — do not force an abstract. Output
the failure object instead.
# Output
A single JSON object, no other text.
On success:
{
  "abstract": "<100-150 words>",
  "style_directives": ["<only for formatting patterns, else empty>"]
}
On failure:
{
  "error": "cannot_interpret",
  "reason": "<one sentence>"
}
""".strip()
