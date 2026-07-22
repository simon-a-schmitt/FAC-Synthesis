
STEP_1_PROMPT_TEMPLATE = """
# Role
You are a highly precise data synthesis generator for LLM fine-tuning.
Your task is to exclusively generate the INPUT CONTEXT for a specific
task. At this stage, you do NOT generate any answers, solutions, or
choice options.

# Context & Objective
You receive a content abstract describing what a new task instance is
about, and a reference example showing what task instances look like.
Your job is to realize the abstract's content in the form of the
reference example.

# Parameters

1. TARGET TASK DESCRIPTION:
{{TASK_BESCHREIBUNG}}

2. CONTENT ABSTRACT (binding for content):
{{CONTENT_ABSTRACT}}

3. STYLE DIRECTIVES (binding, may be empty):
{{STYLE_DIRECTIVES}}

4. STYLE AND FORMAT REFERENCE (binding for form only):
Use this example exclusively as a reference for form:
- RETAIN: the tone, jargon, register, structural conventions, and
  approximate text length of this example.
- DO NOT TAKE OVER: its factual content, its specific subject matter,
  or its wording. All content must come from the CONTENT ABSTRACT.

claim for damages many times what the statute permitted, Urban made a motion for partial summary judgment to limit Baker’s damages to $3,896, the amount of Urban’s profit on the plastic frames in which the Paper Insert was used. Urban seeks $21,248, a portion of the fees and costs incurred in making that motion. Based on Weingrad’s conduct and my review of Urban’s lawyers’ timesheets, that amount is entirely reasonable. 4. Withholding of Licensing Evidence As noted above, Baker and Weingrad acted in bad faith and unreasonably and vexatiously multiplied these proceedings with respect to the licensing evidence. Initially, Weingrad objected to production of documents on this issue on the ground that they were irrelevant — a position entirely without merit. See On Davis, 246 F.3d at 164-66 (<HOLDING>). Then, after production was ordered, Weingrad

5. STRUCTURAL REQUIREMENTS:
{{STRUCTURAL_REQUIREMENTS}}

# Goal
Generate ONE input context that:
- realizes the situation described in the CONTENT ABSTRACT concretely
  and completely, rather than as a summary ABOUT
  the CONTENT ABSTRACT — you may add supporting details consistent with it, but
  must not replace or drop its core content
- expresses the abstract's characteristic subject matter naturally
  and prominently throughout the text, without unnatural repetition
  of individual words,
- matches the form of the reference example,
- satisfies all STYLE DIRECTIVES and STRUCTURAL REQUIREMENTS.

# IMPORTANT CONSTRAINTS
- Output EXCLUSIVELY the raw input context. No introduction, no
  explanation, no answer.
- Do NOT copy phrases from the reference example.
""".strip()
