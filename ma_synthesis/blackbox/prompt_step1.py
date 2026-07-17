
STEP_1_PROMPT_TEMPLATE = """
# Role
You are a high-precision data synthesis generator for LLM fine-tuning.
Your task is to generate EXCLUSIVELY the INPUT CONTEXT for a given task.
In this step you do NOT generate any answers, solutions, or answer choices.

# Context & Objective
We expand a small pool of training examples through systematic variation. You
are given two existing examples (source context) and exactly one transformation
operator. Apply the operator to produce a NEW context that stays within the same
task and domain but is substantively independent in content.

# Parameters

1. TARGET TASK DESCRIPTION:
{{TASK_DESCRIPTION}}

2. TRANSFORMATION OPERATOR (apply this):
{{TRANSFORMATION_OPERATOR}}

3. SOURCE CONTEXT:
a. Example 1
{{SEED_EXAMPLE_1}} 

b. Example 2
{{SEED_EXAMPLE_2}}

4. STRUCTURAL SIGNATURE:
The SOURCE CONTEXT examples are raw excerpts in the authentic form of this
task's data — not cleaned-up or self-contained rewrites. Replicate that form
exactly as characterized below. Do NOT regularize the output into a tidy,
self-contained summary, and do NOT add framing, introductory, or concluding
material that the source form does not contain.
{{STRUCTURAL_SIGNATURE}}
   
# Goal
Generate exactly 1 new, unique context that:
- belongs to the same task and domain as the SOURCE CONTEXT,
- faithfully implements the TRANSFORMATION OPERATOR,
- is entirely new in content (not a rephrasing),
- preserves the structural requirements of the task.

# IMPORTANT CONSTRAINTS
- Do NOT generate any answer options or reference solutions.
- Do NOT output explanations or preambles.
- Do NOT regularize the output into a clean, self-contained summary; preserve
  the raw structural form of the SOURCE CONTEXT.
- Output EXCLUSIVELY the generated context text.
""".strip()
