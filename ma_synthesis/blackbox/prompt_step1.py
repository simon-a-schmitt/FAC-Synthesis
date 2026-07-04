
STEP_1_PROMPT_TEMPLATE = """
# Role
You are a high-precision data synthesis generator for LLM fine-tuning.
Your task is to generate EXCLUSIVELY the INPUT CONTEXT for a given task.
In this step you do NOT yet generate any answers, solutions, or answer choices.

# Context & Objective
We are expanding a small pool of training examples through systematic
variation. You are given an existing example (source context) along with exactly
one transformation operation. Apply the operation to produce a NEW context that
stays within the same task and subject domain but is substantively independent in
content — not a mere rephrasing of the source context.

# Parameters
These are the constraints for your generation:

1. TARGET TASK DESCRIPTION:
{{TASK_DESCRIPTION}}

2. TRANSFORMATION OPERATION:
Produce a standalone variation of the source context. Substantively alter the
concrete subject matter so that a genuinely new example of the same task and
subject domain emerges — not a mere rephrasing. Possible kinds of variation
include: a different set of facts within the same area of law, a different legal
subfield of the same domain, additional factual or legal conditions, or deeper
legal reasoning. You do not have to combine several of these — choose whatever
yields a natural, standalone new example.

3. SOURCE CONTEXT (to be transformed):
{{SEED_CONTEXT}}

4. ADDITIONAL INSTRUCTIONS:
{{ADDITIONAL_INSTRUCTIONS}}

# Goal
Generate exactly 1 new, unique context text that strictly satisfies the following criteria:
- It belongs to the same task and subject domain as the SOURCE CONTEXT.
- It implements the TRANSFORMATION OPERATION.
- It is entirely new in content (not a mere rephrasing of the source context).
- It preserves the structural requirements of the task.

# IMPORTANT CONSTRAINTS
- Do NOT generate any answer options or reference solutions.
- Do NOT output any explanations or preambles ("Here is your text...").
- Your output must consist EXCLUSIVELY of the generated context text.
""".strip()
