STEP_3_MAX_NEW_TOKENS = 1024
STEP_3_TEMPERATURE = 0.7

# TODO: Refine this template once the distractor generation strategy is finalised.
STEP_3_PROMPT_TEMPLATE = """
# Role
You are a highly precise dataset generator for LLM fine-tuning. 
Your task is to generate plausible but strictly incorrect alternative options 
(distractors) for a given context and its correct solution.

# Context
You will receive a text (context) and the single true correct solution (target label). 
Your task is to generate a specific number of incorrect alternatives that look highly 
realistic, match the style and domain, but are logically or legally wrong for this 
specific situation.

# Parameters
Here are the constraints for your generation:

1. TARGET TASK DESCRIPTION:
{{TASK_DESCRIPTION}}

2. GENERATED CONTEXT:
{{GENERATED_CONTEXT}}

3. TRUE LABEL:
{{TRUE_LABEL}}

4. DISTRACTOR GENERATION INSTRUCTIONS:
{{DISTRACTOR_INSTRUCTION}}

# Goal
Generate the exact number of incorrect alternatives as specified. Every single distractor must:
- Be highly deceptive and plausible to someone who only skims the text.
- Be clearly and definitively incorrect upon deep logical/legal analysis of the provided context.
- Maintain a similar length, tone, and complexity as the "TRUE LABEL".

# IMPORTANT CONSTRAINTS
- Do NOT label them as A, B, C, D yet. Just output them as a clean list (one per line).
- Do NOT provide any explanations, justifications, or introductions.
- Do NOT include the "TRUE LABEL" in this list.
- Your output must consist EXCLUSIVELY of the raw incorrect alternatives.
""".strip()
