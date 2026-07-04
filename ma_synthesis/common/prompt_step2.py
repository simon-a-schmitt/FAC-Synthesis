STEP_2_MAX_NEW_TOKENS = 512
STEP_2_TEMPERATURE = 0.3

# TODO: Refine this template once the correct holding generation strategy is finalised.
STEP_2_PROMPT_TEMPLATE = """
# Role
You are a highly precise example labeler for LLM fine-tuning. 
Your task is to generate the single true and logically correct model solution (the target label) for a given context.

# Context
You will receive an incomplete text or a text to be answered (context) that was generated in the previous step. 
Your task is to logically, precisely, and flawlessly complete or answer this text based on the rules of the target task.

# Parameters
Here are the constraints for your generation:

1. TARGET TASK DESCRIPTION:
{{TASK_DESCRIPTION}}

2. GENERATED CONTEXT:
{{CITING_CONTEXT}}

3. FORMAT AND STYLE INSTRUCTION FOR THE SOLUTION:
{{FORMAT_INSTRUCTION}}

4. TARGET LENGTH AND DENSITY REFERENCE:
Use the following example strictly as an anchor for the expected text length, conciseness, and token 
density. Do NOT copy any content or words from this reference; only mimic its brevity 
and structure:
{{TARGET_REFERENCE}}

# Goal
Generate the exact and correct model solution for the context provided above. The solution must:
- Exhibit the highest possible quality of content and logical correctness.
- Exactly match the format required in the "FORMAT AND STYLE INSTRUCTION FOR THE SOLUTION".

# IMPORTANT CONSTRAINTS
- Do NOT provide any explanations, justifications, or introductions ("The correct solution is...").
- Do NOT generate any false alternatives or distractors.
- Your output must consist EXCLUSIVELY of the bare model solution / the final label.
""".strip()
