"""Central store for shared prompt building blocks used across all pipeline steps and paths."""

SYSTEM_PROMPT = """
You follow the user instructions exactly and output only the requested content.
""".strip()




### STEP 1

TASK_DESCRIPTION = (
    "This task involves identifying and completing legal principles (known as \"holdings\") within US court decisions. The provided text describes the context of a case, which cuts off at a point where a binding precedent is cited (indicated by a placeholder). A \"holding\" represents the authoritative rule of law applied when the law is related to these specific facts. The objective for the person performing the task is to determine the single legally and logically correct holding for this context."
)

ADDITIONAL_INSTRUCTIONS = (
    "At the position in the text where the holding must appear, insert the tag (<HOLDING>)."
)


### STEP 2

FORMAT_INSTRUCTION = (
    "Generate a single, complex legal holding statement that begins with the word 'holding' (e.g., 'holding that...'). It must logically and grammatically complete the sentence where the (<HOLDING>) placeholder is located."
    "Do NOT include the surrounding parentheses in your output—just the text of the holding itself."
)

TARGET_REFERENCE = (
    "holding a mistake or surprise in connection with the sale is grounds for setting it aside provided the mistake is harmful and not a mistake of law or one due to the negligence of the party complaining"
)


### STEP 3

DISTRACTOR_INSTRUCTION = (
    "Generate exactly 4 incorrect holding statements. They must be written in the same dense legal style, start with 'holding that', and deal with legal concepts, but they must be incorrect or irrelevant to the specific facts discussed in the context."
)

### STEP 4

INTRODUCTION_CLAUSE = (
    "Identify the single correct legal holding statement from options A-E to fill the <HOLDING> placeholder in the citation and output only the corresponding letter."
)
