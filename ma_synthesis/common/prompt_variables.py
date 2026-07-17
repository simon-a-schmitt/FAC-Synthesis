"""Central store for shared prompt building blocks used across all pipeline steps and paths."""

SYSTEM_PROMPT = """
You follow the user instructions exactly and output only the requested content.
""".strip()




### STEP 1

TASK_DESCRIPTION = (
    "This task involves identifying and completing legal principles (known as \"HOLDING\") within US court decisions. The provided text describes the context of a case, which cuts off at a point where a binding precedent is cited (indicated by a placeholder). A \"HOLDING\" represents the authoritative rule of law applied when the law is related to these specific facts. The objective for the person performing the task is to determine the single legally and logically correct HOLDING for this context."
)

ADDITIONAL_INSTRUCTIONS = (
    "At the position in the text where the holding must appear, insert the tag (<HOLDING>)."
)

OPERATORS = {
    "breadth": {
        "description": "Create a brand-new context in the SAME domain as the source, but covering a rarer or less common sub-area of that domain. Invent an entirely new scenario with new entities and specifics. Keep length and register similar.",
        "weight": 0.60
    },
    "constraints": {
        "description": "Add one or two additional factual, procedural, or conditional elements to the scenario that make the situation richer, while keeping it realistic.",
        "weight": 0.10
    },
    "deepening": {
        "description": "Increase the depth of the scenario: introduce a more layered or nested chain of facts/reasoning leading up to the point the task hinges on.",
        "weight": 0.10
    },
    "concretizing": {
        "description": "Replace generic entities, quantities, and references in the scenario with more specific, concrete ones appropriate to the domain.",
        "weight": 0.10
    },
    "reasoning": {
        "description": "Reshape the scenario so that arriving at the correct answer plausibly requires several distinct inferential steps rather than one.",
        "weight": 0.10
    }
}


STRUCTURAL_SIGNATURE = (
    "- Approximately 100–160 words, matching the length of the source excerpts"
    "- Begins mid-sentence (lowercase) and ends mid-sentence, as a window cut from a longer opinion."
    "- The (<HOLDING>) tag sits inside the ongoing analysis; substantive legal text that depends on the holding continues after it."
    "- Dense appellate register: string citations with reporters/pincites, parentheticals, embedded quotations. No narrative story-summary."
    "- No sentence that signals a holding is missing."
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
    "Generate exactly 4 incorrect holding statements. Each must concern the same "
    "legal doctrine and area of law as the true label, start with 'holding that', "
    "and match its dense legal style and length. Make each plausible to a skimming "
    "reader but clearly wrong on careful analysis of these specific facts, and make "
    "each wrong for a different reason (e.g. opposite outcome, wrong legal standard, "
    "an overstated per-se version of a real rule, or an inapplicable adjacent rule). "
    "Do NOT switch to an unrelated area of law, and do NOT paraphrase the true label."
)

### STEP 4

INTRODUCTION_CLAUSE = (
    "Identify the single correct legal holding statement from options A-E to fill the <HOLDING> placeholder in the citation and output only the corresponding letter."
)
