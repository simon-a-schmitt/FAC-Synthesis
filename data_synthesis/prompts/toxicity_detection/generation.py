"""Generation prompts of the toxicity_detection domain (first-turn user messages to a chat assistant).

SYSTEM_PROMPT is shared by every generation arm (blackbox, feature-guided, hybrid). The user prompt
is BLACKBOX_TEMPLATE in the blackbox arm / hybrid phase 1 and FEATURE_GUIDED_TEMPLATE in the
feature-guided arm / hybrid phase 2. Placeholders: {{SEED_EXAMPLES}} (numbered context examples),
{{FEATURE_EXPLANATION}} / {{FEATURE_SPANS}} (target SAE feature, feature-guided only).
"""

SYSTEM_PROMPT = """
You write first-turn user messages addressed to a conversational AI assistant.
You produce input contexts for a training corpus. User messages only, never
assistant replies, labels, or commentary.

REGISTER
A person typing into a chat box: direct, unpolished, often blunt. Lowercase,
casual spelling, filler words and mild grammar slips where they read as
natural. No assistant voice, no disclaimers, no framing text. Single turn.
Always write in English.

LENGTH
Typically one short sentence or fragment of about ten words. Longer messages,
up to a paragraph or more, occur where the request cannot be made briefly.
Length follows from the request, not from a target.

INTENT
Two kinds of message occur here. They differ in what the message is meant to
achieve.

Benign messages are questions, tasks, requests for advice, opinions,
complaints and small talk. They are the majority.

Messages with harmful intent are hostility and harassment, requests for help
with illegal, deceptive or damaging acts, and attempts to talk the assistant
past its limits. They are a minority.

Intent follows from what a message asks for or does, not from the topic it
addresses. Sensitive subject matter occurs in both forms.

RESTRAINT
These are requests, not answers. The request is the whole message. A message
may ask for something harmful, but it does not itself contain operational
detail. Do not name real people as targets.
""".strip()


BLACKBOX_TEMPLATE = """
EXAMPLES:
{{SEED_EXAMPLES}}

Write one new user message in the same register. Take the examples as
orientation for voice and form only. Write about something else, and want
something else. Write in English.

The examples' length does not carry over. Their intent does not carry over
either: what you write may be benign or may carry harmful intent,
independently of the examples.

Output the message only. No numbering, no headers, no blank lines, no
commentary.
""".strip()


FEATURE_GUIDED_TEMPLATE = """
EXAMPLES:
{{SEED_EXAMPLES}}

TARGET PATTERN:
{{FEATURE_EXPLANATION}}

Places where the pattern occurs. Each excerpt ends where the pattern is
strongest, so its last word carries the peak:
{{FEATURE_SPANS}}

Write one new user message in the same register. Take the examples as
orientation for voice and form only. Write about something else, and want
something else. Write in English.

The message must contain a clear instance of the target pattern, carried by
what the message does rather than placed at its edge. The excerpts come from
general text: carry the pattern over into a user message. Do not take over
their topics, and do not copy them. Use only as much of their wording as the
pattern itself requires. Where the pattern and the examples pull in different
directions, the pattern decides the content and the examples decide voice and
form.

The examples' length does not carry over. Their intent does not carry over
either: what you write may be benign or may carry harmful intent,
independently of the examples. The target pattern does not decide the intent.

Output the message only. No numbering, no headers, no blank lines, no
commentary.
""".strip()
