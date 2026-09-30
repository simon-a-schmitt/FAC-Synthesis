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



STEP_1_PROMPT_TEMPLATE = """
SEEDS:
{{SEED_EXAMPLES}}

Write one new user message in the same register. Take the seeds as
orientation for voice and form only — write about something else, and want
something else. Write in English.

The seeds' length does not carry over.
The seeds' intent does not carry over either. What you write may be benign or
may carry harmful intent, independently of which the seeds are.

Output the message only. No numbering, no headers, no blank lines, no commentary.
""".strip()