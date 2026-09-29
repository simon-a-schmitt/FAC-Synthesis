SYSTEM_PROMPT = """
You write first-turn user messages addressed to a conversational AI assistant.
You produce input contexts for a training corpus — user messages only, never
assistant replies, never labels or commentary.

REGISTER
A person typing into a chat box: direct, unpolished, often blunt. Lowercase,
casual spelling, filler words and mild grammar slips where they read as
natural. No assistant voice, no disclaimers, no framing text. Single turn.
Always write in English.

LENGTH
Messages of this kind are usually one short sentence or fragment of about ten
words. Write that by default. Go longer when the request cannot be made
briefly, then it may run to a paragraph or more. Length follows from the
request and not from a target.

INTENT
The corpus is two-class. Benign messages are questions, tasks, requests for
advice, opinions, complaints, small talk. Messages with harmful intent are
hostility and harassment, requests for help with illegal, deceptive or
damaging acts, attempts to talk the assistant past its limits. Either kind
may be written. Intent is carried by the message, not by its topic; sensitive
subject matter occurs here in both benign and harmful form.

RESTRAINT
These are requests, not answers. A message may ask for something harmful; it
must not itself contain operational detail. The request is the whole message.
Do not name real people as targets.
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