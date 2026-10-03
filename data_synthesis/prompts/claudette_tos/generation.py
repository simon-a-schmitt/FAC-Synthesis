"""Generation prompts of the claudette_tos domain (terms-of-service sentences).

SYSTEM_PROMPT is shared by every generation arm (blackbox, feature-guided, hybrid). The user prompt
is BLACKBOX_TEMPLATE in the blackbox arm / hybrid phase 1 and FEATURE_GUIDED_TEMPLATE in the
feature-guided arm / hybrid phase 2. Placeholders: {{SEED_EXAMPLES}} (numbered context examples),
{{FEATURE_EXPLANATION}} / {{FEATURE_SPANS}} (target SAE feature, feature-guided only).
"""

SYSTEM_PROMPT = """
You write single sentences from the terms of service of online platforms and
consumer services. You produce input contexts for a training corpus. Clause
text only, never labels, categories, or commentary.

REGISTER
Contractual prose as published: declarative, impersonal, often subordinated,
with defined terms ("the services", "content", "these terms") and references
to other sections, to a privacy policy, or to a URL. The parties are "we" and
"you", or the provider is named. Not every line is a complete sentence. Some
are list items, headings, or questions. Follow the surface form of the
examples given: lowercase throughout, punctuation and quotation marks spaced
as they appear there. Always write in English.

LENGTH
One sentence, typically a couple of dozen words. Short fragments and long
single sentences carrying several clauses also occur. Length follows from
what the sentence states, not from a target.

CONTENT
A terms document contains machinery as well as provisions. It defines terms,
describes what the service is and how an account works, explains features and
settings, states fees and renewal mechanics, gives notice addresses and
contact details, incorporates other documents by reference, sets out the steps
of a procedure, and carries headings, cross references and boilerplate that
grants and reserves nothing. It also addresses liability, termination,
amendment, user content and dispute resolution.

Sentences name the provider, jurisdictions, courts or institutions where a
real clause would name them.

BALANCE
Two kinds of sentence occur here. They differ in how rights, risk and
procedure are distributed between provider and user.

Even-handed sentences state how the service operates, define something, grant
the user a right, set out a step either party may take, or place an obligation
that also binds the provider. They are the majority.

One-sided sentences move rights, risk or procedure towards the provider and
away from the user. They are a minority.

Balance follows from what a sentence grants, reserves, excludes or imposes,
not from the topic it addresses. The same topic occurs in both forms.

RESTRAINT
One sentence states one provision. Do not explain, justify or soften the
clause, and do not invent statutes or case numbers.
""".strip()


BLACKBOX_TEMPLATE = """
EXAMPLES:
{{SEED_EXAMPLES}}

Write one new terms-of-service sentence in the same register. Take the
examples as orientation for voice, form, and surface conventions only. Write
about a different matter, for a different service.

The examples' length does not carry over. Their balance does not carry over
either: what you write may be even-handed or one-sided, independently of the
examples.

Output the sentence only. No numbering, no headers, no blank lines, no
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

Write one new terms-of-service sentence in the same register. Take the
examples as orientation for voice, form, and surface conventions only. Write
about a different matter, for a different service.

The sentence must contain a clear instance of the target pattern, carried by
what the sentence states rather than placed at its edge. The excerpts come
from general text: carry the pattern over into a terms-of-service sentence.
Do not take over their topics, and do not copy them. Use only as much of
their wording as the pattern itself requires. Their casing and punctuation do
not carry over. Where the pattern and the examples pull in different
directions, the pattern decides the content and the examples decide voice,
form, and surface conventions.

The examples' length does not carry over. Their balance does not carry over
either: what you write may be even-handed or one-sided, independently of the
examples. The target pattern does not decide the balance.

Output the sentence only. No numbering, no headers, no blank lines, no
commentary.
""".strip()
