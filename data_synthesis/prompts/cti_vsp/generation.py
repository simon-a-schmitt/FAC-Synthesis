"""Generation prompts of the cti_vsp domain (CVE vulnerability descriptions).

SYSTEM_PROMPT is shared by every generation arm (blackbox, feature-guided, hybrid). The user prompt
is BLACKBOX_TEMPLATE in the blackbox arm / hybrid phase 1 and FEATURE_GUIDED_TEMPLATE in the
feature-guided arm / hybrid phase 2. Placeholders: {{SEED_EXAMPLES}} (numbered context examples),
{{FEATURE_EXPLANATION}} / {{FEATURE_SPANS}} (target SAE feature, feature-guided only).
"""

SYSTEM_PROMPT = """
You write vulnerability descriptions as they are published in public CVE
records. You produce input contexts for a training corpus. Description text
only, never scores, severity ratings, or commentary.

REGISTER
Terse and factual, as published in vulnerability records. No marketing
language, no hedging, no meta-commentary. Version qualifiers are concrete and
phrased as records phrase them, for example "through 6.7.1", "prior to 4.2.3"
or "in versions 1.0.0 through 1.4.2", never as placeholders. Always write in
English.

LENGTH
Typically one or two sentences of a few dozen words. Shorter descriptions and
longer ones, where the mechanism takes more words to state, also occur.
Length follows from the defect being described, not from a target.

CONTENT
Descriptions name specific software. This ranges from major vendors and
mainline projects to small vendors, CMS plugins, device firmware, niche
libraries and industrial controllers.

Descriptions differ in granularity. Some name a source file and function,
others only a component, endpoint or parameter.

The vulnerability class is one that can occur in the named component, and it
is an actual security defect, not a configuration choice.

CONDITIONS AND IMPACT
Some descriptions state the conditions under which the defect can be
exploited: who can reach the component, what privileges are needed, whether
a user must act first, or how it is reachable over a network. Others leave
these conditions implicit. The same holds for consequences. Some descriptions
name what an attacker can achieve, others describe only the defect. Both
forms are normal.

RESTRAINT
One description states one defect with one mechanism. A plain, technically
coherent description is preferable to an elaborate one. Do not add
background, remediation advice or explanation. Do not use the vocabulary of
vulnerability scoring, such as severity ratings or rating categories. Do not
invent CVE identifiers, database IDs or advisory references.
""".strip()


BLACKBOX_TEMPLATE = """
EXAMPLES:
{{SEED_EXAMPLES}}

Write one new vulnerability description in the same register. Take the
examples as orientation for voice and form only. Write about a different
vendor, product, component and vulnerability class.

The examples' length does not carry over. Whether they state exploitation
conditions or consequences does not carry over either: what you write may
state them or leave them implicit, independently of the examples.

Output the description only. No numbering, no headers, no blank lines, no
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

Write one new vulnerability description in the same register. Take the
examples as orientation for voice and form only. Write about a different
vendor, product, component and vulnerability class.

The description must contain a clear instance of the target pattern, carried
by the vulnerability it describes rather than placed at its edge. The excerpts
come from general text: carry the pattern over into a vulnerability
description. Do not take over their topics, and do not copy them. Use only as
much of their wording as the pattern itself requires. Where the pattern and
the examples pull in different directions, the pattern decides the content
and the examples decide voice and form.

The examples' length does not carry over. Whether they state exploitation
conditions or consequences does not carry over either: what you write may
state them or leave them implicit, independently of the examples.

Output the description only. No numbering, no headers, no blank lines, no
commentary.
""".strip()
