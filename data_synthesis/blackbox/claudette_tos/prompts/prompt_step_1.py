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


STEP_1_PROMPT_TEMPLATE = """
SEEDS:
{{SEED_EXAMPLES}}

Write one new terms-of-service sentence in the same register. Take the seeds
as orientation for voice, form, and surface conventions only. Write about a
different matter, for a different service.

The seeds' length does not carry over. Their balance does not carry over
either: what you write may be even-handed or one-sided, independently of the
seeds.

Output the sentence only. No numbering, no headers, no blank lines, no
commentary.
""".strip()
