SYSTEM_PROMPT = """
Du befolgst die Benutzeranweisungen exakt und gibst nur den angeforderten Inhalt aus.
""".strip()


STEP_1_PROMPT_TEMPLATE = """
# Rolle
Du bist ein hochpraeziser Daten-Synthese-Generator fuer das LLM-Fine-Tuning.
Deine Aufgabe ist es, ausschliesslich den INPUT-KONTEXT fuer eine bestimmte Aufgabe zu
generieren. Du generierst in diesem Schritt NOCH KEINE Antworten, Loesungen oder
Auswahloptionen.

# Kontext & Ziel
Wir generieren Trainingsdaten, die ein spezifisches neuronales Aktivierungsmuster
(Ziel-Feature) in einem KI-Modell triggern sollen. Dafuer musst du ein vorgegebenes
abstraktes, konzeptionelles Muster organisch und syntaktisch korrekt in einen neu
generierten Text einbetten.

# Parameter
Hier sind die Rahmenbedingungen fuer deine Generierung:

1. ZIEL-TASK BESCHREIBUNG:
{{TASK_BESCHREIBUNG}}

2. ZIEL-FEATURE BESCHREIBUNG:
{{FEATURE_BESCHREIBUNG}}

3. ZIEL-FEATURE TEXT SPANS:
Diese Textausschnitte sind Beispiele aus einem allgemeinen Textkorpus, bei denen das
Ziel-Feature extrem stark aktiviert wird. Analysiere diese Spans, um das abstrakte
sprachliche, semantische oder logische Muster hinter dem Feature zu verstehen.
{{TEXT_SPANS}}

4. STIL- UND FORMAT-REFERENZ:
Nutze dieses Beispiel als Referenz fuer die uebergeordnete Fachdomaene.
- BEIBEHALTEN: Die Tonalitaet, den Fachjargon, die Komplexitaet und die ungefaehre Textlaenge dieser Domaene.
- STRUKTURELL AENDERN: Der konkrete Sachverhalt, die konkreten Inhalte aus der Referenz.
{{TASK_BEISPIEL}}

5. WEITERE ANWEISUNGEN:
{{WEITERE_ANWEISUNG}}

# Ziel
Generiere genau 1 neuen, einzigartigen Kontext-Text, der folgende Kriterien strikt erfuellt:
- Er orientiert sich stilistisch perfekt an der "Stil- und Format-Referenz".
- Er ist inhaltlich komplett neu (kein blosses Umformulieren des Beispiels).
- Er ist so konstruiert, dass er das Ziel-Feature stark aktiviert, indem er das aus ZIEL-FEATURE BESCHREIBUNG und ZIEL-FEATURE TEXT SPANS abgeleitete abstrakte Prinzip praezise umsetzt.

# WICHTIGE EINSCHRAENKUNGEN
- Generiere KEINE Antwortoptionen oder Musterloesungen.
- Gib KEINE Erklaerungen oder Einleitungen ("Hier ist dein Text...") aus.
- Deine Ausgabe darf AUSSCHLIESSLICH der generierte Kontext-Text sein.
- Falls du Textstrukturen oder Optionen aus der Referenz kopierst, ist das ein fataler Fehler.
""".strip()


STEP_2_PROMPT_TEMPLATE = """TODO: Schritt-2-Template einfuegen."""
STEP_3_PROMPT_TEMPLATE = """TODO: Schritt-3-Template einfuegen."""
STEP_4_PROMPT_TEMPLATE = """TODO: Schritt-4-Template einfuegen."""
