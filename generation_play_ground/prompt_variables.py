TARGET_TASK_DESCRIPTION = (
    "Die Aufgabe besteht darin, einen realistischen Textausschnitt aus einer US-Gerichtsentscheidung (einen sogenannten \"citing context\") zu generieren. Dieser Text beschreibt einen rechtlichen Sachverhalt oder die Argumentation eines Gerichts. Der Text muss an einer strategischen Stelle abbrechen – genau dort, wo ein verbindlicher rechtlicher Leitsatz (eine \"Holding\") eines anderen Falls zitiert wird. Eine Holding repräsentiert die maßgebliche Rechtsregel, wenn das Recht auf einen bestimmten Sachverhalt angewendet wird."
)

TARGET_FEATURE_DESCRIPTION = (
    "Sprachliche Muster und Ausdruecke, die dazu dienen, Entitaeten, Konzepte oder Faelle voneinander abzugrenzen, Unterschiede hervorzuheben oder Alleinstellungsmerkmale zu identifizieren (wie \"distinguishes itself from\", \"set Elkhorn apart from\" oder \"How is it different from\")."
)

TARGET_FEATURE_TEXT_SPANS = (
    "Span 1: nRainbow River in Florida is different from\n"
    "Span 2: nThe Saint Lambert March distinguishes itself from\n"
    "Span 3:  architectural style distinguishes Ghadamès from\n"
    "Span 4: nOne aspect that set Elkhorn apart from\n"
    "Span 5: cks on their surface. Lighter in color than\n"
    "Span 6: azzano.\nSo what sets this apart from\n"
    "Span 7: : How is the Chevy Volt different than\n"
    "Span 8:  are a unique feature that sets it apart from many\n"
    "Span 9:  Richard Nelson sees NASA as significantly different when compared to\n"
    "Span 10:  in the world.\nHow is it different from"
)

STEP_1_TASK_EXAMPLE_PROMPT = (
    "was unable to establish a foundation for his federal claims because he could not demonstrate Officer Ahlm’s conduct violated a constitutional right. See Grubbs v. Bailes, 445 F.3d 1275, 1278 (10th Cir.2006). The conclusion was threefold. First, Mr. Titus’s malicious prosecution claim failed because “Officer Ahlm possessed probable cause to believe that Mr. Titus had been operating his vehicle while intoxicated to the slightest degree,” even after determining Mr. Titus had only a .02% breath alcohol concentration (BAC). Aplt. App. at 131. Second, Mr. Titus’s retaliatory prosecution claim failed because Mr. Titus did not plead and prove the absence of probable cause for charging him with DWI. See id. at 137; Hartman v. Moore, 547 U.S. 250, 265-66, 126 S.Ct. 1695, 164 L.Ed.2d 441 (2006) (<HOLDING>)"
)

STEP_1_TASK_EXAMPLE_SOLUTION = "A"
STEP_1_WEITERE_ANWEISUNG = "Füge an der Stelle im Text an der das Holding stehen muss das Tag (<HOLDING>) ein."

STEP_1_MAX_NEW_TOKENS = 2048
STEP_1_TEMPERATURE = 0.5

STEP_2_MAX_NEW_TOKENS = 2048
STEP_2_TEMPERATURE = 0.5

STEP_3_MAX_NEW_TOKENS = 2048
STEP_3_TEMPERATURE = 0.5

STEP_4_MAX_NEW_TOKENS = 2048
STEP_4_TEMPERATURE = 0.5
