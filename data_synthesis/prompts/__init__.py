"""All prompts of data_synthesis, one folder per domain:

  prompts/<domain>/generation.py  SYSTEM_PROMPT (shared by all arms), BLACKBOX_TEMPLATE, FEATURE_GUIDED_TEMPLATE
  prompts/<domain>/labeling.py    SYSTEM_PROMPT, LABEL_FRAGMENTS, optional USER_PROMPT_PREFIX - the
                                  classification prompt, used for labeling AND for every SAE check

labeling.py has no imports, so scripts outside data_synthesis/ can load it by path as well.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationPrompts:
    system: str
    blackbox_template: str
    feature_guided_template: str


@dataclass(frozen=True)
class LabelingPrompt:
    system: str
    fragments: dict
    user_prefix: str  # prepended to the text in the user turn (e.g. cti_vsp's "CVE Description: ")

    def user_content(self, text: str) -> str:
        return f"{self.user_prefix}{text}"


def load_generation_prompts(domain: str) -> GenerationPrompts:
    module = importlib.import_module(f"prompts.{domain}.generation")
    return GenerationPrompts(module.SYSTEM_PROMPT, module.BLACKBOX_TEMPLATE, module.FEATURE_GUIDED_TEMPLATE)


def load_labeling_prompt(domain: str) -> LabelingPrompt:
    module = importlib.import_module(f"prompts.{domain}.labeling")
    return LabelingPrompt(module.SYSTEM_PROMPT, module.LABEL_FRAGMENTS, getattr(module, "USER_PROMPT_PREFIX", ""))
