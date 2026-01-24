# Feature Interpretation

This folder contains the scripts used to **interpret SAE features** by automatically annotating model-activated text spans with human-readable explanations.

## Files

- `OpenaiAPI.py`  
  A lightweight OpenAI API wrapper used by the annotation scripts (e.g., for calling GPT models).

- `annotate_explanations.py`  
  Generates **natural-language interpretations** for SAE features based on extracted activation spans (i.e., “what this feature represents / triggers on”).

- `groupby_textspans.py`  
  Groups and post-processes extracted activation spans.
