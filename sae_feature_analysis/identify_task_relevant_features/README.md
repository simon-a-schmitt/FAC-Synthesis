# Identify Task-Relevant Features

This directory contains scripts for **identifying task-relevant SAE features** via task-conditioned annotation.  

## Files

- `OpenaiAPI.py`  
  A lightweight OpenAI API wrapper used by the annotation scripts.

- `annotate_toxicity.py`  
  Identifies **toxicity-related** features by collecting representative activation examples and assigning toxicity-oriented annotations.

- `annotate_helpfulness.py`  
  Identifies **helpfulness-related** features by annotating high-activation examples under helpfulness criteria.

- `annotate_steering_sycophancy.py`  
  Identifies **sycophancy-related** steering features by annotating examples that reflect agreement-seeking or user-flattering tendencies.

- `annotate_steering_survival.py`  
  Identifies **survival-instinct** steering features by annotating examples associated with self-preservation or self-protective behaviors.

- `annotate_instruction_following.py`  
  Identifies **instruction-following** features by labeling activation examples that correlate with instruction compliance behaviors.

## Notes on Prompts

All task-specific prompts used for feature annotation are specified in the paper.  
To ensure consistency, we apply the **same annotation protocol** across tasks, with only the task definition and labeling criteria changed according to the corresponding experimental settings in the manuscript.
