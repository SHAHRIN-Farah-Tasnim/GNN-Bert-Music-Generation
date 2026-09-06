# Project rules

- Windows + PowerShell only. Never use `&&`, `export`, or forward-slash paths in commands.
- Always use `.venv\Scripts\python.exe`. Never install into system Python.
- Python 3.11. Torch is CPU-only here; training runs on Colab.
- Never write a metric, F1 score, or accuracy number into README, comments,
  or the report unless it came from an actual run. Write TODO instead.
- Read config.yaml for all hyperparameters. Do not hardcode them.
- Set and log the random seed from config.yaml in every script.
- Keep files in the existing src/ structure. Do not reorganize the project.
- Make small, focused changes. One file at a time.
