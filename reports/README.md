# Reports

Research outputs for the Rotor Transformer project. Subdirectories:

- `templates/`: reusable Markdown report templates and their usage. See `templates/README.md`.
- `experiments/`: uniquely named experiment run directories. See `experiments/README.md`.
- `figures/`: generated plots and the scripts that produce them. See `figures/README.md`.
- `papers/`: LaTeX research drafts. See `papers/README.md`.
- `literature/`: reading notes and topic lists. See `literature/README.md`.

## Ground rules

- Never invent results. Every number must come from a recorded run or be labeled `estimate`.
- Report negative and failed runs as well as successful ones.
- Compare structural compression against truncated-SVD baselines at equal stored bytes. Report compute separately.
