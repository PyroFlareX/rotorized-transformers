# Paper draft

The paper source is `main.tex`; section files live in `sections/`, bibliography in `references.bib`, and traceable figure/table assets in their respective directories. The LaTeX manuscript is the source of equations, methodology, and paper tables; Markdown is used for logs and experiment reports.

## Build

From the repository root, run:

```sh
make -C reports/papers pdf
```

This uses `pdflatex`, `bibtex`, and TeX Live packages `amsmath`, `amssymb`, `booktabs`, `graphicx`, `hyperref`, `siunitx`, `natbib`, and `geometry`; output is written to `reports/papers/build/`. `build/` is intended to be gitignored; the repository ignore rule is being added separately.

On Ubuntu, install LaTeX support if needed:

```sh
sudo apt install texlive-latex-recommended texlive-latex-extra texlive-science
```

`latexmk` is optional and is not used by the Makefile. Results are intentionally absent until runs exist under `reports/experiments/`; never invent values to populate the paper.
