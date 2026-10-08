BUILD_DIR := build
LATEX := pdflatex
BIBTEX := bibtex

.PHONY: pdf clean

pdf:
	mkdir -p $(BUILD_DIR)
	$(LATEX) -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD_DIR) main.tex
	cd $(BUILD_DIR) && BIBINPUTS=..: $(BIBTEX) main
	$(LATEX) -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD_DIR) main.tex
	$(LATEX) -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD_DIR) main.tex

clean:
	rm -rf $(BUILD_DIR)
