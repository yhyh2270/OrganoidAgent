PDF=main.pdf

.PHONY: all clean

all: $(PDF)

$(PDF): main.tex
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f main.aux main.log main.out main.toc main.pdf
