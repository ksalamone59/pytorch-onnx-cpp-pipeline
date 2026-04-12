.SECONDEXPANSION:
.PRECIOUS: %.dat

TARGETS := $(patsubst %/,%,$(sort $(dir $(wildcard */*.gnu))))
TARGETS := $(filter-out style,$(TARGETS))

all: $(addprefix pdfs/,$(addsuffix .pdf,$(TARGETS)))

pdfs/%.pdf: %/*.gnu style/*.gnu fix.py $$(wildcard %/*.gp) $$(wildcard %/*.dat) $$(wildcard %/*.txt)
	$(eval GNU_FILE := $(shell find $* -name "*.gnu" -type f))
	$(eval BASE := $(basename $(notdir $(GNU_FILE))))
	(python3 fix.py $*/$(BASE))
	mv $*/$(BASE).pdf pdfs/.

.PHONY: all clean $(TARGETS)
$(TARGETS): %: pdfs/%.pdf

clean:
	rm -f */*.pdf */*.tex */*.eps */*.auc */*.aux */*.log */*latexmk */*.fls */*.gz */*.sty