# latexmk -e "$pdflatex=q/pdflatex -synctex=1 -interaction=nonstopmode/" -pdf %.tex
pdflatex -output-directory=build -synctex=1 -interaction=nonstopmode %.tex
cd build/; biblatex p9_tiefenauer.aux; cd ..
pdflatex -output-directory=build -synctex=1 -interaction=nonstopmode %.tex
pdflatex -output-directory=build -synctex=1 -interaction=nonstopmode %.tex
