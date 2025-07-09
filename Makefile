default: web
clean:
	rm -rf _build
book:
	jupyter-book build --builder pdflatex .
web:
	jupyter book build --html
serve:
	jupyter book start