test:
	flake8 ./sneklang.py ./test_snek.py
	time tox -p all

autotest:
	ls ./sneklang.py ./test_snek.py README.rst | entr make test

.PHONY: test

dist/: setup.py sneklang.py README.rst
	python setup.py build sdist
	twine check dist/*

pypi: test dist/
	twine check dist/*
	twine upload dist/*

clean:
	rm -rf build
	rm -rf dist
