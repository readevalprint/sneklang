test:
	python -V
	flake8 ./sneklang.py ./test_snek.py
	time coverage run --rcfile=setup.cfg  `which pytest` --doctest-modules  --doctest-glob='*.rst'
	coverage annotate --rcfile=setup.cfg
	coverage report --rcfile=setup.cfg
	coverage html --rcfile=setup.cfg

autotest:
	ls *.py *.rst | entr make test

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
