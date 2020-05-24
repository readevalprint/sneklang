from setuptools import setup

__version__ = "0.5.2"

setup(
    name="sneklang",
    py_modules=["sneklang"],
    version=__version__,
    description="Experimental minimal subset of Python for safe evaluation",
    long_description=open("README.rst", "r").read(),
    long_description_content_type="text/x-rst",
    author="Timoth Watts",
    author_email="tim@readevalprint.com",
    url="https://github.com/readevalprint/sneklang",
    keywords=["sandbox", "parse", "ast"],
    test_suite="test_snek",
    use_2to3=True,
    install_requires=["pytest", "ConfigArgParse", "python-forge"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python",
    ],
)
