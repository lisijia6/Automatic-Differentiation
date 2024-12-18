from setuptools import setup, find_packages

setup(
    name="AutoDiff-Team01",
    version="1.4",
    python_requires=">=3.7",
    install_requires=[
        "numpy"
    ],
    extras_require={
        "all": [
            "sphinx", "pytest", "furo", "sphinx-copybutton", "graphviz", "pytest-cov"
        ]
    },
    packages=find_packages(),
)
