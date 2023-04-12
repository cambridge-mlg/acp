import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="approx-cp",
    version="0.0.1",
    description="Python implementation of Approximate full Conformal Prediction (ACP)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cambridge-mlg/acp",
    author="Javier Abad",
    author_email="javier.abadmartinez@ai.ethz.ch",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    keywords="machine learning, AAAI, conformal prediction",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "torchaudio",
        "pandas",
        "tqdm",
        "matplotlib",
        "keras",
        "folktables",
        "scipy",
        "scikit-learn",
        "seaborn",
    ],
    python_requires=">=3",
    project_urls={
        "Paper": "https://arxiv.org/abs/2202.01315",
    },
)
