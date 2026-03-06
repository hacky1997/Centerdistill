from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [
        line.strip() for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name            = "centerdistill",
    version         = "1.0.0",
    author          = (
        "Somyajit Chakraborty, Sayak Naskar, Soham Paul, "
        "Angshuman Jana, Nilotpal Chakraborty, Avijit Gayen"
    ),
    description     = (
        "Weakly-Supervised Distillation for Ambiguity-Aware Cross-Lingual QA"
    ),
    long_description       = long_description,
    long_description_content_type = "text/markdown",
    url             = "https://github.com/your-username/centerdistill",
    packages        = find_packages(),
    python_requires = ">=3.9",
    install_requires= requirements,
    classifiers     = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
