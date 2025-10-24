"""
Setup script for Drug-Target Affinity Prediction project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dta-prediction",
    version="1.0.0",
    author="DTA Prediction Team",
    author_email="team@dta-prediction.com",
    description="Drug-Target Affinity prediction using GNN, ESM-2, and cross-attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/dta-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "monitoring": [
            "wandb>=0.13.0",
            "tensorboard>=2.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dta-train=main:main",
        ],
    },
)
