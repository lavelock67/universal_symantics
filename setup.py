"""Setup script for the Periodic Table of Information Primitives."""

from setuptools import setup, find_packages

setup(
    name="periodic-primitives",
    version="0.1.0",
    description="Periodic Table of Information Primitives - Cross-modal primitive discovery and integration",
    author="Periodic Primitives Team",
    author_email="team@periodic-primitives.org",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "torch>=1.9.0",
        "transformers>=4.15.0",
        "networkx>=2.6.0",
        "rdflib>=6.0.0",
        "requests>=2.25.0",
        "tqdm>=4.62.0",
        "click>=8.0.0",
        "pydantic>=1.8.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "plotly>=5.0.0",
        "jupyter>=1.0.0",
        "ipywidgets>=7.6.0",
    ],
)
