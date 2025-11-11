"""
Setup script для Alpha RAG System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Чтение README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Чтение requirements
requirements = (this_directory / "requirements.txt").read_text(encoding='utf-8').splitlines()

setup(
    name="alpha-rag-system",
    version="1.0.0",
    author="Neuro Bureau",
    author_email="team@neurobureau.com",
    description="Intelligent Document Retrieval System for Alfa-Bank",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neurobureau/alpha-rag-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.7.0',
            'ruff>=0.0.285',
            'jupyter>=1.0.0',
            'ipython>=8.14.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'alpha-build-indices=scripts.build_indices:main',
            'alpha-run-retrieval=scripts.run_retrieval:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
