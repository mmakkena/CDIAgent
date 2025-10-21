"""
Setup configuration for the CDI RAG system package.

This allows the package to be installed via pip:
    pip install -e .          # Development/editable install
    pip install .             # Regular install
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_file(filename):
    """Read file contents."""
    with open(os.path.join(os.path.dirname(__file__), filename), encoding='utf-8') as f:
        return f.read()


# Read requirements from requirements.txt
def read_requirements(filename='requirements.txt'):
    """Read requirements from file."""
    try:
        with open(filename, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        # Fallback to hardcoded requirements if file doesn't exist
        return [
            'torch>=2.0.0',
            'transformers>=4.30.0',
            'accelerate>=0.20.0',
            'langchain>=0.1.0',
            'langchain-community>=0.0.20',
            'langchain-core>=0.1.23',
            'langchain-chroma>=0.1.0',
            'langchain-huggingface>=0.0.1',
            'langchain-classic>=0.0.1',
            'chromadb>=0.4.0',
            'sentence-transformers>=2.2.0',
            'scikit-learn>=1.3.0',
            'numpy>=1.24.0',
            'faiss-cpu>=1.7.4',
            'pydantic>=2.0.0',
            'fastapi>=0.100.0',
            'uvicorn>=0.23.0',
        ]


setup(
    # Package metadata
    name='cdi-rag',
    version='2.0.0',
    author='CDI Team',
    author_email='cdi@example.com',
    description='Clinical Documentation Integrity (CDI) Retrieval-Augmented Generation System',
    long_description=read_file('README.md') if os.path.exists('README.md') else 'CDI RAG System',
    long_description_content_type='text/markdown',
    url='https://github.com/your-org/cdi-rag',  # Update with actual URL

    # Package configuration
    packages=find_packages(exclude=['tests', 'tests.*', 'docs']),
    package_data={
        'cdi_rag': ['data/*.py'],
    },
    include_package_data=True,

    # Python version requirement
    python_requires='>=3.8',

    # Dependencies
    install_requires=read_requirements(),

    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'pytest-xdist>=3.3.0',
            'pytest-mock>=3.11.1',
            'pytest-asyncio>=0.21.0',
            'black>=23.7.0',
            'flake8>=6.1.0',
            'mypy>=1.5.0',
            'ipython>=8.0.0',
            'jupyter>=1.0.0',
        ],
        'test': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'pytest-xdist>=3.3.0',
            'pytest-mock>=3.11.1',
        ],
        'docs': [
            'sphinx>=6.0.0',
            'sphinx-rtd-theme>=1.2.0',
            'sphinx-autodoc-typehints>=1.23.0',
        ],
        'gpu': [
            'torch>=2.0.0+cu118',  # CUDA version
            'bitsandbytes>=0.41.0',
        ],
    },

    # Project classification
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],

    # Keywords for search
    keywords=[
        'clinical documentation',
        'CDI',
        'RAG',
        'retrieval augmented generation',
        'medical AI',
        'healthcare NLP',
        'language models',
        'transformers',
    ],

    # Entry points for command-line scripts (optional)
    entry_points={
        'console_scripts': [
            # Add CLI commands here if needed
            # 'cdi-rag=cdi_rag.cli:main',
        ],
    },

    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/your-org/cdi-rag/issues',
        'Source': 'https://github.com/your-org/cdi-rag',
        'Documentation': 'https://github.com/your-org/cdi-rag/blob/main/README.md',
    },

    # License
    license='MIT',

    # Zip safe
    zip_safe=False,
)
