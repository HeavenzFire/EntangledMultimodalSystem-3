from setuptools import setup, find_packages

setup(
    name="quantum_consciousness",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pennylane>=0.22.0",
        "scipy>=1.7.0",
        "networkx>=2.6.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.0",
        "pytest>=6.2.5",
        "pytest-cov>=2.12.0",
        "codecov>=2.1.11",
        "matplotlib>=3.4.0",
        "tensorflow>=2.6.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "torch>=1.9.0",
        "transformers>=4.11.0"
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "codecov",
            "black",
            "isort",
            "flake8"
        ]
    },
    python_requires=">=3.9",
    author="Quantum Consciousness Team",
    author_email="quantum@consciousness.ai",
    description="A framework for quantum consciousness and sacred geometry",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quantum-consciousness",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics"
    ]
) 