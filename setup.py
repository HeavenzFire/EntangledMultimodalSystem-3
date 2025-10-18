from setuptools import setup, find_packages

setup(
    name="quantum_consciousness",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "qiskit>=0.34.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.0",
            "black>=21.7b0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pylint>=2.9.0",
        ],
    },
    python_requires=">=3.8",
    author="Quantum Consciousness Team",
    author_email="quantum@consciousness.ai",
    description="A revolutionary quantum computing framework with sacred geometry integration",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quantum-consciousness",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Quantum Computing",
    ],
) 