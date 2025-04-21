from setuptools import setup, find_packages

setup(
    name="entangled-multimodal-system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "flask>=2.0.0",
        "requests>=2.26.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-mock>=3.10.0",
        "pytest-asyncio>=0.21.0",
        "pytest-timeout>=2.1.0",
        "pytest-xdist>=3.3.0",
        "pytest-html>=3.2.0",
        "pytest-metadata>=3.0.0",
        "coverage>=7.0.0",
        "pytest-benchmark>=3.4.1",
        "pytest-randomly>=3.12.0",
        "python-dotenv>=0.19.0",
        "pyyaml>=5.4.0",
        "loguru>=0.5.3",
        "matplotlib>=3.4.0",
        "boto3>=1.26.0",
        "qiskit>=0.39.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0"
    ],
    extras_require={
        "tensorflow": ["tensorflow>=2.12.0"],
        "dev": [
            "black>=21.5b2",
            "isort>=5.9.0",
            "flake8>=3.9.0",
            "mypy>=0.910"
        ]
    },
    python_requires=">=3.8",
) 