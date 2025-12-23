from setuptools import setup, find_packages

setup(
    name="wavira",
    version="0.1.0",
    description="Wi-Fi-based Person Re-Identification using Deep Learning",
    author="Wavira Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "h5py>=3.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.23.0",
            "httpx>=0.26.0",
            "fastapi>=0.109.0",
            "uvicorn>=0.27.0",
            "pydantic>=2.5.0",
            "pyyaml>=6.0.0",
            "pyserial>=3.5",
            "websockets>=12.0",
        ],
        "logging": [
            "tensorboard>=2.14.0",
        ],
        "config": [
            "pyyaml>=6.0",
        ],
        "full": [
            "tensorboard>=2.14.0",
            "pyyaml>=6.0",
            "pyserial>=3.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "wavira-train=scripts.train:main",
            "wavira-eval=scripts.evaluate:main",
        ],
    },
)
