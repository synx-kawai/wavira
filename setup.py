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
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "wavira-train=scripts.train:main",
            "wavira-eval=scripts.evaluate:main",
        ],
    },
)
