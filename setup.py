from setuptools import setup, find_packages

setup(
    name="bss-maitri",
    version="0.1.0",
    description="Bharatiya Space Station Maitri - Multimodal AI Assistant for Crew Well-being",
    author="BSS Maitri Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "librosa>=0.10.0",
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "loguru>=0.7.0",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "bss-maitri=bss_maitri.main:main",
        ],
    },
)