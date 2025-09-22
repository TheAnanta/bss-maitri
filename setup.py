from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bss-maitri",
    version="1.0.0",
    author="BSS Maitri Team",
    description="Multimodal AI Assistant for Crew Emotional and Physical Well-being",
    long_description=long_description,
    long_description_content_type="text/markdown",
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
    ],
    python_requires=">=3.8",
    install_requires=[
        "ollama>=0.1.8",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "librosa>=0.10.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pillow>=10.0.0",
        "transformers>=4.30.0",
        "gradio>=3.40.0",
        "pyaudio>=0.2.11",
        "soundfile>=0.12.0",
        "webrtcvad>=2.0.10",
        "mediapipe>=0.10.0",
        "face-recognition>=1.3.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "click>=8.1.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "bss-maitri=bss_maitri.main:main",
        ],
    },
)