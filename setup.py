"""Package installation setup."""

from setuptools import setup, find_packages

# Read long description from README
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "UFM: A Simple Path towards Unified Dense Correspondence with Flow"

# Core dependencies
install_requires = [
    "torch",
    "torchvision",
    "torchaudio",
    "numpy",
    "matplotlib",
    "opencv-python",
    "flow_vis",
    "huggingface_hub",
    "einops",
    "gradio",
    # UniCeption dependencies
    "timm",
    "jaxtyping",
    "Pillow",
    "scikit-learn",
]

# Optional dependencies
extras_require = {
    "dev": [
        "black",
        "isort",
        "pre-commit",
        "pytest",
    ],
    "demo": [
        "gradio",
    ],
    "all": [
        "black",
        "isort",
        "pre-commit",
        "pytest",
        "gradio",
    ],
}

setup(
    name="uniflowmatch",
    version="0.1.0",
    description="UFM: A Simple Path towards Unified Dense Correspondence with Flow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Yuchen Zhang",
    author_email="yuchenz7@andrew.cmu.edu",
    url="https://uniflowmatch.github.io/",
    license="BSD Clause-3",
    packages=find_packages(include=["uniception", "uniflowmatch*"]),
    package_dir={
        "uniception": "UniCeption/uniception",  # Map uniception package
        "uniflowmatch": "uniflowmatch",  # Map uniflowmatch package
    },
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "ufm=uniflowmatch.cli:main",
            "ufm-demo=gradio_demo:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="computer-vision, optical-flow, correspondence, deep-learning, pytorch, transformer",
)
