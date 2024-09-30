import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DOLPHINN",
    version="0.1.0",
    author="Rudy Alkarem",
    author_email="yuksel.alkarem@maine.edu",
    description="A predictive model for Wave Reconstruction Predictions and Digital Twin Applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Yuksel-Rudy/DOLPHINN",
    packages=['vmod'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        "numpy",
        "matplotlib",
        "pyyaml",
        "scipy", 
        "pandas",
        "scikit-learn",
        "tensorflow",
        "keras",
        "h5py",
        "pillow",
        "requests",
        "aiohttp",
        "sqlalchemy",
        "optuna",
        "tqdm",
    ],
)