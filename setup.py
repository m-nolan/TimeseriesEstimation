import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_decription = fh.read()

install_requires = [
    'torch',
    'h5py',
    'pyyaml',
]

setuptools.setup(
    name="timeseries-prediction",
    version="0.0.1",
    author="Michael Nolan",
    author_email="manolan@uw.edu",
    description="Timeseries data reconstruction and prediction models in pytorch.",
    long_decription=long_decription,
    long_decription_content_type="test/markdown",
    url="https://github.com/m-nolan/timeseries_prediction",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=install_requires,
)