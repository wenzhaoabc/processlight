from setuptools import find_packages, setup

setup(
    name="rltsc",
    version="0.1.0",
    author="Wen",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "accelerate>=1.10.1",
        "datasets>=4.3.0",
        "peft>=0.17.1",
        "transformers>=4.57.1",
        "trl>=0.24.0",
    ],
    zip_safe=False,
)
