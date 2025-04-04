from setuptools import setup, find_packages

setup(
    name="neuropilot",
    version="0.1.0",
    description="NeuroPilot: Lightweight multi-dataset DL/ML experiment manager",
    author="Austin Dibble",
    author_email="you@example.com",
    packages=find_packages(),  # Automatically finds 'autotrainer'
    install_requires=[
        "pyyaml>=5.3",
        "pandas>=1.1.0",
        "loguru>=0.6.0"
    ],
    python_requires=">=3.7",
    include_package_data=True,
)