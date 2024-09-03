from setuptools import setup, find_packages

setup(
    name="branch-tensor",
    version="1.0",
    author="Antoine Debouchage",
    author_email="antoine311200@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)