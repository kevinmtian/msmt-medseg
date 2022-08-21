from setuptools import setup, find_packages

exec(open('pytorch3dunet/__version__.py').read())
setup(
    name="pytorch3dunet",
    packages=find_packages(exclude=["tests"]),
    version=__version__,
    author="",
    url="",
    license="MIT",
    python_requires='>=3.7'
)
