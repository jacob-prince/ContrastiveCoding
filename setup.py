from setuptools import setup, find_packages
from git import Repo

repo_name = Repo('.').remotes.origin.url.split('.git')[0].split('/')[-1]

setup(
    name = repo_name,
    packages = find_packages(),
)