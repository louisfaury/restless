from setuptools import setup, find_packages

try:
    # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:
    # for pip <= 9.0.3
    from pip.req import parse_requirements

USER_REQUIREMENTS = 'requirements.txt'
install_reqs = [r.requirement for r in parse_requirements(USER_REQUIREMENTS, session="setup")]

setup(
    name="restless",
    version="0.0",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.6",
    license="Apache License 2.0",
    author="Louis Faury",
    author_email="l.faury@criteo.com",
    description="Restless Bandit Simulator",
)
