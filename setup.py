from setuptools import setup, find_packages

def read_requirements(filename):
    with open(filename) as f:
        requirements = []
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ' #' in line:
                line = line.split(' #')[0].strip()
            requirements.append(line)
        return requirements

setup(
    name="ai_scientist",
    packages=find_packages(),
    install_requires=read_requirements("requirements.txt"),
    python_requires=">=3.8,<3.12",
)
