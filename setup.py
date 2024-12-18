from setuptools import setup, find_packages

def read_requirements(filename, category=None):
    with open(filename) as f:
        requirements = []
        current_category = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                current_category = line[2:].strip().lower()  # Convert to lowercase for consistency
                continue
            if category and current_category != category.lower():
                continue
            if ' #' in line:
                line = line.split(' #')[0].strip()
            requirements.append(line)
        return requirements

# Get all requirements without filtering by category
def get_all_requirements(filename):
    return [req for req in read_requirements(filename) if req]

setup(
    name="ai_scientist",
    packages=find_packages(),
    install_requires=get_all_requirements("requirements.txt"),  # Install all requirements by default
    python_requires=">=3.8,<3.12",
)
