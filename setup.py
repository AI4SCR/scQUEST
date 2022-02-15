"""Install package."""
import io
import re

from pkg_resources import parse_requirements
from setuptools import find_packages, setup


def read_version(filepath: str) -> str:
    """Read the __version__ variable from the file.

    Args:
        filepath: probably the path to the root __init__.py

    Returns:
        the version
    """
    match = re.search(
        r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        io.open(filepath, encoding="utf_8_sig").read(),
    )
    if match is None:
        raise SystemExit("Version number not found.")
    return match.group(1)


# ease installation during development
vcs = re.compile(r"(git|svn|hg|bzr)\+")
try:
    with open("requirements.txt") as fp:
        VCS_REQUIREMENTS = [
            str(requirement)
            for requirement in parse_requirements(fp)
            if vcs.search(str(requirement))
        ]
except FileNotFoundError:
    # requires verbose flags to show
    print("requirements.txt not found.")
    VCS_REQUIREMENTS = []

# TODO: Update these values according to the name of the module.
setup(
    name="starProtocols",
    version=read_version("starProtocols/__init__.py"),  # single place for version
    description="Installable starProtocols package. Modify as needed.",
    long_description=open("README.md").read(),
    url="https://github.ibm.com/art-zurich/starProtocols-python-package",
    author="Adriano Martinelli",
    author_email="art@zurich.ibm.com",
    # the following exclusion is to prevent shipping of tests.
    # if you do include them, add pytest to the required packages.
    packages=find_packages(".", exclude=["*tests*"]),
    package_data={"starProtocols": ["py.typed"]},
    entry_points="""
        [console_scripts]
        salutation=starProtocols.complex_module.core:formal_introduction
    """,
    scripts=["bin/brief_salutation", "bin/a_shell_script"],
    extras_require={
        "vcs": VCS_REQUIREMENTS,
        "test": ["pytest", "pytest-cov"],
        "dev": [
            # tests
            "pytest",
            "pytest-cov",
            # checks
            "black==21.5b0",
            "flake8",
            "mypy",
            # docs
            "sphinx",
            "sphinx-autodoc-typehints",
            "better-apidoc",
            "six",
            "sphinx_rtd_theme",
            "myst-parser",
        ],
    },
    install_requires=[
        # versions should be very loose here, just exclude unsuitable versions
        # because your dependencies also have dependencies and so on ...
        # being too strict here will make dependency resolution harder
        "click",
        'pandas',
        'torch',
        'scikit-learn',
        'numpy >= 1.2',
        'pytorch-lightning',
        'anndata',
        'matplotlib',
        'seaborn',
        'torchmetrics',
        'pytorch_lightning',
    ],
)
