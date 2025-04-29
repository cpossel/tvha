"""Collection of tests for code quality.

The tests can be run via the following command:

```bash
pip install nox
nox
```

If you want to run only a subset of tests, they can be specified via
```nox -s <session_name>```
where <session_name> is the name of the respective function in this script.
Or via
```nox -t <tag_name>```
where <tag_name> is the name of the tag
(e.g. "style" for code style related testing or "tests" for testing of the code functionality).
"""

import nox

nox.needs_version = ">=2024.04"

# Specify default tags
# nox.options.tags = ["style", "tests"]


@nox.session(reuse_venv=True, tags=["style"])
def black(session: nox.Session) -> None:
    """Code format check using black."""
    session.install("black")
    session.run("black", ".", "--diff", "--check")


@nox.session(reuse_venv=True, tags=["style"])
def ruff(session: nox.Session) -> None:
    """Code check using ruff."""
    session.install("ruff")
    session.run("ruff", "check", "vha/")


@nox.session(reuse_venv=True, tags=["style", "tests"])
def mypy(session: nox.Session) -> None:
    """Code check using mypy."""
    session.install(".", "mypy")
    session.run("mypy", ".")


# @nox.session(python=["3.11", "3.12"], reuse_venv=True, tags=["tests"])
# def pytest(session: nox.Session) -> None:
#     """Test suite using pytest."""
#     session.install(".[plot]", "pytest")
#     session.run("pytest", "-v")
