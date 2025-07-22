# How to contribute

It's great that you found your way to this place and want to contribute!
Any contribution, help, and feedback is highly appreciated!

The following sections give a guide for the different types of feedback/contributions.

## Feature requests and bug reports

Please open an [issue](https://github.com/cpossel/tvha/issues) in this repo.
Please try to provide as much information as needed to understand (and reproduce) the issue.
For bug reports a minimal example would be highly appreciated.

## Bug fixes/own contributions

This project uses ruff for code formatting and linting and pytest for testing.
Type annotations shall be used but they aren't enforced too strictly since mypy often reports false positives and sometimes feels more troublemore than helpful...

The settings for all these tools can be found in [pyproject.toml](pyproject.toml) and are used automatically if you use `nox` as described in below section [local testing](#local-testing).

### Local testing

Before pushing code to the git repository, it is helpful to run tests locally.
The same tests are run within the CI/CD pipeline in the git repository so failure/rejection of a push becomes far more unlikely if you run the tests locally beforewards.
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

Detailed information about the implemented tests can be found in the [noxfile](noxfile.py).

## Further questions

If you have further questions that do not fit into the scope of above, you can write an email to <clemens.possel@ict.fraunhofer.de>.
Be aware that this software comes with no warranty and all support is purely voluntarily.
