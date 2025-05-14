# truncated Variational Hamiltonian Ansatz

This repository contains the source code for the Variational Hamiltonian Ansatz VHA and its variant truncated Variational Hamiltonian Ansatz tVHA.
It is based on the code of Qiskit and Qiskit Nature.

- [truncated Variational Hamiltonian Ansatz](#truncated-variational-hamiltonian-ansatz)
  - [Installation](#installation)
  - [First steps](#first-steps)
  - [Troubleshooting](#troubleshooting)
  - [Contribute your own code](#contribute-your-own-code)
    - [Local testing](#local-testing)
  - [Chemistry Drivers for Windows](#chemistry-drivers-for-windows)
    - [PSI4 (Windows)](#psi4-windows)
    - [PyQuante (Windows)](#pyquante-windows)
    - [Gaussian (Windows)](#gaussian-windows)

## Installation

Download this package from [tVHA homepage](https://gitlab.cc-asp.fraunhofer.de/qc/tvha/) or clone it via git `git clone git@gitlab.cc-asp.fraunhofer.de:qc/tvha.git`.

Go into the main folder of the downloaded folder and run

```bash
pip install .
```

For developers it is recommended to run

```bash
pip install -e .[develop]
```

To check whether the installation was successful, one can run `pip show tvha`.

If you are on a Linux system or use [WSL](https://learn.microsoft.com/windows/wsl/install), installation is complete after above step.
On Windows systems, please check [below section](#chemistry-drivers-for-windows).

## First steps

The best starting point is to go to the [examples](./examples/) folder and check out the files there.

## Troubleshooting

If reading the documentation didn't help, feel free to check out our [GitLab page](https://gitlab.cc-asp.fraunhofer.de/qc/tvha/-/issues).
Here you can also submit bug reports and feature requests.

If you fixed some bug in the code on your own, please don't forget to [contribute](https://gitlab.cc-asp.fraunhofer.de/qc/tvha/-/merge_requests) your bug fix ;)

## Contribute your own code

Feel free to contribute to this project at [GitLab](https://gitlab.cc-asp.fraunhofer.de/qc/tvha)!

Any contribution and help is highly appreciated!

This project uses ruff (and mypy) for code formatting and linting and pytest for testing.
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

## Chemistry Drivers for Windows

A computational chemistry program/library, called "driver", is needed in addition to the Python packages listed in [pyproject.toml](pyproject.toml).

By default, [PySCF](https://pyscf.org/) is already installed on Linux when you follow above instructions.

Since the recommended driver [PySCF](https://pyscf.org/) does not support Windows, the recommended way is to install [WSL](https://learn.microsoft.com/windows/wsl/install).
If you still want to stick to Windows, you can find the installation instructions below.

For Windows, these drivers are available:

- [PSI4](https://psicode.org) (recommended)
- [PyQuante](https://github.com/rpmuller/pyquante2) (reduced functionality)
- [Gaussian](https://gaussian.com/gaussian16/) (commercial)

Please be aware that the code must be adjusted in some places to use these drivers instead of PySCF.

### PSI4 (Windows)

```powershell
cd path/to/this/repo
conda create -n my_qiskit_env
conda activate my_qiskit_env
conda install psi4 python=3.12 -c psi4 -c conda-forge
pip install .
```

Above commands create a new `conda` environment called `my_qiskit_psi4_env` and install all required packages into it.

Since installing [PSI4](https://psicode.org) is sometimes troublesome, please follow above commands closely.
It is important to create a new empty environment from the command line without any python version or package specified before installing [PSI4](https://psicode.org) as done with above commands.
Using the `Anaconda` GUI to create an empty environment didn't work during testing.

Test in a python shell if the installation of [PSI4](https://psicode.org) succeeded:

```python
import psi4
```

### PyQuante (Windows)

```powershell
conda install -c rpmuller pyquante2_pure
```

PyQuante is not actively maintained and contains only very basic functionality.
So, use it only if you have no other choice.

### Gaussian (Windows)

See [Qiskit/Gaussian](https://qiskit.org/documentation/apidoc/qiskit.chemistry.drivers.gaussiand.html) and [Gaussian](http://gaussian.com/gaussian16/).

Below guide assumes Gaussian is installed in `C:\G16W\`. You should adjust the path accordingly.

1. Add `g16.exe` to `PATH`

    In conda environment (easiest permanent solution; see also <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#setting-environment-variables>):

    ```powershell
    conda env config vars set PATH="$env:PATH;C:\G16W"
    ```

    Check if it succeeded (might need `conda activate <my_env>` for changes to take place):

    ```powershell
    $env:PATH
    ```

    or

    ```powershell
    conda env config vars list
    ```

    If successful, a warning about overwriting the `PATH` environment variable arises everytime the conda environment is launched. During 'overwriting' only a path is appended and nothing deleted, so this shouldn't (hopefully) pose any problem.

    Alternatively one of the following approaches can be used (non-permanent):

    - In python (before all other imports):

        ```python
        import os
        os.environ["PATH"] += ";C:\G16W"
        ```

    - Or in powershell (before starting the python script):

        ```powershell
        $env:Path += ";C:\G16W"
        ```

2. Set GAUSS_EXEDIR environment variable

    Add `GAUSS_EXEDIR=C:\G16W` permanently to the currently active conda environment:

    ```powershell
    conda env config vars set GAUSS_EXEDIR=C:\G16W
    ```

    Check if it succeeded (might need `conda activate <my_env>` for changes to take place):

    ```powershell
    $env:GAUSS_EXEDIR
    ```

    or

    ```powershell
    conda env config vars list
    ```

    Now everything should be set up correctly.

    You might get one of the following errors:
    - `ImportError: DLL load failed while importing qcmatrixio: Das angegebene Modul wurde nicht gefunden.` (The specified module was not found.)
    - `qiskit_nature.exceptions.QiskitNatureError: 'qcmatrixio extension not found. See Gaussian driver readme to build qcmatrixio.F using f2py'`

    The missing module can be installed via

    ```powershell
    conda install -c anaconda intel-fortran-rt
    ```

    Alternatively, one can try `conda install icc_rt`.

