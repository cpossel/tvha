"""FermionicOp from qiskit monkey patched with some useful functions for Hamiltonians."""

import logging
from collections.abc import Callable

import numpy as np
from qiskit.exceptions import QiskitError
from qiskit_nature.second_q.operators import FermionicOp

logger = logging.getLogger(__name__)


# Define some useful methods for fermionic operators.
# Since they are added in a monkey patch fashion, print a warning.
logger.warning(
    "Monkey patching some methods to class '%s.FermionicOp' (see file '%s'). "
    "In long term these methods should probably be added to the official Qiskit code...",
    FermionicOp.__module__,
    __file__,
)


def _add_function_to_class(function: Callable, cls: type, is_static: bool = False) -> None:
    """Adds function as method to class protecting from overwriting existing methods/attributes."""
    if hasattr(cls, function.__name__):
        raise QiskitError(
            "Cannot overwrite already defined class method "
            "'{method.__name__}' from '{cls.__module__}.{cls.__name__}'."
        )
    setattr(cls, function.__name__, staticmethod(function) if is_static else function)


def get_one_body_hamiltonian(self: FermionicOp) -> FermionicOp:
    """Gets the one-body terms of the given Hamiltonian."""
    return FermionicOp(
        data={
            label: coeff.real if isinstance(coeff, complex) else coeff
            for label, coeff in self.items()
            if len([lbl.split("_") for lbl in label.split()]) == 2
        },
        num_spin_orbitals=self.num_spin_orbitals,
    )


_add_function_to_class(get_one_body_hamiltonian, FermionicOp)


def get_two_body_hamiltonian(self: FermionicOp) -> FermionicOp:
    """Gets the two-body terms of the given Hamiltonian.

    All two-body terms (i.e. sum of get_two_body_hamiltonian_coulomb_terms and
    get_two_body_hamiltonian_complex_terms)."""
    return FermionicOp(
        data={
            label: coeff.real if isinstance(coeff, complex) else coeff
            for label, coeff in self.items()
            if len([lbl.split("_") for lbl in label.split()]) == 4
        },
        num_spin_orbitals=self.num_spin_orbitals,
    )


_add_function_to_class(get_two_body_hamiltonian, FermionicOp)


def get_two_body_hamiltonian_coulomb_terms(self: FermionicOp) -> FermionicOp:
    r"""Gets the two-body terms of the given Hamiltonian (only Coulomb terms).

    Only terms of shape a_i^\dagger a_j^\dagger a_i a_j."""
    return FermionicOp(
        data={
            label: coeff.real if isinstance(coeff, complex) else coeff
            for label, coeff in self.items()
            if len([lbl.split("_") for lbl in label.split()]) == 4
            and label.split()[0].split("_")[-1] == label.split()[2].split("_")[-1]
            and label.split()[1].split("_")[-1] == label.split()[3].split("_")[-1]
        },
        num_spin_orbitals=self.num_spin_orbitals,
    )


_add_function_to_class(get_two_body_hamiltonian_coulomb_terms, FermionicOp)


def get_two_body_hamiltonian_complex_terms(self: FermionicOp) -> FermionicOp:
    r"""Gets the two-body terms of the given Hamiltonian.

    Only terms of shape a_i^\dagger a_j^\dagger a_k a_l with i != k and j != l."""
    return FermionicOp(
        data={
            label: coeff.real if isinstance(coeff, complex) else coeff
            for label, coeff in self.items()
            if len([lbl.split("_") for lbl in label.split()]) == 4
            and (
                label.split()[0].split("_")[-1] != label.split()[2].split("_")[-1]
                or label.split()[1].split("_")[-1] != label.split()[3].split("_")[-1]
            )
        },
        num_spin_orbitals=self.num_spin_orbitals,
    )


_add_function_to_class(get_two_body_hamiltonian_complex_terms, FermionicOp)


def is_diagonal(self: FermionicOp) -> bool:
    """Returns true if the given Hamiltonian is diagonal."""
    for label in self:
        op = [lbl.split("_") for lbl in label.split()]
        if len(op) != 2:
            logger.error("Method is_diagonal can only be used for one-body Hamiltonians.")
            return False
        if op[0][1] != op[1][1]:
            return False
    return True


_add_function_to_class(is_diagonal, FermionicOp)


def is_antisymmetric(self: FermionicOp) -> bool:
    """Check if a given Hamiltonian is antisymmetric (only two-body terms).

    This implementation iterates inefficiently over all operators
    though it would be sufficient to iterate only over those with i<j and k<l;
    if it turns out that this method is very slow, adjusting this might yield a factor 4 speedup.
    """
    for label, coeff in self.items():
        op = [lbl.split("_") for lbl in label.split()]
        if len(op) == 4:
            if op[0][1] == op[1][1] or op[2][1] == op[3][1]:
                # terms that should be filtered out with FermionicOp.simplify() but obviously aren't
                return False
            labels_antisymmetric = FermionicOp.get_label_antisymmetric_counterparts(op)
            if not np.isclose(
                [
                    -self[labels_antisymmetric[0]],
                    -self[labels_antisymmetric[1]],
                    self[labels_antisymmetric[2]],
                ],
                coeff,
            ).all():
                return False
    return True


_add_function_to_class(is_antisymmetric, FermionicOp)


def get_compressed_hamiltonian(self: FermionicOp) -> FermionicOp:
    r"""Compress a given Hamiltonian only keeping non-redundant terms.

    The Hamiltonian's information content is equivalent to the one from
    FermionicOp.get_antisymmetrized_hamiltonian().
    The comporessed Hamiltonian representation is not antisymmetric but yields the same
    Pauli strings/qubit operator after mapping as the antisymmetrized Hamiltonian.

    Two-body operators a†_i a†_j a_k a_l are antisymmetrized then redundant ones merged;
    one-body operators are kept unchanged.
    Non-physical terms in the Hamiltonian (e.g. a†_i a†_i a_k a_l or a†_i a†_j a_k a_k)
    are filtered out with method FermionicOp.simplify.

    Only non-redundant terms are kept, i.e. summation is done only of a forth of the terms
    compared to those returned from FermionicOp.get_antisymmetrized_hamiltonian().
    \tilde h_ijkl = (h_ijkl-h_jikl-h_ijlk+h_jilk)
    H_{\rm int} = \sum_{ijkl} h_{ijkl} a_i^\dagger a_j^\dagger a_k a_l
    = \sum_{i<j, k<l} \tilde h_{ijkl} a_i^\dagger a_j^\dagger a_k a_l.
    Due to usage of FermionicOp.get_sorted_fermionic_operators, it is guaranteed that only
    two-body terms with i<=j and k<=l will appear in the compressed Hamiltonian.
    """
    fermionic_op = self.simplify()
    compressed_terms = {}  # final data of the compressed Hamiltonian
    discarded_labels: list[str] = []  # redundant terms which will be discarded

    for label, coeff in fermionic_op.get_sorted_fermionic_operators().items():
        if label in discarded_labels:
            continue

        op = [lbl.split("_") for lbl in label.split()]
        if len(op) == 2:
            compressed_terms[label] = coeff.real
        elif len(op) == 4:
            labels = self.get_label_antisymmetric_counterparts(op)
            coeff_antisymmetrized = (
                coeff
                - fermionic_op.get(labels[0], 0)
                - fermionic_op.get(labels[1], 0)
                + fermionic_op.get(labels[2], 0)
            )
            coeff_antisymmetrized = coeff_antisymmetrized.real

            label_permuted, switch_sign = self.get_permuted_label(label)
            if switch_sign:
                coeff_antisymmetrized = -coeff_antisymmetrized

            if not np.isclose(coeff_antisymmetrized, 0.0):  # only add non-zero terms
                compressed_terms[label_permuted] = coeff_antisymmetrized
            discarded_labels.extend((label, labels[0], labels[1], labels[2]))
        else:
            raise ValueError(
                "Encountered unsupported term with wrong number of creation/annihilation operators"
            )

    return FermionicOp(data=compressed_terms, num_spin_orbitals=fermionic_op.num_spin_orbitals)


_add_function_to_class(get_compressed_hamiltonian, FermionicOp)


def get_antisymmetrized_hamiltonian(self: FermionicOp) -> FermionicOp:
    """Antisymmetrize a given Hamiltonian.

    Two-body operators a†_i a†_j a_k a_l are antisymmetrized;
    one-body operators are kept unchanged.
    Non-physical terms in the Hamiltonian (e.g. a†_i a†_i a_k a_l or a†_i a†_j a_k a_k)
    are filtered out with method FermionicOp.simplify.

    The FermionicOps returned by get_compressed_hamiltonian and get_antisymmetrized_hamiltonian
    are equivalent with regard to mapping to Pauli words/qubit operators;
    the one returned by get_compressed_hamiltonian should be preferred
    since it more memory efficient by a factor of 4.
    """
    fermionic_op = self.simplify()
    antisymmetric_terms = {}

    for label, coeff in fermionic_op.items():
        if label in antisymmetric_terms:
            continue

        op = [lbl.split("_") for lbl in label.split()]
        if len(op) == 2:
            antisymmetric_terms[label] = coeff.real
        elif len(op) == 4:
            # Fully antisymmetrized Hamiltonian;
            # results in a longer Hamiltonian than the pure .simplify() Hamiltonian
            # though both translate to the same pauli operator (with same coefficients)
            labels = FermionicOp.get_label_antisymmetric_counterparts(op)
            coeff_antisymmetrized = (
                coeff
                - fermionic_op.get(labels[0], 0)
                - fermionic_op.get(labels[1], 0)
                + fermionic_op.get(labels[2], 0)
            ) / 4
            coeff_antisymmetrized = coeff_antisymmetrized.real
            if not np.isclose(coeff_antisymmetrized, 0.0):  # only add non-zero terms
                antisymmetric_terms[label] = coeff_antisymmetrized
                antisymmetric_terms[labels[0]] = -coeff_antisymmetrized
                antisymmetric_terms[labels[1]] = -coeff_antisymmetrized
                antisymmetric_terms[labels[2]] = coeff_antisymmetrized

        else:
            raise ValueError(
                "Encountered unsupported term with wrong number of creation/annihilation operators"
            )

    return FermionicOp(data=antisymmetric_terms, num_spin_orbitals=fermionic_op.num_spin_orbitals)


_add_function_to_class(get_antisymmetrized_hamiltonian, FermionicOp)


def get_label_antisymmetric_counterparts(op: list[list[str]]) -> list[str]:
    """Switches the indices of the antisymmetric counterparts of Hamiltonian terms.

    a†_i a†_j a_k a_l -> [a†_j a†_i a_k a_l, a†_i a†_j a_l a_k, a†_j a†_i a_l a_k]

    Returns: list of 3 counterparts of which the first 2 contain only a single index switch
        thus needing a negative sign,
        and the 3rd one contains 2 index switches thug needing no sign change.
        The signs need to be accounted for manually (not included in this function)."""
    if len(op) == 4:
        label_counterpart = []
        label_counterpart.append(
            f"{op[1][0]}_{op[1][1]} {op[0][0]}_{op[0][1]} "
            f"{op[2][0]}_{op[2][1]} {op[3][0]}_{op[3][1]}"
        )  # switched i <-> j
        label_counterpart.append(
            f"{op[0][0]}_{op[0][1]} {op[1][0]}_{op[1][1]} "
            f"{op[3][0]}_{op[3][1]} {op[2][0]}_{op[2][1]}"
        )  # switched k <-> l
        label_counterpart.append(
            f"{op[1][0]}_{op[1][1]} {op[0][0]}_{op[0][1]} "
            f"{op[3][0]}_{op[3][1]} {op[2][0]}_{op[2][1]}"
        )  # switched i <-> j and k <-> l
        return label_counterpart
    raise ValueError(
        "Encountered unsupported term with wrong number of creation/annihilation operators. "
    )


_add_function_to_class(get_label_antisymmetric_counterparts, FermionicOp, is_static=True)


def get_label_of_hermitian_counterpart(label: str) -> str:
    """Returns hermitian counterpart of a fermionic operator.

    Gets a†_l a†_k a_j a_i from a†_i a†_j a_k a_l.
    In qiskit language: "+_l +_k -_j -_i" from input "+_i +_j -_k -_l"
    """
    op = [lbl.split("_") for lbl in label.split()]
    if len(op) != 4:
        raise ValueError("'get_label_of_hermitian_counterpart' only supports for two-body terms.")
    return (
        f"{op[0][0]}_{op[3][1]} {op[1][0]}_{op[2][1]} {op[2][0]}_{op[1][1]} {op[3][0]}_{op[0][1]}"
    )


_add_function_to_class(get_label_of_hermitian_counterpart, FermionicOp, is_static=True)


def get_sorted_fermionic_operators(self: FermionicOp) -> dict[str, complex | float]:
    """Returns the data dict of FermionicOp sorted numerically by label.

    The leftmost term takes precedence during sorting
    (i.e. i over j in a†_i a_j and i over j over k over l in a†_i a†_j a_k a_l).
    It is taken care that the sorting is done numerically (1,2,...9,10,11,..)
    and not lexicographically (1,10,11,..,2,20,21,...)."""
    one_body = self.get_one_body_hamiltonian()
    one_body_terms_sorted = dict(
        sorted(one_body.items(), key=lambda x: [int(lbl.split("_")[1]) for lbl in x[0].split()])
    )
    two_body = self.get_two_body_hamiltonian()
    two_body_terms_sorted = dict(
        sorted(two_body.items(), key=lambda x: [int(lbl.split("_")[1]) for lbl in x[0].split()])
    )
    return one_body_terms_sorted | two_body_terms_sorted


_add_function_to_class(get_sorted_fermionic_operators, FermionicOp)


def get_permuted_label(label: str) -> tuple[str, bool]:
    """Gets the permuted label so that i<j and k<l.

    The bool indicates whether to add a minus sign.
    """
    switch_sign = False
    op = [lbl.split("_") for lbl in label.split()]
    # If a label arises with i>j and/or k>l switch them and
    # add a minus sign for each permutation
    if op[0][1] > op[1][1]:
        _idx = op[1][1]
        op[1][1] = op[0][1]
        op[0][1] = _idx
        switch_sign = not switch_sign
    if op[2][1] > op[3][1]:
        _idx = op[3][1]
        op[3][1] = op[2][1]
        op[2][1] = _idx
        switch_sign = not switch_sign
    # Rebuild the label
    label_permuted = (
        f"{op[0][0]}_{op[0][1]} {op[1][0]}_{op[1][1]} "
        f"{op[2][0]}_{op[2][1]} {op[3][0]}_{op[3][1]}"
    )
    return label_permuted, switch_sign


_add_function_to_class(get_permuted_label, FermionicOp, is_static=True)
