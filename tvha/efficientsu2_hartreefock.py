# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# Modification note:
# This code is written by Clemens Possel, taking the qiskit code as basis.

"""Implementation of "warm start" of EfficientSU2 with HartreeFock initial state."""

from collections.abc import Callable

import numpy as np
from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.circuit.library.standard_gates import CXGate, RYGate, RZGate
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import QubitMapper


class EfficientSU2_HartreeFock(EfficientSU2):  # noqa: N801
    r"""The hardware efficient SU(2) 2-local circuit with HartreeFock initial state.

    The ``EfficientSU2`` circuit consists of layers of single qubit operations spanned by SU(2)
    and :math:`CX` entanglements. This is a heuristic pattern that can be used to prepare trial wave
    functions for variational quantum algorithms or classification circuit for machine learning.

    SU(2) stands for special unitary group of degree 2, its elements are :math:`2 \times 2`
    unitary matrices with determinant 1, such as the Pauli rotation gates.

    On 3 qubits and using the Pauli :math:`Y` and :math:`Z` su2_gates as single qubit gates, the
    hardware efficient SU(2) circuit is represented by:

    .. parsed-literal::

    ┌───┐ ░ ┌──────────┐┌──────────┐ ░                 ░       ░ ┌───────────┐┌───────────┐
    ┤ X ├─░─┤ Ry(θ[0]) ├┤ Rz(θ[4]) ├─░─────────────■───░─ ... ─░─┤ Ry(θ[24]) ├┤ Rz(θ[28]) ├
    ├───┤ ░ ├──────────┤├──────────┤ ░           ┌─┴─┐ ░       ░ ├───────────┤├───────────┤
    ┤ X ├─░─┤ Ry(θ[1]) ├┤ Rz(θ[5]) ├─░────────■──┤ X ├─░─ ... ─░─┤ Ry(θ[25]) ├┤ Rz(θ[29]) ├
    ├───┤ ░ ├──────────┤├──────────┤ ░      ┌─┴─┐└───┘ ░       ░ ├───────────┤├───────────┤
    ┤ X ├─░─┤ Ry(θ[2]) ├┤ Rz(θ[6]) ├─░───■──┤ X ├──────░─ ... ─░─┤ Ry(θ[26]) ├┤ Rz(θ[30]) ├
    ├───┤ ░ ├──────────┤├──────────┤ ░ ┌─┴─┐└───┘      ░       ░ ├───────────┤├───────────┤
    ┤ X ├─░─┤ Ry(θ[3]) ├┤ Rz(θ[7]) ├─░─┤ X ├───────────░─ ... ─░─┤ Ry(θ[27]) ├┤ Rz(θ[31]) ├
    └───┘ ░ └──────────┘└──────────┘ ░ └───┘           ░       ░ └───────────┘└───────────┘

    See :class:`~qiskit.circuit.library.RealAmplitudes` for more detail on the possible arguments
    and options such as skipping unentanglement qubits, which apply here too.

    In contrast to its parent ansatz EfficientSU2, this class implements
    the hardware efficient SU(2) circuit with a HartreeFock_EfficientSU2 initial state.
    The initial state is created via prepending a HartreeFock initial state and
    the inverse of the EfficientSU2 circuit to the parameterized EfficientSU2 ansatz circuit.
    Thereby it is required that the initial_point for EfficientSU2 is the all-zero point
    (i.e. binding all parameters to the value zero for the first iteration).
    This allows to reduce the inverse EfficientSU2 circuit to solely the entangling layer
    (rotation layer turns out to be identity when binding all-zero parameters to it).
    Since the naive approach would yield an overhead of factor 2
    (the same amount of 'cx' gates as for EfficientSU2 prepended for the initial state),
    the 'cx' gates are evaluated using the knowledge that 'cx' on pure 0 or 1 states
    (i.e. those prepared by the HartreeFock initial state)
    can be directly evaluated classically without loosing any entanglement information.
    This procedure finally yields a HartreeFock_EfficientSU2 initial state tailored
    especially for the EfficientSU2 ansatz circuit.

    The HartreeFock_EfficientSU2 initial state is (in contrast to HartreeFock initial state)
    highly dependent on the ansatz;
    changing the number of repetitions `reps` or the `entanglement` scheme in the ansatz
    might yield a completely different HartreeFock_EfficientSU2 initial state.

    Since the naming convention can be confusing:
    HartreeFock initial state is the initial state as prepared by
    qiskit_nature.second_q.circuit.library.HartreeFock which is suitable e.g. for UCC ansatz.
    In contrast, the HartreeFock state used here is named HartreeFock_EfficientSU2 to show
    that it is tailored especially for the EfficientSU2 ansatz.

    Example:
    >>> H_2 = MoleculeInfo(
    >>>     symbols=["H", "H"],
    >>>     coords=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.74279)],
    >>>     multiplicity=1,
    >>>     charge=0,
    >>> )
    >>> driver = PySCFDriver.from_molecule(H_2)
    >>> problem = driver.run()
    >>> mapper = JordanWignerMapper()
    >>> circuit = EfficientSU2(problem=problem, mapper=mapper, reps=1)
    >>> ansatz.decompose().draw()
            ┌───┐    ┌──────────┐┌──────────┐                           ┌──────────┐┌───────────┐
    q_0: ───┤ X ├────┤ Ry(θ[0]) ├┤ Rz(θ[4]) ├───────────────────■───────┤ Ry(θ[8]) ├┤ Rz(θ[12]) ├
            ├───┤    ├──────────┤├──────────┤                 ┌─┴─┐     ├──────────┤├───────────┤
    q_1: ───┤ X ├────┤ Ry(θ[1]) ├┤ Rz(θ[5]) ├──────■──────────┤ X ├─────┤ Ry(θ[9]) ├┤ Rz(θ[13]) ├
         ┌──┴───┴───┐├──────────┤└──────────┘    ┌─┴─┐    ┌───┴───┴───┐┌┴──────────┤└───────────┘
    q_2: ┤ Ry(θ[2]) ├┤ Rz(θ[6]) ├─────■──────────┤ X ├────┤ Ry(θ[10]) ├┤ Rz(θ[14]) ├─────────────
         ├──────────┤├──────────┤   ┌─┴─┐    ┌───┴───┴───┐├───────────┤└───────────┘
    q_3: ┤ Ry(θ[3]) ├┤ Rz(θ[7]) ├───┤ X ├────┤ Ry(θ[11]) ├┤ Rz(θ[15]) ├──────────────────────────
         └──────────┘└──────────┘   └───┘    └───────────┘└───────────┘
    """

    def __init__(
        self,
        num_spatial_orbitals: int,
        num_particles: tuple[int, int],
        mapper: QubitMapper,
        num_qubits: int | None = None,
        su2_gates: (
            str
            | type
            | Instruction
            | QuantumCircuit
            | list[str | type | Instruction | QuantumCircuit]
            | None
        ) = None,
        entanglement: str | list[list[int]] | Callable[[int], list[int]] = "reverse_linear",
        reps: int = 3,
        skip_unentangled_qubits: bool = False,
        skip_final_rotation_layer: bool = False,
        parameter_prefix: str = "θ",
        insert_barriers: bool = False,
        name: str = "EfficientSU2",
    ) -> None:
        """Create a new EfficientSU2 2-local circuit with HartreeFock initial state.

        Args:
            mapper: A qubit mapper.
            num_spatial_orbitals: Number of spatial orbitals
                (num_spin_orbitals=2*num_spatial_orbitals),
            num_particles: Number of particles (electrons) in the system,
            num_qubits: The number of qubits of the EfficientSU2 circuit.
                If the ParityMapper is used, the correct number is automatically collected from
                the mapper and the explicitly passed value is ignored.
                In all other cases, the number of qubits must be passed explicitly.
                Typically, it can be derived from the qubit operator (qubit_operator = mapper.map())
                via qubit_operator.num_qubits.
                In typical workflows (with JordanWignerMapper) it can be derived via
                qiskit_nature.second_q.problems.ElectronicStructureProblem().num_spatial_orbitals*2.
                As last resort it falls back to num_particles which is only correct for
                JordanWignerMapper (and other mappers with the same number of qubits).
            reps: Specifies how often the structure of a rotation layer followed by an entanglement
                layer is repeated.
            su2_gates: The SU(2) single qubit gates to apply in single qubit gate layers.
                If only one gate is provided, the same gate is applied to each qubit.
                If a list of gates is provided, all gates are applied to each qubit in the provided
                order.
            entanglement: Specifies the entanglement structure. Can be a string
                ('full', 'linear', 'reverse_linear', 'circular' or 'sca'), a list of integer-pairs
                specifying the indices of qubits entangled with one another, or a callable
                returning such a list provided with the index of the entanglement layer.
                Default to 'reverse_linear' entanglement.
                Note that 'reverse_linear' entanglement provides the same unitary as 'full'
                with fewer entangling gates.
                See the Examples section of :class:`~qiskit.circuit.library.TwoLocal` for more
                detail.
            skip_unentangled_qubits: If True, the single qubit gates are only applied to qubits
                that are entangled with another qubit. If False, the single qubit gates are applied
                to each qubit in the Ansatz. Defaults to False.
            skip_final_rotation_layer: If False, a rotation layer is added at the end of the
                ansatz. If True, no rotation layer is added.
            parameter_prefix: The parameterized gates require a parameter to be defined, for which
                we use :class:`~qiskit.circuit.ParameterVector`.
            insert_barriers: If True, barriers are inserted in between each layer. If False,
                no barriers are inserted.
            name: Name of the circuit.

        """
        if su2_gates is None:
            su2_gates = [RYGate, RZGate]

        try:
            num_qubits = sum(mapper.num_particles)
        except AttributeError:
            num_qubits = num_qubits
        if num_qubits is None:
            num_qubits = 2 * num_spatial_orbitals

        initial_state = self._get_initial_state(
            entanglement=entanglement,
            reps=reps,
            num_particles=num_particles,
            num_spatial_orbitals=num_spatial_orbitals,
            num_qubits=num_qubits,
            mapper=mapper,
        )

        # EfficientSU2 does not insert a barrier between initial state and the following layers.
        # So, it must be added explicitly.
        if insert_barriers:
            initial_state.barrier()

        super().__init__(
            num_qubits=num_qubits,
            su2_gates=su2_gates,
            entanglement=entanglement,
            reps=reps,
            skip_unentangled_qubits=skip_unentangled_qubits,
            skip_final_rotation_layer=skip_final_rotation_layer,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            initial_state=initial_state,
            name=name,
        )

    def _get_initial_state(
        self,
        entanglement: str | list[list[int]] | Callable[[int], list[int]],
        reps: int,
        num_particles: tuple[int, int],
        num_spatial_orbitals: int,
        num_qubits: int,
        mapper: QubitMapper,
    ) -> QuantumCircuit:
        """Returns the initial state composed of 'x' gates.

        The initial state is created via prepending a HartreeFock initial state and
        the inverse of the EfficientSU2 circuit to the parameterized EfficientSU2 ansatz circuit.
        Thereby it is required that the initial_point for EfficientSU2 is the all-zero point
        (i.e. binding all parameters to the value zero for the first iteration).
        This allows to reduce the inverse EfficientSU2 circuit to solely the entangling layer
        (rotation layer turns out to be identity when binding all-zero parameters to it).
        Since the naive approach would yield an overhead of factor 2
        (the same amount of 'cx' gates as for EfficientSU2 prepended for the initial state),
        the 'cx' gates are evaluated using the knowledge that 'cx' on pure 0 or 1 states
        (i.e. those prepared by the HartreeFock initial state)
        can be directly evaluated classically without loosing any entanglement information.
        This procedure finally yields a HartreeFock initial state tailored
        especially for the EfficientSU2 ansatz circuit.
        """
        circuit_inverse = EfficientSU2(
            num_qubits=num_qubits,
            su2_gates=[],
            entanglement=entanglement,
            reps=reps,
            initial_state=None,
            name="EfficientSU2_inverse_no_rotation_layer",
        ).inverse()
        hartree_fock = HartreeFock(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            qubit_mapper=mapper,
        )

        # Create equivalent circuit for initial circuit using only x gates instead of cx gates.

        # Start with HartreeFock state
        bitstring = hartree_fock._bitstr  # noqa: SLF001

        # Continue with inverse circuit
        for instruction in circuit_inverse.decompose()._data:  # noqa: SLF001
            if not isinstance(instruction.operation, CXGate):
                raise ValueError(
                    f"Unexpected instruction {instruction.operation.name}. "
                    "Circuit preprocessing only works for 'cx' gates."
                )
            qubit_index_control = circuit_inverse.find_bit(instruction.qubits[0]).index
            qubit_index_target = circuit_inverse.find_bit(instruction.qubits[1]).index
            if bitstring[qubit_index_control]:
                bitstring[qubit_index_target] = not bitstring[qubit_index_target]

        initial_state = QuantumCircuit(num_qubits)
        for i, bit in enumerate(bitstring):
            if bit:
                initial_state.x(i)
        # initial_state = initial_state.to_gate(label="HartreeFock initial state for EfficientSU2")

        return initial_state

    @property
    def parameter_bounds(self) -> list[tuple[float, float]]:
        """Return the parameter bounds.

        Returns:
            The parameter bounds.
        """
        return self.num_parameters * [(-np.pi, np.pi)]

    @property
    def preferred_init_points(self) -> list[float]:
        """The initial points for the parameters. Should be all zero for the HF initial state.

        Returns:
            The initial values for the parameters
        """
        return np.zeros(self.num_parameters_settable)
