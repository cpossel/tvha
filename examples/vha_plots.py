"""Collection of plots for tVHA.

The plots help investigate tVHA and tune it.
"""

import itertools
import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from pyscf import gto, scf
from qiskit.circuit.library import BlueprintCircuit
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit_algorithms.minimum_eigensolvers import VQE, NumPyMinimumEigensolver
from qiskit_machine_learning.optimizers import SBPLX
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.algorithms.initial_points import HFInitialPoint
from qiskit_nature.second_q.circuit.library import UCC, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from tqdm import tqdm

from tvha.efficientsu2_hartreefock import EfficientSU2_HartreeFock
from tvha.fermionic_operator import FermionicOp
from tvha.tvha import VariationalHamiltonianAnsatz

logger = logging.getLogger(__name__)

# agg is a non-GUI backend so plt.show() does not work with it
plt.switch_backend("agg")  # quick fix for some runtime error in tkinter

_color_scheme_fhg = {
    "Weiß": (255, 255, 255),
    "Schwarz": (0, 0, 0),
    "Orange": (245, 130, 32),
    "Dunkelgrün": (23, 156, 125),  # Akzent 1
    "Dunkelblaugrün (mittel)": (0, 91, 127),  # Akzent 2
    "Eisblau": (166, 187, 200),  # Akzent 3
    "Dunkelblaugrün (hell)": (0, 133, 152),  # Akzent 4
    "Türkis": (57, 193, 205),  # Akzent 5
    "Gelbgrün": (178, 210, 53),  # Akzent 6
    "Aquamarin (dunkel)": (51, 124, 153),
    "Aquamarin (hell)": (102, 157, 178),
    "Himmelblau (dunkel)": (153, 189, 204),
    "Himmelblau (mittel)": (204, 222, 229),
    "Himmelblau (hell)": (229, 238, 242),
    "Dunkelblaugrün (dunkel)": (28, 63, 82),
    "Gelbbraun": (211, 199, 174),
    "Gold": (253, 185, 19),
    "Pflaume (hell)": (187, 0, 86),
    "Pflaume (dunkel)": (124, 21, 77),
}
# Rescale RGB values from [0, 255] to [0, 1]
color_scheme_fhg = {
    name: tuple(i / 255 for i in rgb_tuple) for name, rgb_tuple in _color_scheme_fhg.items()
}
color_circle = [
    color_scheme_fhg["Dunkelgrün"],
    color_scheme_fhg["Dunkelblaugrün (mittel)"],
    color_scheme_fhg["Eisblau"],
    color_scheme_fhg["Dunkelblaugrün (hell)"],
    color_scheme_fhg["Türkis"],
    color_scheme_fhg["Gelbgrün"],
]
colors_tvha = [
    x[0] for x in sorted(zip(color_circle, [0, 2, 1, 3, 4, 5], strict=False), key=lambda x: x[1])
]
colors_ucc = [color_circle[3], color_circle[5]]
color_hea = color_scheme_fhg["Orange"]
color_hf = color_scheme_fhg["Himmelblau (mittel)"]
color_fci = color_scheme_fhg["Gold"]


class VHAPlots:
    """Collection of plots for closer evaluation of tVHA method."""

    def __init__(
        self,
        output_path: Path,
        molecule_name: str,
        problem: ElectronicStructureProblem,
        mapper: FermionicMapper | None,
    ) -> None:
        """Initializes all needed variables for plotting of properties of tVHA."""
        self._epsilon = 1e-13  # tolerance for floats to be considered equal
        output_path.mkdir(exist_ok=True)
        self.output_path = output_path.resolve()
        self.file_energies = self.output_path.joinpath("energies_vha_statevector.csv")
        self.problem = problem
        self.mapper = mapper or JordanWignerMapper()
        # Dummy VHA in order to easily access the final threshold gamma
        self._vha_dummy = VariationalHamiltonianAnsatz(problem=self.problem, mapper=self.mapper)

        self.numerical_energies = self.get_numerical_energy()

        self.molecule_name = molecule_name
        self.energy_data_header, self.energy_data = self._get_energies_from_file(
            file=self.file_energies
        )

    def _get_final_thresholds_gamma(self, thresholds_gamma: Iterable[float]) -> list[float]:
        """Gets sorted list of final truncation thresholds with removed duplicates."""
        return sorted({self._vha_dummy.get_threshold_gamma(t)[1] for t in thresholds_gamma})

    def _get_final_threshold_gamma(self, threshold_gamma: float) -> float:
        """Gets final truncation threshold."""
        return self._vha_dummy.get_threshold_gamma(threshold_gamma)[1]

    def _get_energies_from_file(self, file: Path) -> tuple[tuple[str], list]:
        """Reads energy data from csv file.

        Args:
            file: csv file where the energy data is stored.

        Return:
            tuple(energy_data_header, energy_data)
        """
        try:
            data = pd.read_csv(
                file,
                index_col=False,
                converters={"optimal_parameters": lambda x: json.loads(x)},
            )
            energy_data_header = tuple(data.columns.tolist())
            data_clean = data.drop_duplicates(
                subset=("molecule_name", "trotter_steps", "threshold_gamma"),
                keep="first",
                ignore_index=True,
            )
            energy_data = data_clean.values.tolist()
        except FileNotFoundError:
            logger.info(
                "Can't find the file that is supposed to contain the energy data.", exc_info=True
            )
            energy_data_header = (
                "molecule_name",
                "energy",
                "trotter_steps",
                "threshold_gamma",
                "optimal_parameters",
            )
            energy_data = []
        return energy_data_header, energy_data

    def _flush_datapoint_to_file(self, energy_datapoint: list) -> None:
        """Flushs a single energy datapoint to the file 'self.file_energies' in append mode."""
        if not self.file_energies.exists() or self.file_energies.stat().st_size == 0:
            with self.file_energies.open("a") as file:
                file.write(",".join(self.energy_data_header) + "\n")
        with self.file_energies.open("a") as file:
            file.write(
                ",".join(
                    (
                        entry
                        if isinstance(entry, str)
                        else (
                            '"' + json.dumps(entry) + '"'
                            if isinstance(entry, Iterable)
                            else json.dumps(entry)
                        )
                    )
                    for entry in energy_datapoint
                )
                + "\n"
            )

    def _is_in_energy_data(
        self, trotter_steps: int, threshold_gamma: float, molecule_name: str | None = None
    ) -> bool:
        """Checks whether the energy data entry for given options exists.

        Checks for its existence in the attribute 'self.energy_data'.
        """
        try:
            self._get_energy(
                trotter_steps=trotter_steps,
                threshold_gamma=threshold_gamma,
                molecule_name=molecule_name,
            )
            return True
        except ValueError:
            return False

    def _calculate_missing_energies(
        self,
        list_of_trotter_steps: int | Iterable[int],
        thresholds_gamma: float | Iterable[float],
    ) -> None:
        """Calculates the energy for the given list of trotter steps and thresholds gamma.

        Only re-calculates them if they aren't already in self.energy_data yet.
        Saves the calculated energies both in the variable 'self.energy_data' and
        flushes them into the file 'self.file_energies'."""
        # Do some sanity checks on the input
        list_of_trotter_steps = (
            list_of_trotter_steps
            if isinstance(list_of_trotter_steps, Iterable)
            else [list_of_trotter_steps]
        )
        if any(not isinstance(i, int) for i in list_of_trotter_steps) or any(
            i < 1 for i in list_of_trotter_steps
        ):
            raise ValueError("Only positive integer values for Trotter steps allowed.")

        thresholds_gamma = (
            thresholds_gamma if isinstance(thresholds_gamma, Iterable) else [thresholds_gamma]
        )

        # Sorted list of desired thresholds with possible duplicates removed:
        final_thresholds_gamma = self._get_final_thresholds_gamma(thresholds_gamma=thresholds_gamma)

        missing_datapoints = []
        for trotter_steps in list_of_trotter_steps:
            for threshold_gamma in final_thresholds_gamma:
                if not self._is_in_energy_data(
                    trotter_steps=trotter_steps, threshold_gamma=threshold_gamma
                ):
                    missing_datapoints.append((trotter_steps, threshold_gamma))

        for trotter_steps, threshold_gamma in tqdm(
            missing_datapoints, desc="Energy calc", position=0, disable=len(missing_datapoints) <= 1
        ):
            energy, optimal_parameters, _ = self._calculate_statevector_energy(
                trotter_steps=trotter_steps, threshold_gamma=threshold_gamma
            )
            # Convert to python datatypes (should already be the case but to be sure)
            tmp_datapoint = {
                "molecule_name": str(self.molecule_name),
                "energy": float(energy),
                "trotter_steps": int(trotter_steps),
                "threshold_gamma": float(threshold_gamma),
                "optimal_parameters": [float(p) for p in optimal_parameters],
            }

            # Ensure proper order using the energy_data_header as reference
            new_datapoint = [tmp_datapoint[keyword] for keyword in self.energy_data_header]

            self.energy_data.append(new_datapoint)
            self._flush_datapoint_to_file(new_datapoint)

    def _get_energy(
        self,
        trotter_steps: int,
        threshold_gamma: float,
        molecule_name: str | None = None,
        return_full_datapoint: bool = False,
    ) -> float | list:
        """Gets the energy for given trotter steps and truncation threshold.

        Searches self.energy_data for it, so no previous check is performed whether the datapoint
        was calculated previously.
        """
        if molecule_name is None:
            molecule_name = self.molecule_name
        idx_molecule_name = self.energy_data_header.index("molecule_name")
        idx_trotter_steps = self.energy_data_header.index("trotter_steps")
        idx_threshold_gamma = self.energy_data_header.index("threshold_gamma")
        idx_energy = self.energy_data_header.index("energy")
        for datapoint in self.energy_data:
            if (
                datapoint[idx_molecule_name] == molecule_name
                and np.isclose(
                    datapoint[idx_trotter_steps],
                    trotter_steps,
                    rtol=self._epsilon,
                    atol=self._epsilon,
                )
                and np.isclose(
                    datapoint[idx_threshold_gamma],
                    threshold_gamma,
                    rtol=self._epsilon,
                    atol=self._epsilon,
                )
            ):
                if return_full_datapoint:
                    return datapoint
                return float(datapoint[idx_energy])
        raise ValueError(
            f"Energy for {trotter_steps} trotter steps and threshold γ "  # noqa: RUF001
            f"of {threshold_gamma} not found ({molecule_name})."
            "Please invoke '_calculate_missing_energies' first."
        )

    def get_energies(
        self,
        list_of_trotter_steps: Iterable[int] | int,
        thresholds_gamma: Iterable[float] | float,
        return_full_datapoint: bool = False,
    ) -> float | list | np.ndarray:
        """Gets the energy for given Trotter steps and truncation thresholds.

        Under the hood it calls '_calculate_missing_energies' and '_get_energy'.

        Returns:
            Array of energies; the 1st index resembles the one of list_of_trotter_steps,
            the 2nd index the one of thresholds_gamma.
            If any of the input has a length of one or is a scalar, the dimension of the returned
            numpy array is reduced.
            If both have a length of one or are scalars, the final output will also be a scalar
            (or the datapoint as list if return_full_datapoint is set to 'True').
            For example
                energies[i,j] returns the energy associated with
                Trotter step list_of_trotter_steps[i] and
                truncation threshold thresholds_gamma[j].
                energies[i] returns the energy associated with
                Trotter step list_of_trotter_steps[i]
                if len(thresholds_gamma)==1 or thresholds_gamma is a scalar;
                or it returns the energy associated with
                truncation threshold thresholds_gamma[j]
                if len(list_of_trotter_steps)==1 or list_of_trotter_steps is a scalar.
                If len(thresholds_gamma)==1 AND len(list_of_trotter_steps)==1 (or they are scalars),
                then a scalar float is returned.
        """
        if not isinstance(list_of_trotter_steps, Iterable):
            list_of_trotter_steps = [list_of_trotter_steps]
        if not isinstance(thresholds_gamma, Iterable):
            thresholds_gamma = [thresholds_gamma]
        final_thresholds_gamma = self._get_final_thresholds_gamma(thresholds_gamma=thresholds_gamma)

        self._calculate_missing_energies(
            list_of_trotter_steps=list_of_trotter_steps, thresholds_gamma=final_thresholds_gamma
        )

        energies = np.empty(
            (len(list_of_trotter_steps), len(thresholds_gamma)),
            dtype=object if return_full_datapoint else float,
        )
        for i, trotter_steps in enumerate(list_of_trotter_steps):
            for j, threshold_gamma in enumerate(final_thresholds_gamma):
                energies[i, j] = self._get_energy(
                    trotter_steps=trotter_steps,
                    threshold_gamma=threshold_gamma,
                    return_full_datapoint=return_full_datapoint,
                )
        energies_squeezed = np.squeeze(energies)

        if np.ndim(energies_squeezed) == 0:
            energies_squeezed = energies_squeezed.item()

        return energies_squeezed

    def _calculate_statevector_energy(
        self,
        trotter_steps: int = 1,
        threshold_gamma: float = 1.0,
        ansatz: BlueprintCircuit = None,
    ) -> tuple[float, list[float], BlueprintCircuit]:
        """Gets the ground state energy of statevector simulator calculation of tVHA.

        Args:
            trotter_steps: Number of steps for Trotter of the (adiabatic)
                time evolution.
            threshold_gamma: The truncation threshold to use for building the tVHA ansatz.
            ansatz: Only use this, if you explicitly need another ansatz than tVHA.
                If this arg is used, threshold_gamma and trotter_steps are silently discarded.

        Returns: tuple(energy, optimal_parameters, ansatz)
            energy: electronic part of groundstate energy of statevector calculation.
            optimal_parameters: ansatz parameters after optimization loop.
            ansatz: (VHA) ansatz used for the calculation
        """
        estimator = StatevectorEstimator()
        if ansatz is None:
            ansatz = VariationalHamiltonianAnsatz(
                problem=self.problem,
                trotter_steps=trotter_steps,
                threshold_gamma=threshold_gamma,
                mapper=self.mapper,
            )
        vqe = VQE(
            estimator=estimator,
            ansatz=ansatz,
            optimizer=SBPLX(max_evals=1000),
            initial_point=ansatz.get_initial_point(),
        )
        try:
            second_q_op = ansatz.hamilton_operator  # for VHA
        except AttributeError:
            second_q_op = self.problem.second_q_ops()[0]  # for EfficientSU2 and UCC
        result_vha_statevector = vqe.compute_minimum_eigenvalue(
            operator=self.mapper.map(second_q_ops=second_q_op)
        )
        energy = float(result_vha_statevector.eigenvalue)

        return (
            energy,
            result_vha_statevector.optimal_point.tolist(),
            ansatz,
        )

    def get_numerical_energy(self) -> dict[str, float]:
        """Gets the numerical solution of ground state energy calculation.

        Solution of the exact diagonalization of the Hamiltonian matrix eigenvalue problem,
        i.e. FCI (Full Configuration Interaction; supposed to be exact up to numerical precision).

        First attempts to read from file.
        If this does not exist, a new calculation is started
        (writing the result to a text file for possible consecutive runs).

        Returns: dict(energy_name, energy_value)
        """
        file = self.output_path.joinpath("energy_numerical.json")
        try:
            return json.loads(file.read_text())
        except (FileNotFoundError, TypeError):

            def filter_criterion(
                eigenstate: list | np.ndarray,  # noqa: ARG001
                eigenvalue: float,  # noqa: ARG001
                aux_values: dict[str, tuple[float | complex]],
            ) -> bool | np.bool_:
                eval_num_particles = aux_values.get("ParticleNumber")
                if eval_num_particles is None:
                    return True
                num_particles_close = np.isclose(
                    eval_num_particles[0], self.problem.num_alpha + self.problem.num_beta
                )

                eval_angular_momentum = aux_values.get("AngularMomentum")
                if eval_angular_momentum is None:
                    return num_particles_close
                spin = self.problem.num_alpha - self.problem.num_beta
                expected_angular_momentum = spin / 2 * (spin / 2 + 1)
                angular_momentum_close = np.isclose(
                    eval_angular_momentum[0], expected_angular_momentum
                )

                return num_particles_close and angular_momentum_close

            algo = NumPyMinimumEigensolver(filter_criterion=filter_criterion)
            solver = GroundStateEigensolver(self.mapper, algo)
            result_fci = solver.solve(self.problem)
            print(result_fci)
            numerical_energies = {
                "electronic_energy": result_fci.electronic_energies[0],
                "computed_energy": result_fci.computed_energies[0],
                "inactive_space_energy": result_fci.electronic_energies[0]
                - result_fci.computed_energies[0],
                "hartree_fock_energy": result_fci.hartree_fock_energy,
                "nuclear_repulsion_energy": result_fci.nuclear_repulsion_energy,
                "total_energy": result_fci.total_energies[0],
            }
            file.write_text(json.dumps(numerical_energies, sort_keys=False))
            return numerical_energies

    def get_reference_UCC_energy(  # noqa: N802
        self, excitations: Iterable[int] = (1, 2)
    ) -> dict[str, float | list[float]]:
        """Gets energy from calculation with UCCSD ansatz as reference.

        Args:
            excitations: By default singles and doubles (i.e. UCCSD); for other excitations, pass
                a list with the excitation numbers (1=singles, 2=doubles, 3=triples, 4=duadruples).

        Returns: dict(entry_name, value), where entry_name can be 'energy' and 'optimal_point'."""
        excitationstring = ""
        if 1 in excitations:
            excitationstring += "s"
        if 2 in excitations:
            excitationstring += "d"
        if 3 in excitations:
            excitationstring += "t"
        if 4 in excitations:
            excitationstring += "q"

        file = self.output_path.joinpath(f"energy_ucc{excitationstring}.json")
        try:
            return json.loads(file.read_text())
        except (FileNotFoundError, TypeError):
            ucc = UCC(
                excitations=list(excitations),
                num_spatial_orbitals=self.problem.num_spatial_orbitals,
                num_particles=self.problem.num_particles,
                qubit_mapper=self.mapper,
                initial_state=HartreeFock(
                    num_spatial_orbitals=self.problem.num_spatial_orbitals,
                    num_particles=self.problem.num_particles,
                    qubit_mapper=self.mapper,
                ),
            )
            initial_point = HFInitialPoint()
            initial_point.ansatz = ucc
            initial_point.problem = self.problem
            ucc.get_initial_point = initial_point.to_numpy_array

            energy, optimal_point, _ = self._calculate_statevector_energy(ansatz=ucc)
            file.write_text(
                json.dumps({"energy": energy, "optimal_point": optimal_point}, sort_keys=False)
            )
            return {"energy": energy, "optimal_point": optimal_point}

    def get_reference_HEA_energy(self) -> dict[str, float | list[float]]:  # noqa: N802
        """Gets energy from calculation with UCCSD ansatz as reference.

        Returns: dict(energy_name, energy_value)"""
        file = self.output_path.joinpath("energy_hea.json")
        try:
            return json.loads(file.read_text())
        except (FileNotFoundError, TypeError):
            hea = EfficientSU2_HartreeFock(
                num_spatial_orbitals=self.problem.num_spatial_orbitals,
                num_particles=self.problem.num_particles,
                mapper=self.mapper,
                num_qubits=len(self.mapper.map(self.problem.second_q_ops()[0])[0].paulis[0]),
                entanglement="reverse_linear",
                reps=3,
            )
            hea.get_initial_point = lambda: hea.preferred_init_points
            energy, optimal_point, _ = self._calculate_statevector_energy(ansatz=hea)
            file.write_text(
                json.dumps({"energy": energy, "optimal_point": optimal_point}, sort_keys=False)
            )
            return {"energy": energy, "optimal_point": optimal_point}

    def get_VHA_circuit_counts(self) -> pd.DataFrame:  # noqa: N802
        """Gets number of CNOT gates and circuit depth for VHA.

        If you are interested in the circuit count for a larger number of Trotter steps,
        just mulitply the circuit counts from this method with the number of Trotter steps.
        With heavy circuit optimization, one might be able to get slightly smaller circuits
        but still above way is a good approximation.
        """
        file = self.output_path.joinpath("circuit_counts_vha.csv")
        try:
            data = pd.read_csv(file, index_col=False)
            return data.drop_duplicates(
                subset="threshold_gamma", keep="first", ignore_index=True
            ).sort_values("threshold_gamma")
        except FileNotFoundError:
            circuit_data_header = (
                "threshold_gamma",
                "num_nonlocal_gates",
                "depth",
                "num_parameters",
            )
            circuit_data = []
            for threshold_gamma in tqdm(
                self._vha_dummy._get_possible_thresholds_gamma(),  # noqa: SLF001
                desc="Circuit counts",
                position=0,
            ):
                ansatz = VariationalHamiltonianAnsatz(
                    problem=self.problem,
                    mapper=self.mapper,
                    trotter_steps=1,
                    threshold_gamma=threshold_gamma,
                )
                circuit = ansatz.decompose(reps=4)
                circuit_data.append(
                    [
                        threshold_gamma,
                        circuit.num_nonlocal_gates(),
                        circuit.depth(),
                        circuit.num_parameters,
                    ]
                )
            data = pd.DataFrame(data=circuit_data, columns=circuit_data_header).sort_values(
                "threshold_gamma"
            )
            data.to_csv(file, index=False)
            return data

    def get_reference_UCC_circuit_counts(  # noqa: N802
        self, excitations: Iterable[int] = (1, 2)
    ) -> dict[str, float]:
        """Gets number of CNOT gates and circuit depth for UCCSD ansatz as reference.

        Args:
            excitations: By default singles and doubles (i.e. UCCSD); for other excitations, pass
                a list with the excitation numbers (1=singles, 2=doubles, 3=triples, 4=duadruples).

        Returns: dict(entry_name, count), where entry_name can be 'num_nonlocal_gates' and 'depth'.
        """
        excitationstring = ""
        if 1 in excitations:
            excitationstring += "s"
        if 2 in excitations:
            excitationstring += "d"
        if 3 in excitations:
            excitationstring += "t"
        if 4 in excitations:
            excitationstring += "q"

        file = self.output_path.joinpath(f"circuit_counts_ucc{excitationstring}.json")
        try:
            return json.loads(file.read_text())
        except (FileNotFoundError, TypeError):
            ansatz = UCC(
                excitations=list(excitations),
                num_spatial_orbitals=self.problem.num_spatial_orbitals,
                num_particles=self.problem.num_particles,
                qubit_mapper=self.mapper,
                initial_state=HartreeFock(
                    num_spatial_orbitals=self.problem.num_spatial_orbitals,
                    num_particles=self.problem.num_particles,
                    qubit_mapper=self.mapper,
                ),
            )
            circuit = ansatz.decompose(reps=4)
            circuit_counts = {
                "num_nonlocal_gates": circuit.num_nonlocal_gates(),
                "depth": circuit.depth(),
                "num_parameters": circuit.num_parameters,
            }
            file.write_text(json.dumps(circuit_counts, sort_keys=False))
            return circuit_counts

    def get_reference_HEA_circuit_counts(self) -> dict[str, float]:  # noqa: N802
        """Gets number of CNOT gates and circuit depth for hardware efficient ansatz as reference.

        Returns: dict(entry_name, count), where entry_name can be 'num_nonlocal_gates' and 'depth'.
        """
        file = self.output_path.joinpath("circuit_counts_hea.json")
        try:
            return json.loads(file.read_text())
        except (FileNotFoundError, TypeError):
            ansatz = EfficientSU2_HartreeFock(
                num_spatial_orbitals=self.problem.num_spatial_orbitals,
                num_particles=self.problem.num_particles,
                mapper=self.mapper,
                num_qubits=len(self.mapper.map(self.problem.second_q_ops()[0])[0].paulis[0]),
                entanglement="reverse_linear",
                reps=3,
            )
            circuit = ansatz.decompose(reps=4)
            circuit_counts = {
                "num_nonlocal_gates": circuit.num_nonlocal_gates(),
                "depth": circuit.depth(),
                "num_parameters": circuit.num_parameters,
            }
            file.write_text(json.dumps(circuit_counts, sort_keys=False))
            return circuit_counts

    def plot_histogram(
        self,
        hamiltonian: FermionicOp,
        one_body_terms: bool = True,
        two_body_terms_coulomb: bool = True,
        two_body_terms_noncoulomb: bool = True,
        log_x: bool = False,
        log_y: bool = False,
        add_title: bool = True,
    ) -> None:
        """Histogram of prefactors of the second quantization operators.

        Args:
            hamiltonian: the Hamiltonian of the system.
            one_body_terms: whether to include the one-body terms into the plot.
            two_body_terms_coulomb: whether to include the Coulomb two-body terms into the plot.
            two_body_terms_noncoulomb: whether to include the non-Coulomb two-body terms into the
                plot.
            log_x: whether to use a logarithmic scale for the x axis.
            log_y: whether to use a logarithmic scale for the y axis.
            add_title: whether to add a title to the plot.
        """
        coeffs, labels, colors = [], [], []
        if one_body_terms:
            coeffs_one_body = [
                abs(op.real) for op in hamiltonian.get_one_body_hamiltonian().values()
            ]
            coeffs.append(coeffs_one_body)
            labels.append("One-body terms")
            colors.append(color_circle[1])
        if two_body_terms_coulomb:
            coeffs_two_body_coulomb = [
                abs(op.real) for op in hamiltonian.get_two_body_hamiltonian_coulomb_terms().values()
            ]
            coeffs.append(coeffs_two_body_coulomb)
            labels.append("Two-body terms (Coulomb)")
            colors.append(color_scheme_fhg["Orange"])
        if two_body_terms_noncoulomb:
            coeffs_two_body_noncoulomb = [
                abs(op.real)
                for op in hamiltonian.get_two_body_hamiltonian_noncoulomb_terms().values()
            ]
            coeffs.append(coeffs_two_body_noncoulomb)
            labels.append("Two-body terms (non-Coulomb)")
            colors.append(color_circle[0])
        coeffs_flattened = list(itertools.chain.from_iterable(coeffs))

        if log_x:
            _, bins = np.histogram(np.log10(coeffs_flattened), bins="auto")
        else:
            bins = np.histogram_bin_edges(coeffs_flattened, bins="sqrt")
            if len(bins) < 20:
                bins = np.histogram_bin_edges(coeffs_flattened, bins="auto")
            if len(bins) < 20:
                bins = 20
        ax = plt.figure().gca()
        plt.hist(
            coeffs,
            log=log_y,
            bins=10**bins if log_x else bins,
            label=labels,
            color=colors,
            stacked=True,
        )
        if log_x:
            plt.xscale("log")
        _, ymax = plt.ylim()
        if log_y:
            plt.ylim(0.9, ymax)
        else:
            plt.ylim(0, ymax)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if add_title:
            plt.title(f"Terms of Hamiltonian (${self.molecule_name}$)")
        plt.xlabel("Magnitude of coefficient (absolute value in Hartree)")
        plt.ylabel("Number of terms")
        plt.legend()
        filename = f"{self.molecule_name}_hist"
        if one_body_terms and two_body_terms_coulomb and two_body_terms_noncoulomb:
            filename += "_all_terms"
        else:
            if one_body_terms:
                filename += "_onebody"
            if two_body_terms_coulomb:
                filename += "_twobodycoulomb"
            if two_body_terms_noncoulomb:
                filename += "_twobodynoncoulomb"
        if log_x:
            filename += "_logx"
        if log_y:
            filename += "_logy"
        filename += ".svg"
        plt.savefig(self.output_path.joinpath(filename), format="svg")
        plt.close()

    def plot_cumulated_density_distribution_all_terms(
        self, hamiltonian: FermionicOp, add_title: bool = True
    ) -> None:
        """Plot the cumulative density distribution for second quantization operators.

        Args:
            hamiltonian: the Hamiltonian of the system.
            add_title: whether to add a title to the plot.
        """
        coeffs = np.sort([abs(op) for op in hamiltonian.get_compressed_hamiltonian().values()])[
            ::-1
        ]
        coeffs = np.insert(coeffs, 0, 0.0)

        cumsum = coeffs.cumsum()  # cumulative sum of elements
        prefactors_sum = coeffs.sum()  # overall sum of all prefactors
        logger.debug("Number of terms: %s", len(coeffs))
        plt.plot(cumsum / prefactors_sum, linestyle="--", marker="x")
        if add_title:
            plt.title(
                f"Cumulated density distribution of all one- and two-body terms (${self.molecule_name}$)"
            )
        plt.xlabel("Number of terms")
        plt.ylabel("Cumulated normalized weight")
        plt.grid(visible=True)
        plt.savefig(
            self.output_path.joinpath(
                f"{self.molecule_name}_cumulated_density_distribution_all_terms.svg"
            ),
            format="svg",
        )
        plt.close()

    def plot_cumulated_density_distribution_noncoulomb_terms(
        self, hamiltonian: FermionicOp, add_title: bool = True
    ) -> None:
        """Plot the cumulative density distribution for second quantization operators.

        Args:
            hamiltonian: the Hamiltonian of the system.
            add_title: whether to add a title to the plot.
        """
        coeffs_two_body = np.sort(
            [
                abs(op)
                for op in hamiltonian.get_two_body_hamiltonian_noncoulomb_terms()
                .get_compressed_hamiltonian()
                .values()
            ]
        )[::-1]
        coeffs_two_body = np.insert(coeffs_two_body, 0, 0.0)

        cumsum = coeffs_two_body.cumsum()  # cumulative sum of elements
        prefactors_sum = coeffs_two_body.sum()  # overall sum of all prefactors
        logger.debug("Number of two-body terms: %s", len(coeffs_two_body))
        plt.plot(cumsum / prefactors_sum, linestyle="--", marker="x")
        if add_title:
            plt.title(
                f"Cumulated density distribution of non-Coulomb two-body terms (${self.molecule_name}$)"
            )
        plt.xlabel("Number of terms")
        plt.ylabel("Cumulated normalized weight")
        plt.grid(visible=True)
        plt.savefig(
            self.output_path.joinpath(
                f"{self.molecule_name}_cumulated_density_distribution_noncoulomb_terms.svg"
            ),
            format="svg",
        )
        plt.close()

    def plot_cnot_count_over_truncation_threshold(
        self,
        log_y: bool = False,
        add_title: bool = True,
    ) -> None:
        """Plots CNOT count of tVHA depending on the truncation threshold.

        Args:
            log_y: whether to use a logarithmic scale for the y axis.
            add_title: whether to add a title to the plot.
        """
        circuit_data = self.get_VHA_circuit_counts()

        ax = plt.figure().gca()

        # tVHA
        plt.plot(
            circuit_data.threshold_gamma,
            circuit_data.num_nonlocal_gates,
            label="tVHA",
            marker="x",
            linestyle="dotted",
            color=colors_tvha[0],
        )
        xmin, xmax = plt.xlim()

        # UCCSD
        circuit_counts_uccsd = self.get_reference_UCC_circuit_counts()
        plt.hlines(
            y=circuit_counts_uccsd["num_nonlocal_gates"],
            xmin=xmin,
            xmax=xmax,
            label="UCCSD",
            linestyle="dashdot",
            color=colors_ucc[0],
        )

        # UCCSDT
        circuit_counts_uccsdt = self.get_reference_UCC_circuit_counts(excitations=[1, 2, 3])
        if (
            circuit_counts_uccsdt["num_nonlocal_gates"]
            != circuit_counts_uccsd["num_nonlocal_gates"]
        ):
            # In case of H2, there are no triple excitations
            plt.hlines(
                y=circuit_counts_uccsdt["num_nonlocal_gates"],
                xmin=xmin,
                xmax=xmax,
                label="UCCSDT",
                linestyle="dashdot",
                color=colors_ucc[1],
            )

        # HEA
        circuit_counts_hea = self.get_reference_HEA_circuit_counts()
        plt.hlines(
            y=circuit_counts_hea["num_nonlocal_gates"],
            xmin=xmin,
            xmax=xmax,
            label="HEA",
            linestyle="dashed",
            color=color_hea,
        )

        plt.xlim(xmin, xmax)

        plt.yscale("log" if log_y else "linear")
        if not log_y:
            _, ymax = plt.ylim()
            plt.ylim(0, ymax)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        if add_title:
            plt.title(f"Two-qubit gate count for tVHA, UCC and HEA (${self.molecule_name}$)")
        plt.xlabel("Truncation threshold")
        plt.ylabel("Number of CNOTs")
        plt.legend()

        filename = f"{self.molecule_name}_cnot_count_over_truncation_threshold"
        if log_y:
            filename += "_logy"
        filename += ".svg"
        plt.savefig(self.output_path.joinpath(filename), format="svg")
        plt.close()

    def plot_circuit_depth_over_truncation_threshold(
        self,
        log_y: bool = False,
        add_cnot_count: bool = True,
        add_title: bool = True,
    ) -> None:
        """Plots circuit depth (and CNOT count) of tVHA depending on the truncation threshold.

        Args:
            log_y: whether to use a logarithmic scale for the y axis.
            add_cnot_count: whether to include the number of CNOTs to the plot.
            add_title: whether to add a title to the plot.
        """
        circuit_data = self.get_VHA_circuit_counts()

        ax = plt.figure().gca()
        labels = []

        label_string = "Circuit depth / # CNOTs" if add_cnot_count else "Circuit depth"

        # tVHA
        plt.plot(
            circuit_data.threshold_gamma,
            circuit_data.depth,
            label="Circuit depth (tVHA)" if add_cnot_count else "tVHA",
            marker="x",
            linestyle="dotted",
            color=colors_tvha[0],
        )
        if add_cnot_count:
            plt.plot(
                circuit_data.threshold_gamma,
                circuit_data.num_nonlocal_gates,
                label="# CNOTs (tVHA)",
                marker="2",
                linestyle="dashed",
                color=colors_tvha[0],
            )
        labels.append(label_string + " (tVHA)")
        xmin, xmax = plt.xlim()

        # UCCSD
        circuit_counts_uccsd = self.get_reference_UCC_circuit_counts()
        plt.hlines(
            y=circuit_counts_uccsd["depth"],
            xmin=xmin,
            xmax=xmax,
            label="Circuit depth (UCCSD)" if add_cnot_count else "UCCSD",
            linestyle="dashdot",
            color=colors_ucc[0],
        )
        if add_cnot_count:
            plt.hlines(
                y=circuit_counts_uccsd["num_nonlocal_gates"],
                xmin=xmin,
                xmax=xmax,
                label="# CNOTs (UCCSD)",
                linestyle="dashed",
                color=colors_ucc[0],
            )
        labels.append(label_string + " (UCCSD)")

        # UCCSDT
        circuit_counts_uccsdt = self.get_reference_UCC_circuit_counts(excitations=[1, 2, 3])
        # In case of H2, there are no triple excitations
        if (
            circuit_counts_uccsdt["num_nonlocal_gates"]
            != circuit_counts_uccsd["num_nonlocal_gates"]
        ):
            plt.hlines(
                y=circuit_counts_uccsdt["depth"],
                xmin=xmin,
                xmax=xmax,
                label="Circuit depth (UCCSDT)" if add_cnot_count else "UCCSDT",
                linestyle="dashdot",
                color=colors_ucc[1],
            )
            if add_cnot_count:
                plt.hlines(
                    y=circuit_counts_uccsdt["num_nonlocal_gates"],
                    xmin=xmin,
                    xmax=xmax,
                    label="# CNOTs (UCCSDT)",
                    linestyle="dashed",
                    color=colors_ucc[1],
                )
            labels.append(label_string + " (UCCSDT)")

        # HEA
        circuit_counts_hea = self.get_reference_HEA_circuit_counts()
        plt.hlines(
            y=circuit_counts_hea["depth"],
            xmin=xmin,
            xmax=xmax,
            label="Circuit depth (HEA)" if add_cnot_count else "HEA",
            linestyle="dashed",
            color=color_hea,
        )
        if add_cnot_count:
            plt.hlines(
                y=circuit_counts_hea["num_nonlocal_gates"],
                xmin=xmin,
                xmax=xmax,
                label="# CNOTs (HEA)",
                linestyle="dotted",
                color=color_hea,
            )
        labels.append(label_string + " (HEA)")

        plt.xlim(xmin, xmax)

        plt.yscale("log" if log_y else "linear")
        if not log_y:
            _, ymax = plt.ylim()
            plt.ylim(0, ymax)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        from matplotlib.legend_handler import HandlerTuple

        if add_title:
            plt.title(f"Circuit depth for tVHA, UCC and HEA (${self.molecule_name}$)")
        plt.xlabel("Truncation threshold")
        plt.ylabel("Circuit depth")
        handles, _ = plt.gca().get_legend_handles_labels()
        if add_cnot_count:
            plt.legend(
                handles=zip(handles[::2], handles[1::2], strict=True),
                labels=labels,
                handler_map={tuple: HandlerTuple(None)},
                handlelength=len(handles) - 1,
            )
        else:
            plt.legend(handles=handles, labels=labels)

        filename = f"{self.molecule_name}_circuit_depth_over_truncation_threshold"
        if log_y:
            filename += "_logy"
        filename += ".svg"
        plt.savefig(self.output_path.joinpath(filename), format="svg")
        plt.close()

    def plot_energy_over_truncation_threshold(
        self,
        trotter_steps: int | Iterable[int] = 1,
        thresholds_gamma: Iterable[float] | None = None,
        add_title: bool = True,
    ) -> None:
        """Plots energy of tVHA depending on the truncation threshold.

        Args:
            trotter_steps: the number of Trotter steps to use.
                If given as single element, only a single line is plotted.
                If given as list, all list elements are used in sorted order
                creating a line for each number of Trotter steps.
            thresholds_gamma: the truncation thresholds to use for this plot.
                If 'None', all possible truncation thresholds are used.
            add_title: whether to add a title to the plot.
        """
        list_of_trotter_steps = (
            sorted(trotter_steps) if isinstance(trotter_steps, Iterable) else [trotter_steps]
        )
        if thresholds_gamma is None:
            thresholds_gamma = self._vha_dummy.possible_thresholds_gamma
        else:
            thresholds_gamma = self._get_final_thresholds_gamma(thresholds_gamma=thresholds_gamma)

        if len(list_of_trotter_steps) > len(colors_tvha):
            raise ValueError(
                "The energy over truncation threshold plot is not intended for a "
                "large amount of Trotter steps. Please reduce them from "
                f"{len(list_of_trotter_steps)} to at most {len(colors_tvha)}.",
            )

        self._calculate_missing_energies(
            list_of_trotter_steps=list_of_trotter_steps, thresholds_gamma=thresholds_gamma
        )
        energies = {
            trotter_steps: tuple(
                self._get_energy(trotter_steps=trotter_steps, threshold_gamma=i)
                for i in thresholds_gamma
            )
            for trotter_steps in list_of_trotter_steps
        }

        # tVHA
        for idx, trotter_steps in enumerate(list_of_trotter_steps):
            if trotter_steps == 1:
                label = f"tVHA ({trotter_steps} Trotter step)"
            else:
                label = f"tVHA ({trotter_steps} Trotter steps)"
            plt.plot(
                thresholds_gamma,
                energies[trotter_steps],
                label=label,
                marker="x",
                linestyle="dotted",
                linewidth=0.8,
                color=colors_tvha[idx],
            )
        xmin, xmax = plt.xlim()

        labels_close_to_hf = ["HF"]
        labels_close_to_fci = ["FCI"]

        energy_hf = (
            self.problem.reference_energy
            - self.numerical_energies["nuclear_repulsion_energy"]
            - self.numerical_energies["inactive_space_energy"]
        )
        energy_fci = self.numerical_energies["computed_energy"]

        # UCC
        energy_uccsd = self.get_reference_UCC_energy()["energy"]
        energy_uccsdt = self.get_reference_UCC_energy(excitations=[1, 2, 3])["energy"]

        # UCCSD
        if np.isclose(energy_uccsd, energy_fci):
            labels_close_to_fci.append("UCCSD")
        elif np.isclose(energy_uccsd, energy_hf):
            labels_close_to_hf.append("UCCSD")
        else:
            plt.hlines(
                y=energy_uccsd,
                xmin=xmin,
                xmax=xmax,
                label="UCCSD / UCCSDT" if np.isclose(energy_uccsdt, energy_uccsd) else "UCCSD",
                linestyles="dashdot",
                color=colors_ucc[0],
            )

        # UCCSDT
        if np.isclose(energy_uccsdt, energy_fci):
            labels_close_to_fci.append("UCCSDT")
        elif np.isclose(energy_uccsdt, energy_hf):
            labels_close_to_hf.append("UCCSDT")
        elif not np.isclose(energy_uccsdt, energy_uccsd):
            plt.hlines(
                y=energy_uccsdt,
                xmin=xmin,
                xmax=xmax,
                label="UCCSDT",
                linestyles="dashdot",
                color=colors_ucc[1],
            )

        # HEA
        energy_hea = self.get_reference_HEA_energy()["energy"]
        if np.isclose(energy_hea, energy_fci):
            labels_close_to_fci.append("HEA")
        elif np.isclose(energy_uccsd, energy_hf):
            labels_close_to_hf.append("HEA")
        else:
            plt.hlines(
                y=energy_hea,
                xmin=xmin,
                xmax=xmax,
                label="HEA",
                linestyles="dashed",
                color=color_hea,
            )

        # HF energy
        plt.hlines(
            energy_hf,
            xmin=xmin,
            xmax=xmax,
            label=" / ".join(labels_close_to_hf),
            color=color_hf,
            linestyles="solid",
            zorder=0,
        )

        # FCI energy
        plt.hlines(
            self.numerical_energies["computed_energy"],
            xmin=xmin,
            xmax=xmax,
            label=" / ".join(labels_close_to_fci),
            color=color_fci,
            linestyles="solid",
            zorder=0,
        )
        plt.fill_between(
            (xmin, xmax),
            self.numerical_energies["computed_energy"],
            self.numerical_energies["computed_energy"] + 0.0015,
            label="chemical accuracy",
            color=color_fci,
            alpha=0.4,
            zorder=0,
        )

        plt.legend()
        plt.xlabel("Truncation threshold")
        plt.ylabel("Energy in Hartree")
        if add_title:
            plt.title(
                f"Energy of tVHA depending on the truncation threshold (${self.molecule_name}$)"
            )
        plt.xlim(xmin, xmax)
        filename = f"{self.molecule_name}_energy_over_truncation_threshold"
        filename += "_" + "_".join(str(trotter_steps) for trotter_steps in list_of_trotter_steps)
        filename += ".svg"
        plt.savefig(self.output_path.joinpath(filename), format="svg")
        plt.close()

    def plot_energy_over_trotter_steps(
        self,
        list_of_trotter_steps: Iterable[int] = (1, 2, 3, 4, 5),
        threshold_gamma: float | Iterable[float] = 1.0,
        add_title: bool = True,
    ) -> None:
        """Plots the energy of tVHA depending on the number of Trotter steps.

        Args:
            list_of_trotter_steps: the numbers of Trotter steps to use.
            threshold_gamma: the truncation threshold to use for this plot.
                If given as single element, only a single line is plotted.
                If given as list, all list elements are used in sorted order
                creating a line for each truncation threshold.
            add_title: whether to add a title to the plot.

        """
        list_of_trotter_steps = sorted(list_of_trotter_steps)
        if isinstance(threshold_gamma, Iterable):
            thresholds_gamma = self._get_final_thresholds_gamma(thresholds_gamma=threshold_gamma)
        else:
            thresholds_gamma = [self._get_final_threshold_gamma(threshold_gamma=threshold_gamma)]

        if len(thresholds_gamma) > len(colors_tvha):
            raise ValueError(
                "The energy over Trotter steps plot is not intended for a "
                "large amount of Trotter steps. Please reduce them from "
                f"{len(list_of_trotter_steps)} to at most {len(colors_tvha)}.",
            )

        self._calculate_missing_energies(
            list_of_trotter_steps=list_of_trotter_steps, thresholds_gamma=thresholds_gamma
        )
        energies = {
            threshold_gamma: tuple(
                self._get_energy(trotter_steps=i, threshold_gamma=threshold_gamma)
                for i in list_of_trotter_steps
            )
            for threshold_gamma in thresholds_gamma
        }

        ax = plt.figure().gca()

        # tVHA
        for idx, threshold_gamma in enumerate(thresholds_gamma):
            plt.plot(
                list_of_trotter_steps,
                energies[threshold_gamma],
                label=f"tVHA (truncation threshold {threshold_gamma:.4g})",
                marker="x",
                linestyle="dotted",
                linewidth=0.8,
                color=colors_tvha[idx],
            )

        xmin, xmax = plt.xlim()

        labels_close_to_hf = ["HF"]
        labels_close_to_fci = ["FCI"]

        energy_hf = (
            self.problem.reference_energy
            - self.numerical_energies["nuclear_repulsion_energy"]
            - self.numerical_energies["inactive_space_energy"]
        )
        energy_fci = self.numerical_energies["computed_energy"]

        # UCCSD
        energy_uccsd = self.get_reference_UCC_energy()["energy"]
        if np.isclose(energy_uccsd, energy_fci):
            labels_close_to_fci.append("UCCSD")
        elif np.isclose(energy_uccsd, energy_hf):
            labels_close_to_hf.append("UCCSD")
        else:
            plt.hlines(
                y=energy_uccsd,
                xmin=xmin,
                xmax=xmax,
                label="UCCSD",
                linestyles="dashdot",
                color=colors_ucc[0],
            )

        # UCCSDT
        energy_uccsdt = self.get_reference_UCC_energy(excitations=[1, 2, 3])["energy"]
        if np.isclose(energy_uccsdt, energy_fci):
            labels_close_to_fci.append("UCCSDT")
        elif not np.isclose(energy_uccsdt, energy_uccsd):
            plt.hlines(
                y=energy_uccsdt,
                xmin=xmin,
                xmax=xmax,
                label="UCCSDT",
                linestyles="dashdot",
                color=colors_ucc[1],
            )

        # HEA
        energy_hea = self.get_reference_HEA_energy()["energy"]
        if np.isclose(energy_hea, energy_fci):
            labels_close_to_fci.append("HEA")
        elif np.isclose(energy_uccsd, energy_hf):
            labels_close_to_hf.append("HEA")
        else:
            plt.hlines(
                y=energy_hea,
                xmin=xmin,
                xmax=xmax,
                label="HEA",
                linestyles="dashed",
                color=color_hea,
            )

        # HF energy
        plt.hlines(
            energy_hf,
            xmin=xmin,
            xmax=xmax,
            label=" / ".join(labels_close_to_hf),
            color=color_hf,
            linestyles="solid",
            zorder=0,
        )

        # FCI energy
        plt.hlines(
            self.numerical_energies["computed_energy"],
            xmin=xmin,
            xmax=xmax,
            label=" / ".join(labels_close_to_fci),
            color=color_fci,
            linestyles="solid",
            zorder=0,
        )
        plt.fill_between(
            (xmin, xmax),
            self.numerical_energies["computed_energy"],
            self.numerical_energies["computed_energy"] + 0.0015,
            label="Chemical accuracy",
            color=color_fci,
            alpha=0.4,
            zorder=0,
        )

        plt.legend()
        plt.xlabel("Trotter steps")
        plt.ylabel("Energy in Hartree")
        if add_title:
            plt.title(
                f"Energy of tVHA denpending on the number of Trotter steps (${self.molecule_name}$)"
            )
        plt.xlim(xmin, xmax)
        filename = f"{self.molecule_name}_energy_over_trotter_steps"
        filename += "_" + "_".join(f"{threshold_gamma:.4g}" for threshold_gamma in thresholds_gamma)
        filename += ".svg"
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(self.output_path.joinpath(filename), format="svg")
        plt.close()

    def plot_energy_over_truncation_threshold_and_trotter_steps(
        self,
        list_of_trotter_steps: Iterable[int] = (1, 2, 3, 4, 5),
        thresholds_gamma: Iterable[float] | None = None,
        add_title: bool = True,
    ) -> None:
        """Plots energy of tVHA depending on truncation threshold and Trotter steps as heatmap.

        Args:
            list_of_trotter_steps: the numbers of Trotter steps to use.
            thresholds_gamma: the truncation thresholds to use for this plot.
                If 'None', all possible truncation thresholds are used.
            add_title: whether to add a title to the plot.
        """
        list_of_trotter_steps = sorted(list_of_trotter_steps)
        if thresholds_gamma is None:
            thresholds_gamma = self._vha_dummy.possible_thresholds_gamma
        else:
            thresholds_gamma = self._get_final_thresholds_gamma(thresholds_gamma=thresholds_gamma)

        self._calculate_missing_energies(
            list_of_trotter_steps=list_of_trotter_steps, thresholds_gamma=thresholds_gamma
        )

        A, B = np.meshgrid(thresholds_gamma, list_of_trotter_steps, indexing="ij")  # noqa: N806

        def _get_energy_tmp(i: int, j: int) -> float | list:
            return self._get_energy(
                threshold_gamma=thresholds_gamma[int(i)],
                trotter_steps=list_of_trotter_steps[int(j)],
            )

        energies_reshaped = np.fromfunction(
            np.vectorize(_get_energy_tmp),
            shape=A.shape,
        )

        ax = plt.figure().gca()
        plt.pcolormesh(A, B, energies_reshaped, cmap="hot", edgecolors="face")
        plt.colorbar()
        plt.xlabel("Truncation threshold")
        plt.ylabel("Number of Trotter steps")
        if add_title:
            plt.title(f"Energy in Hartree (${self.molecule_name}$)")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(
            self.output_path.joinpath(
                f"{self.molecule_name}_energy_heatmap_threshold_trottersteps.svg"
            ),
            format="svg",
        )
        plt.close()

    def plot_energy_landscape(
        self,
        alphas: Iterable[float] = np.linspace(0, 30, 10),
        betas: Iterable[float] = np.linspace(0, 30, 10),
        gammas: Iterable[float] = np.linspace(0, 30, 10),
    ) -> None:
        """Plots the energy landscape for parameters alpha and beta in the VHA ansatz.

        This plot makes only sense for a single Trotter step;
        else it would be a multidimensional plot due to the large amount of parameters.

        In contrast to the other plot methods, this one is quite inefficient since it does not
        store calculated energy values. So, every run inefficiently re-calculates all energy values.
        """
        alphas = list(alphas)
        betas = list(betas)
        gammas = list(gammas)

        A, B, C = np.meshgrid(alphas, betas, gammas, indexing="ij")  # noqa: N806
        A_small, B_small = np.meshgrid(alphas, betas, indexing="ij")  # noqa: N806
        parameter_values = list(zip(A.flatten(), B.flatten(), C.flatten(), strict=True))

        ansatz_vha = VariationalHamiltonianAnsatz(
            problem=self.problem, trotter_steps=1, mapper=self.mapper
        )
        estimator = StatevectorEstimator()
        job = estimator.run(
            circuits=[ansatz_vha] * len(parameter_values),
            observables=[self.mapper.map(second_q_ops=ansatz_vha.hamilton_operator)]
            * len(parameter_values),
            parameter_values=parameter_values,
        )
        result = job.result()
        energies = result.values.reshape(A.shape)

        # Minimum energy and alpha and beta values belonging to it
        alpha_min = A[np.unravel_index(energies.argmin(), energies.shape)]
        beta_min = B[np.unravel_index(energies.argmin(), energies.shape)]
        gamma_min = C[np.unravel_index(energies.argmin(), energies.shape)]
        print(
            f"Minimum energy (from energy landscape) {energies.min():.3f} Hartree with "
            f"α={alpha_min:.3g}, β={beta_min:.3g}, γ={gamma_min:.3g}"  # noqa: RUF001
        )

        period_alpha = []
        for beta_index in range(len(betas)):
            for gamma_index in range(len(gammas)):
                n = energies[:, beta_index, gamma_index].size
                dt = abs(alphas[1] - alphas[0])

                rfft = np.fft.rfft(energies[:, beta_index, gamma_index])
                coeffs_argmax = rfft[1:].argmax() + 1
                freq_max = np.fft.rfftfreq(n)[coeffs_argmax]
                period_alpha.append(dt / freq_max)
        period_beta = []
        for alpha_index in range(len(alphas)):
            for gamma_index in range(len(gammas)):
                n = energies[alpha_index, :, gamma_index].size
                dt = abs(betas[1] - betas[0])

                rfft = np.fft.rfft(energies[alpha_index, :, gamma_index])
                coeffs_argmax = rfft[1:].argmax() + 1
                freq_max = np.fft.rfftfreq(n)[coeffs_argmax]
                period_beta.append(dt / freq_max)
        period_gamma = []
        for alpha_index in range(len(alphas)):
            for beta_index in range(len(betas)):
                n = energies[alpha_index, beta_index, :].size
                dt = abs(gammas[1] - gammas[0])

                rfft = np.fft.rfft(energies[alpha_index, beta_index, :])
                coeffs_argmax = rfft[1:].argmax() + 1
                freq_max = np.fft.rfftfreq(n)[coeffs_argmax]
                period_gamma.append(dt / freq_max)
        logger.info("period_a %s", period_alpha)
        logger.info("period_a %s", period_beta)
        logger.info("period_a %s", period_gamma)
        logger.info("period_a (final value) %s", np.bincount(period_alpha).argmax())

        import typing

        import matplotlib as mpl

        mpl.use("TKAgg")

        # ---------------------------------------------------------------------
        # Plotting with animation
        import matplotlib.widgets
        import mpl_toolkits.axes_grid1
        from matplotlib.animation import FuncAnimation
        from matplotlib.figure import Figure

        class Player(FuncAnimation):
            @typing.no_type_check
            def __init__(
                self,
                fig,  # noqa: ANN001
                func,  # noqa: ANN001
                frames=None,  # noqa: ANN001, ARG002
                init_func=None,  # noqa: ANN001
                fargs=None,  # noqa: ANN001
                save_count=None,  # noqa: ANN001
                mini=0,  # noqa: ANN001
                maxi=100,  # noqa: ANN001
                repeating: bool = False,
                pos=(0.125, 0.92),  # noqa: ANN001
                **kwargs,  # noqa: ANN003
            ) -> None:
                self.i = 0
                self.min = mini
                self.max = maxi
                self.runs = True
                self.forwards = True
                self.fig = fig
                self.repeating = repeating
                self.func = func
                self.setup(pos)
                FuncAnimation.__init__(
                    self,
                    fig=self.fig,
                    func=self.func,
                    frames=self.play(),
                    # repeat=self.repeating,
                    init_func=init_func,
                    fargs=fargs,
                    save_count=save_count,
                    **kwargs,
                )

            @typing.no_type_check
            def play(self) -> Iterator[int]:
                while self.runs:
                    self.i = self.i + 1 if self.forwards else self.i - 1
                    if self.i > self.min and self.i < self.max:
                        yield self.i
                    elif self.repeating:
                        if self.i == self.max or self.min:
                            yield self.i
                        elif self.i > self.max:
                            self.i = self.min
                            yield self.i
                        else:  # if self.i < self.min
                            self.i = self.max
                            yield self.i
                    else:
                        self.pause()
                        yield self.i

            @typing.no_type_check
            def start(self, event=None) -> None:  # noqa: ARG002, ANN001
                self.runs = True
                self.event_source.start()

            @typing.no_type_check
            def pause(self, event=None) -> None:  # noqa: ARG002, ANN001
                self.runs = False
                self.event_source.stop()

            @typing.no_type_check
            def forward(self, event=None) -> None:  # noqa: ARG002, ANN001
                self.forwards = True
                self.start()

            @typing.no_type_check
            def backward(self, event=None) -> None:  # noqa: ARG002, ANN001
                self.forwards = False
                self.start()

            @typing.no_type_check
            def oneforward(self, event=None) -> None:  # noqa: ARG002, ANN001
                self.forwards = True
                self.onestep()

            @typing.no_type_check
            def onebackward(self, event=None) -> None:  # noqa: ARG002, ANN001
                self.forwards = False
                self.onestep()

            @typing.no_type_check
            def on_scroll(self, event) -> None:  # noqa: ANN001
                tmp_forwards = self.forwards
                self.forwards = event.button == "up"
                self.onestep()
                self.forwards = tmp_forwards

            @typing.no_type_check
            def onestep(self) -> None:
                if self.i > self.min and self.i < self.max:
                    # increment (decrement) by 1
                    self.i = self.i + 1 if self.forwards else self.i - 1
                else:
                    if self.repeating:
                        # jump to minimum (maximum)
                        self.i = self.min if self.forwards else self.max
                    else:
                        # stay at maximum (minimum)
                        self.i = self.max if self.forwards else self.min

                self.func(self.i)
                self.fig.canvas.draw_idle()

            @typing.no_type_check
            def setup(self, pos) -> None:  # noqa: ANN001
                playerax = self.fig.add_axes([pos[0], pos[1], 0.22, 0.04])
                divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
                obax = divider.append_axes("right", size="80%", pad=0.05)
                sax = divider.append_axes("right", size="80%", pad=0.05)
                ofax = divider.append_axes("right", size="80%", pad=0.05)
                fax = divider.append_axes("right", size="100%", pad=0.05)
                self.button_back = matplotlib.widgets.Button(playerax, label="$\u25c0$")
                self.button_oneback = matplotlib.widgets.Button(obax, label="$\u29cf$")
                self.button_pause = matplotlib.widgets.Button(sax, label="$\u25a0$")
                self.button_oneforward = matplotlib.widgets.Button(ofax, label="$\u29d0$")
                self.button_forward = matplotlib.widgets.Button(fax, label="$\u25b6$")
                self.button_oneback.on_clicked(self.onebackward)
                self.button_back.on_clicked(self.backward)
                self.button_pause.on_clicked(self.pause)
                self.button_forward.on_clicked(self.forward)
                self.button_oneforward.on_clicked(self.oneforward)
                self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)

        class AnimationWithButtons:
            def __init__(
                self,
                fig: Figure,
                ax: plt.Axes,
                A: np.ndarray,  # noqa: N803
                B: np.ndarray,  # noqa: N803
                gammas: Iterable[float],
                data: np.ndarray,
            ) -> None:
                self.index = 0
                self.gammas = gammas
                self.data = data
                self.fig = fig
                self.ax = ax
                self.vmin = data.min()
                self.vmax = data.max()

                norm = mpl.colors.PowerNorm(vmin=self.vmin, vmax=self.vmax, gamma=0.4)
                self.im = self.ax.pcolormesh(
                    A, B, self.data[:, :, self.index], cmap="hot", norm=norm
                )
                self.fig.colorbar(self.im)

                self.anim = Player(
                    fig=self.fig,
                    func=self.update,
                    mini=0,
                    maxi=len(self.gammas) - 1,
                    repeating=False,
                    interval=500,
                    blit=False,
                    pos=(0.05, 0.93),
                )

            def update(self, index: int) -> None:
                self.ax.set_title(f"Energy in Hartree\ngamma:\n{self.gammas[index]:.4g}")
                self.im.set_array(self.data[:, :, index])

        fig, ax = plt.subplots()
        ax.set_xlabel("$\\alpha$")
        ax.set_ylabel("$\\beta$")
        _anim = AnimationWithButtons(
            fig=fig, ax=ax, A=A_small, B=B_small, gammas=gammas, data=energies
        )

        # _anim.anim.save(
        #     self.output_path.joinpath("energy_landscape.mp4"), writer=animation.FFMpegWriter(fps=1)
        # )
        image_path = self.output_path.joinpath("energy_landscape")
        image_path.mkdir(exist_ok=True)
        _anim.anim.save(image_path.joinpath("energy_landscape.png"), writer="imagemagick")
        plt.show()
        plt.close()


def plot_parameter_count(
    output_path: Path,
    problem: ElectronicStructureProblem,
    mapper: FermionicMapper,
    molecule_names: Iterable[str],
    log_y: bool = False,
    add_title: bool = True,
) -> None:
    """Plots the number of parameters for each ansatz and molecule.

    Compares the required number of parameters for the different ansätze.
    It is shown for all given molecules in a single plot.

    Args:
        output_path: the folder for saving the figure.
        problem: the formulation of the electronic structure ploblem.
        mapper: the mapper from Fermionic operator to spin operator.
        molecule_names: the names of the molecules for the comparison.
        log_y: whether to use a logarithmic scale for the y axis.
        add_title: whether to add a title to the plot.
    """
    molecule_names = list(molecule_names)

    num_parameters_vha = []
    num_parameters_uccsd = []
    num_parameters_uccsdt = []
    num_parameters_hea = []
    for molecule_name in molecule_names:
        vha_plots = VHAPlots(
            output_path=output_path.joinpath(f"plots_{molecule_name.replace('_', '')}"),
            molecule_name=molecule_name,
            problem=problem,
            mapper=mapper,
        )
        num_parameters_vha.append(vha_plots.get_VHA_circuit_counts()["num_parameters"][0])
        num_parameters_uccsd.append(
            vha_plots.get_reference_UCC_circuit_counts(excitations=(1, 2))["num_parameters"]
        )
        num_parameters_uccsdt.append(
            vha_plots.get_reference_UCC_circuit_counts(excitations=(1, 2, 3))["num_parameters"]
        )
        num_parameters_hea.append(vha_plots.get_reference_HEA_circuit_counts()["num_parameters"])

    ax = plt.figure().gca()

    x_offset = 0.24
    width = 0.21

    # VHA
    plt.bar(
        x=np.arange(len(molecule_names)) - x_offset,
        height=[5 * num for num in num_parameters_vha],
        width=width,
        edgecolor=colors_tvha[0],
        linestyle="dotted",
        fill=False,
        label="VHA (5 Trotter steps)",
    )
    plt.bar(
        x=np.arange(len(molecule_names)) - x_offset,
        height=[2 * num for num in num_parameters_vha],
        width=width,
        edgecolor=colors_tvha[0],
        fill=False,
        label="VHA (2 Trotter steps)",
    )
    plt.bar(
        x=np.arange(len(molecule_names)) - x_offset,
        height=num_parameters_vha,
        width=width,
        color=colors_tvha[0],
        edgecolor=colors_tvha[0],
        label="VHA (1 Trotter step)",
    )

    # UCC
    plt.bar(
        x=np.arange(len(molecule_names)),
        height=num_parameters_uccsdt,
        width=width,
        edgecolor=colors_ucc[0],
        fill=False,
        label="UCCSDT",
    )
    plt.bar(
        x=np.arange(len(molecule_names)),
        height=num_parameters_uccsd,
        width=width,
        tick_label=[
            "$H_2 / CH_2$" if name == "H_2" else "$" + name + "$" for name in molecule_names
        ],
        color=colors_ucc[0],
        edgecolor=colors_ucc[0],
        label="UCCSD",
    )

    # HEA
    plt.bar(
        x=np.arange(len(molecule_names)) + x_offset,
        height=[5 * num / 3 for num in num_parameters_hea],
        width=width,
        edgecolor=color_hea,
        fill=False,
        label="HEA (5 layers)",
    )
    plt.bar(
        x=np.arange(len(molecule_names)) + x_offset,
        height=num_parameters_hea,
        width=width,
        color=color_hea,
        label="HEA (3 layers)",
    )

    plt.yscale("log" if log_y else "linear")
    if not log_y:
        _, ymax = plt.ylim()
        plt.ylim(0, ymax)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if add_title:
        plt.title(
            "Parameter count for tVHA, UCC and HEA "
            f"({' '.join(['$' + name + '$' for name in molecule_names])})"
        )
    plt.ylabel("Number of parameters")

    handles, labels = plt.gca().get_legend_handles_labels()
    new_order = [2, 1, 0, 4, 3, 6, 5]
    plt.legend([handles[idx] for idx in new_order], [labels[idx] for idx in new_order])

    filename = "parameter_count"
    if log_y:
        filename += "_logy"
    filename += ".svg"
    plt.savefig(output_path.joinpath(filename), format="svg")
    plt.close()


def find_somos(local_occ: list[list[int]]) -> list[int]:
    """Identify SOMO (singly occupied molecular orbitals) indices.

    Args:
        local_occ:           UHF Orbital occupation.

    Returns:
        idx:                 Identified SOMOs indices.

    """
    if not len(local_occ[0]) == len(local_occ[1]):
        raise ValueError("This function works with UHF or UKS formalism.")
    occ_diff = np.array(local_occ[0]) - np.array(local_occ[1])
    somos = np.where(occ_diff == 1)[0]
    return somos.tolist()


def conv_mf(
    local_mol: gto.Mole, dm0: np.ndarray | None = None, stability_loop: bool = False
) -> scf.hf.SCF:
    """Create an scf instance and converge it.

    Args:
        local_mol:          PySCF Mole object.
        dm0:                Initial guess for the density matrix.
        stability_loop:     Whether to use the stability loop.

    Returns:
        mean field object.
    """
    # Initial SCF guess.
    initial_guess = "minao"

    mf = scf.UHF(local_mol)
    mf.init_guess = initial_guess
    mf.max_cycle = 250
    mf = mf.newton()
    mf.kernel(dm0=dm0)
    if stability_loop:
        mo_new = mf.stability()[0]
        while not np.all(np.isclose(mo_new, mf.mo_coeff)) or not mf.converged:
            mf.kernel(dm0=mf.make_rdm1(mo=mo_new))
            mo_new = mf.stability()[0]
    if not mf.converged:
        raise ValueError("SCF calculations did not converge.")
    return mf


def get_minimal_active_space_size(lowest_spin: int) -> tuple[int, int]:
    """Gets the size of the minimal active space.

    Minimal active space is defined as Number of singly occupied in the high spin configurations.
    Meaning (3 number of electrons, 3 number of orbitals) for the CH and NO3, and (2, 2) for CH_2.

    Returns: tuple(active_space_size, active_space_electrons)
    """
    if lowest_spin == 1:
        return (3, 3)
    if lowest_spin == 0:
        return (2, 2)
    raise ValueError("The system spin must be either 0 or 1.")


def get_electronic_structure_problem(
    molecule_name: str,
    basis_set: str = "def2-svp",
    path_data_files: Path = Path(__file__).parent,
) -> ElectronicStructureProblem:
    """Initializes ElectronicStructureProblem from molecule name."""
    # molecules without active space
    if molecule_name.startswith("H_"):
        try:
            number_of_atoms = int(molecule_name.split("_")[-1])
        except Exception as err:
            raise ValueError(f"Unsupported molecule {molecule_name}") from err
        molecule = MoleculeInfo(
            symbols=["H"] * number_of_atoms,
            coords=[(0.0, 0.0, 0.74279 * i) for i in range(number_of_atoms)],
            multiplicity=1,
            charge=0,
        )
        driver = PySCFDriver.from_molecule(molecule)
        driver.basis = basis_set
        return driver.run()
    if molecule_name == "LiH":
        molecule = MoleculeInfo(
            symbols=["Li", "H"],
            coords=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.596)],
            multiplicity=1,
            charge=0,
        )
        driver = PySCFDriver.from_molecule(molecule)
        driver.basis = basis_set
        return driver.run()
    if molecule_name == "H_2O":
        molecule = MoleculeInfo(
            symbols=["O", "H", "H"],
            coords=[(0.0, 0.0, 0.0), (0.758602, 0.0, 0.504284), (0.758602, 0.0, -0.504284)],
            multiplicity=1,
            charge=0,
        )
        driver = PySCFDriver.from_molecule(molecule)
        driver.basis = basis_set
        return driver.run()

    # molecules with active space
    if molecule_name == "CH":
        input_structure = "ch_opt.xyz"
        lowest_spin = 1
        system_charge = 0
    elif molecule_name == "CH_2":
        input_structure = "ch2_opt.xyz"
        lowest_spin = 0
        system_charge = 0
    elif molecule_name == "NO_3":
        input_structure = "no3_opt.xyz"
        lowest_spin = 1
        system_charge = 0
    else:
        raise ValueError(f"Unsupported molecule {molecule_name}")
    _, active_space_electrons = get_minimal_active_space_size(lowest_spin=lowest_spin)

    mol = gto.Mole()
    mol.atom = str(path_data_files.joinpath(input_structure))
    mol.basis = basis_set
    mol.verbose = 4

    # High spin is usually easier to converge for the mean-field theory.
    mol.spin = lowest_spin + 2
    mol.charge = system_charge
    mol.build()

    # Calculate UHF for the high spin state.
    # High spin
    hs_mf = conv_mf(mol, stability_loop=False)

    # Find the indices of singly occupied molecular orbitals (from the high spin).
    somos_idx = find_somos(hs_mf.mo_occ)

    driver = PySCFDriver(
        atom=[f"{i[0]} {i[1][0]} {i[1][1]} {i[1][2]}" for i in mol._atom],  # noqa: SLF001
        charge=system_charge,
        spin=lowest_spin,
        basis=basis_set,
    )
    full_problem = driver.run()
    active_space_transformer = ActiveSpaceTransformer(
        num_electrons=(
            int((active_space_electrons + lowest_spin) / 2),
            int((active_space_electrons - lowest_spin) / 2),
        ),
        num_spatial_orbitals=len(somos_idx),
        active_orbitals=somos_idx,
    )
    return active_space_transformer.transform(full_problem)


def main() -> None:
    """Main function to create the plots."""
    # ---------------------------------------------------------------------------------------------
    # --- SETTINGS --------------------------------------------------------------------------------
    molecule_name = "H_2"
    # molecule_name = "H_4"
    # molecule_name = "LiH"
    # molecule_name = "CH_2"
    # molecule_name = "NO_3" # more complex due to non-diagonal fock operator
    # molecule_name = "CH" # HF is already within chemical accuracy

    if molecule_name in ("H_2", "H_4", "LiH"):  # noqa: SIM108
        basis_set = "sto3g"  # until now used for H_2, H_4, LiH; all others use def2-svp by default
    else:
        basis_set = "def2-svp"

    list_of_trotter_steps = (1, 2, 5)

    mapper = JordanWignerMapper()

    thresholds_gamma = None  # for all possible thresholds
    # info: number of datapoints (i.e. non-Coulomb two-body terms + 1):
    # 5 for H_2, 141 for H_4, 529 for LiH, 13 for CH_2, 43 for NO_3, 13 for CH
    plot_histograms = True
    plot_density_distributions = True
    plot_cnot_count_over_truncation_threshold = True
    plot_circuit_depth_over_truncation_threshold = False
    plot_energy_over_truncation_threshold = False
    plot_energy_over_trotter_steps = False
    plot_energy_over_truncation_threshold_and_trotter_steps = True

    plot_parameter_count_over_ansatz = False

    # Enable carefully (highly inefficient since it does not save costly intermediate results):
    plot_energy_landscape = False

    # Option to add a title to the plots:
    add_title = True
    # ---------------------------------------------------------------------------------------------

    output_folder = (
        Path(__file__)
        .parent.joinpath("data_for_paper_tvha")
        .joinpath(f"plots_{molecule_name.replace('_', '')}")
    )
    output_folder.mkdir(exist_ok=True)

    problem = get_electronic_structure_problem(
        molecule_name=molecule_name, basis_set=basis_set, path_data_files=output_folder.parent
    )

    vha = VariationalHamiltonianAnsatz(problem=problem, trotter_steps=1, mapper=mapper)

    if thresholds_gamma is None:
        thresholds_gamma = vha.possible_thresholds_gamma
    else:
        if thresholds_gamma != vha.possible_thresholds_gamma:
            logger.info(
                "γ thresholds were adjusted to the molecule. These are the new thresholds %s",  # noqa: RUF001
                vha.possible_thresholds_gamma,
            )
            thresholds_gamma = vha.possible_thresholds_gamma

    vha_plots = VHAPlots(
        output_path=output_folder, molecule_name=molecule_name, problem=problem, mapper=mapper
    )

    if plot_histograms:
        vha_plots.plot_histogram(
            hamiltonian=vha.hamilton_operator, log_x=True, log_y=False, add_title=add_title
        )
        vha_plots.plot_histogram(
            hamiltonian=vha.hamilton_operator, log_x=False, log_y=False, add_title=add_title
        )
        vha_plots.plot_histogram(
            hamiltonian=vha.hamilton_operator, log_x=True, log_y=True, add_title=add_title
        )
        vha_plots.plot_histogram(
            hamiltonian=vha.hamilton_operator, log_x=False, log_y=True, add_title=add_title
        )
    if plot_density_distributions:
        vha_plots.plot_cumulated_density_distribution_all_terms(
            hamiltonian=vha.hamilton_operator, add_title=add_title
        )
        vha_plots.plot_cumulated_density_distribution_noncoulomb_terms(
            hamiltonian=vha.hamilton_operator, add_title=add_title
        )

    # All energies are given as electronic energies without the nuclear repulsion energy
    # unless stated explicitly to be the total energy.

    if logger.getEffectiveLevel() <= logging.DEBUG:
        datapoint = vha_plots.get_energies(
            list_of_trotter_steps=1, thresholds_gamma=1.0, return_full_datapoint=True
        )
        energy_statevector = datapoint[vha_plots.energy_data_header.index("energy")]
        optimal_parameters = datapoint[vha_plots.energy_data_header.index("optimal_parameters")]
        if len(optimal_parameters) == 3:
            optimal_parameters_string = (
                f"α={optimal_parameters[0]:.5g}, "  # noqa: RUF001
                f"β={optimal_parameters[1]:.5g}, "
                f"γ={optimal_parameters[2]:.5g}"  # noqa: RUF001
            )
        else:
            optimal_parameters_string = optimal_parameters
        result_string = (
            "Total ground state energy in Hartree Fock approximation "
            f"{problem.reference_energy:.3f};\n"
            "FCI energy (numerical diagonalization; only active space for active space calculations): "
            f"{vha_plots.numerical_energies['computed_energy'] + vha_plots.numerical_energies['nuclear_repulsion_energy']:.3f}; \n"
            "Total ground state energy (VHA statevector simulator): "
            f"{energy_statevector + vha_plots.numerical_energies['nuclear_repulsion_energy']:.3f}; \n"
            "Improvement over HF approx: "
            f"{problem.reference_energy - (energy_statevector + vha_plots.numerical_energies['nuclear_repulsion_energy']):.3g}; \n"
            "Difference to FCI energy: "
            f"{vha_plots.numerical_energies['computed_energy'] - energy_statevector:.3g} \n"
            "Optimal parameters: " + optimal_parameters_string
        )
        print(result_string)

    if plot_cnot_count_over_truncation_threshold:
        print("Plotting CNOT count over truncation threshold...")
        vha_plots.plot_cnot_count_over_truncation_threshold(log_y=True, add_title=add_title)
        vha_plots.plot_cnot_count_over_truncation_threshold(log_y=False, add_title=add_title)
        print("Done.")

    if plot_circuit_depth_over_truncation_threshold:
        print("Plotting circuit depth over truncation threshold...")
        vha_plots.plot_circuit_depth_over_truncation_threshold(
            log_y=True, add_cnot_count=False, add_title=add_title
        )
        vha_plots.plot_circuit_depth_over_truncation_threshold(
            log_y=False, add_cnot_count=False, add_title=add_title
        )
        print("Done.")

    if plot_energy_landscape:
        print("Plotting energy landscape...")
        vha_plots.plot_energy_landscape(
            alphas=np.linspace(-1 * 3, 2 * 3, 50),  # periodicity for H2: roughly 3
            betas=np.linspace(-1 * 9, 2 * 9, 50),  # periodicity for H2: roughly 9
            gammas=np.linspace(-1 * 21, 2 * 21, 50),  # periodicity for H2: roughly 21
        )
        print("Done.")

    if plot_energy_over_truncation_threshold:
        print("Plotting energy over truncation threshold...")
        vha_plots.plot_energy_over_truncation_threshold(
            trotter_steps=1, thresholds_gamma=thresholds_gamma, add_title=add_title
        )
        print("Done.")

    if plot_energy_over_trotter_steps:
        print("Plotting energy over Trotter steps...")
        vha_plots.plot_energy_over_trotter_steps(
            list_of_trotter_steps=list_of_trotter_steps,
            threshold_gamma=np.linspace(1, 0, 4, endpoint=False),
            add_title=add_title,
        )
        print("Done.")

    if plot_energy_over_truncation_threshold_and_trotter_steps:
        print("Plotting energy over truncation threshold and Trotter steps...")
        vha_plots.plot_energy_over_truncation_threshold_and_trotter_steps(
            list_of_trotter_steps=list_of_trotter_steps,
            thresholds_gamma=thresholds_gamma,
            add_title=add_title,
        )
        vha_plots.plot_energy_over_truncation_threshold(
            trotter_steps=list_of_trotter_steps,
            thresholds_gamma=thresholds_gamma,
            add_title=add_title,
        )
        print("Done.")

    if plot_parameter_count_over_ansatz:
        print("Plotting number of parameters over ansatz...")
        plot_parameter_count(
            output_path=output_folder.parent,
            problem=problem,
            mapper=mapper,
            molecule_names=("H_2", "H_4", "LiH"),
            log_y=False,
            add_title=add_title,
        )
        plot_parameter_count(
            output_path=output_folder.parent,
            problem=problem,
            mapper=mapper,
            molecule_names=("H_2", "H_4", "LiH"),
            log_y=True,
            add_title=add_title,
        )
        print("Done.")


if __name__ == "__main__":
    main()
