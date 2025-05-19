"""Test suite for running tVHA."""

import numpy as np
import pytest
from qiskit.primitives import Estimator as Statevector_Estimator
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_machine_learning.optimizers import L_BFGS_B, NFT, SBPLX, Optimizer
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import JordanWignerMapper

from tvha.tvha import VariationalHamiltonianAnsatz

# ruff: noqa: S101

optimizer_l_bfgs = L_BFGS_B(maxiter=1000)
optimizer_nft = NFT(maxiter=500, maxfev=1000, reset_interval=100)
optimizer_sbplx = SBPLX(max_evals=1000)


class TestVariationalHamiltonianAnsatz:
    """Collection of tests for the Variational Hamiltonian Ansatz."""

    @pytest.mark.parametrize("optimizer", [optimizer_l_bfgs, optimizer_nft, optimizer_sbplx])
    def test_vha_minimal_example(self, optimizer: Optimizer) -> None:
        """Minimal example for calculations with tVHA algorithm for H2 molecule.

        Args:
            optimizer: The classical optimization algorithm to use for this test case.
        """
        estimator = Statevector_Estimator()
        mapper = JordanWignerMapper()
        basis_set = "sto-3g"
        number_of_atoms = 2
        molecule = MoleculeInfo(
            symbols=["H"] * number_of_atoms,
            coords=[(0.0, 0.0, 0.74279 * i) for i in range(number_of_atoms)],
            multiplicity=1,
            charge=0,
        )
        driver = PySCFDriver.from_molecule(molecule)
        driver.basis = basis_set
        problem = driver.run()

        trotter_steps = 1
        threshold_gamma = 0.5

        ansatz = VariationalHamiltonianAnsatz(
            problem=problem,
            trotter_steps=trotter_steps,
            threshold_gamma=threshold_gamma,
            mapper=mapper,
        )

        vqe = VQE(
            estimator=estimator,
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=ansatz.get_initial_point(),
        )
        result_vha_statevector = vqe.compute_minimum_eigenvalue(
            operator=mapper.map(second_q_ops=ansatz.hamilton_operator)
        )

        assert np.allclose(float(result_vha_statevector.eigenvalue), -1.8496717733201034)
