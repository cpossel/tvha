"""Minimal example for calculations with tVHA algorithm for H2, LiH, and H4 molecule."""

from qiskit.primitives import Estimator as Statevector_Estimator
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_machine_learning.optimizers import SBPLX
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import JordanWignerMapper

from tvha.tvha import VariationalHamiltonianAnsatz

# SETTINGS =========================================================================================
# Adjust the name of the molecule here:
molecule_name = "H_2"
# ==================================================================================================

molecule_settings = {
    "H_2": {
        "trotter_steps": 1,
        "threshold_gamma": 0.49999,
        "molecule": MoleculeInfo(
            symbols=["H"] * 2,
            coords=[(0.0, 0.0, 0.74279 * i) for i in range(2)],
            multiplicity=1,
            charge=0,
        ),
    },
    "LiH": {  # full molecule without any active space
        "trotter_steps": 2,
        "threshold_gamma": 0.5,
        "molecule": MoleculeInfo(
            symbols=["Li", "H"],
            coords=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.596)],
            multiplicity=1,
            charge=0,
        ),
    },
    "H_4": {
        "trotter_steps": 4,
        "threshold_gamma": 0.5,
        "molecule": MoleculeInfo(
            symbols=["H"] * 4,
            coords=[
                (0.0, 0.0, 0.74279 * i) for i in range(4)
            ],  # other (non-equilibrium) geometries are in principle as valid as this one
            multiplicity=1,
            charge=0,
        ),
    },
}

molecule_settings[molecule_name]  # checking whether the molecule settings exist

estimator = Statevector_Estimator()
mapper = JordanWignerMapper()
basis_set = "sto-3g"
molecule = molecule_settings[molecule_name]["molecule"]
driver = PySCFDriver.from_molecule(molecule)
driver.basis = basis_set
problem = driver.run()

ansatz = VariationalHamiltonianAnsatz(
    problem=problem,
    trotter_steps=molecule_settings[molecule]["trotter_steps"],
    threshold_gamma=molecule_settings[molecule]["threshold_gamma"],
    mapper=mapper,
)

vqe = VQE(
    estimator=estimator,
    ansatz=ansatz,
    optimizer=SBPLX(max_evals=1000),
    initial_point=ansatz.get_initial_point(),
)
result_vha_statevector = vqe.compute_minimum_eigenvalue(
    operator=mapper.map(second_q_ops=ansatz.hamilton_operator)
)
energy = float(result_vha_statevector.eigenvalue)

print(f"The calculated energy is {energy}")
