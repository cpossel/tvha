"""Minimal example for calculations with tVHA algorithm for H2 molecule."""

from qiskit.primitives import Estimator as Statevector_Estimator
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import JordanWignerMapper

from vha.sbplx import SBPLX
from vha.vha import VHA

# SETTINGS =========================================================================================
# Adjust the name of the molecule here:
molecule_name = "H_2"
# ==================================================================================================

molecule_settings = {
    "H_2": {
        "discretization_steps": 1,
        "threshold_gamma": 0.49999,
        "molecule": MoleculeInfo(
            symbols=["H"] * 2,
            coords=[(0.0, 0.0, 0.74279 * i) for i in range(2)],
            multiplicity=1,
            charge=0,
        ),
    },
    "LiH full molecule": {  # TODO: add here also the reduced active space molecule!
        "discretization_steps": 2,
        "threshold_gamma": 0.5,
        "molecule": MoleculeInfo(
            symbols=["Li", "H"],
            coords=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.596)],
            multiplicity=1,
            charge=0,
        ),
    },
    "H_4": {
        "discretization_steps": 4,
        "threshold_gamma": 0.5,
        "molecule": MoleculeInfo(
            symbols=["H"] * 4,
            coords=[
                (0.0, 0.0, 0.74279 * i) for i in range(4)
            ],  # TODO: other geometries are in principle as valid as this one --> need to decide for one
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

threshold_method = "coeff_value"

ansatz = VHA(
    problem=problem,
    discretization_steps=molecule_settings[molecule]["discretization_steps"],
    threshold_gamma=molecule_settings[molecule]["threshold_gamma"],
    threshold_method=threshold_method,
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
