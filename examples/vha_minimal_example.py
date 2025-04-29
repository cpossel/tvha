"""Minimal example for calculations with tVHA algorithm for H2 molecule."""

from qiskit.primitives import Estimator as Statevector_Estimator
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import JordanWignerMapper

from vha.sbplx import SBPLX
from vha.vha import VHA

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

discretization_steps = 1
threshold_gamma = 0.5
threshold_method = "coeff_value"

ansatz = VHA(
    problem=problem,
    discretization_steps=discretization_steps,
    threshold_gamma=threshold_gamma,
    threshold_method=threshold_method,
    mapper=mapper,
)

vqe = VQE(
    estimator=estimator,
    ansatz=ansatz,
    # optimizer=L_BFGS_B(maxiter=1000),
    # optimizer=NFT(maxiter=500, maxfev=1000, reset_interval=100),
    optimizer=SBPLX(max_evals=1000),
    initial_point=ansatz.get_initial_point(),
)
result_vha_statevector = vqe.compute_minimum_eigenvalue(
    operator=mapper.map(second_q_ops=ansatz.hamilton_operator)
)
energy = float(result_vha_statevector.eigenvalue)

print(f"The calculated energy is {energy}")
