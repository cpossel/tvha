"""Minimal example for calculations with tVHA algorithm for H2 molecule."""

from qiskit.primitives import Estimator as Statevector_Estimator
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_machine_learning.optimizers import SBPLX
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import JordanWignerMapper

from tvha.tvha import VariationalHamiltonianAnsatz

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
