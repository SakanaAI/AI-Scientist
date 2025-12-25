import os
import json
import argparse
import numpy as np
from datetime import datetime

class SimpleVQE:
    """
    A simple Variational Quantum Eigensolver (VQE) problem.
    The goal is to find the ground state energy of the Pauli-Z Hamiltonian.
    H = Z = [[1, 0], [0, -1]]
    The ansatz is a Ry(theta) rotation gate on a |0> state.
    |psi(theta)> = Ry(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>
    The energy expectation value is E(theta) = <psi(theta)|Z|psi(theta)> = cos(theta).
    The analytical minimum is -1 at theta = pi.
    """
    def __init__(self):
        self.dim = 1 # single parameter theta

    def __call__(self, theta: np.ndarray) -> float:
        """ Calculates the energy E(theta) = cos(theta). """
        return np.cos(theta[0])

    def grad(self, theta: np.ndarray) -> np.ndarray:
        """ Calculates the gradient of the energy dE/dtheta = -sin(theta). """
        return np.array([-np.sin(theta[0])])

    def quantum_fisher_information(self, theta: np.ndarray) -> np.ndarray:
        """
        Calculates the Quantum Fisher Information (QFI) matrix, g.
        For a single qubit rotation Ry(theta), the QFI is a constant scalar matrix [1].
        This metric tensor is used by the Quantum Natural Gradient optimizer.
        """
        # For Ry(theta), the QFI is exactly 1.
        # We return a 1x1 matrix.
        return np.array([[1.0]])

def gradient_descent(func, grad, theta0, lr=0.1, max_iters=100):
    """ Standard Gradient Descent optimizer. """
    theta = theta0.copy()
    history = []
    for _ in range(max_iters):
        g = grad(theta)
        theta -= lr * g
        history.append(func(theta))
    return theta, history

def quantum_natural_gradient(func, grad, qfi_fn, theta0, lr=0.1, max_iters=100):
    """
    Quantum Natural Gradient (QNG) optimizer.
    The update rule is: theta_new = theta_old - lr * g_inv * grad
    where g_inv is the inverse of the Quantum Fisher Information matrix.
    """
    theta = theta0.copy()
    history = []
    for _ in range(max_iters):
        g = grad(theta)
        qfi_matrix = qfi_fn(theta)
        # Add a small epsilon for numerical stability before inverting
        qfi_inv = np.linalg.pinv(qfi_matrix + 1e-8 * np.eye(qfi_matrix.shape[0]))
        
        update = qfi_inv @ g
        theta -= lr * update
        history.append(func(theta))
    return theta, history

def run_experiment(out_dir):
    """
    Runs the VQE optimization experiment with GD and QNG optimizers.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # --- Experiment Setup ---
    vqe_problem = SimpleVQE()
    # Initial parameter, start away from the minimum (pi)
    theta0 = np.array([0.5]) 
    n_iters = 100
    learning_rate = 0.2
    
    # --- Run Baselines ---
    # 1. Standard Gradient Descent
    best_theta_gd, hist_gd = gradient_descent(
        vqe_problem, vqe_problem.grad, theta0, lr=learning_rate, max_iters=n_iters
    )

    # 2. Quantum Natural Gradient
    best_theta_qng, hist_qng = quantum_natural_gradient(
        vqe_problem, vqe_problem.grad, vqe_problem.quantum_fisher_information, 
        theta0, lr=learning_rate, max_iters=n_iters
    )
    
    results = {
        'vqe_ground_state': {
            'gd': {'best_val': float(vqe_problem(best_theta_gd)), 'history': hist_gd},
            'qng': {'best_val': float(vqe_problem(best_theta_qng)), 'history': hist_qng}
        }
    }
    
    # --- Save Results ---
    # Save raw histories for plotting
    np.save(os.path.join(out_dir, "gd_history.npy"), np.array(hist_gd))
    np.save(os.path.join(out_dir, "qng_history.npy"), np.array(hist_qng))
    
    # Create summary.json for a quick overview
    summary = {
        'vqe_ground_state': {
            alg: results['vqe_ground_state'][alg]['best_val'] 
            for alg in results['vqe_ground_state']
        }
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Create final_info.json for the AI scientist framework
    final_info = {
        'vqe_ground_state': {
            "means": {
                "gd": float(vqe_problem(best_theta_gd)),
                "qng": float(vqe_problem(best_theta_qng))
            }
        }
    }
    with open(os.path.join(out_dir, 'final_info.json'), 'w') as f:
        json.dump(final_info, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save experiment results.")
    args = parser.parse_args()
    run_experiment(args.out_dir)