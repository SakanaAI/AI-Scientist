import argparse
import json
import os

import numpy as np
from scipy.integrate import odeint

# -----------------------------------------------------------------------------
# SEIR model is a differential equation model that describes the dynamics of infectious diseases such as COVID-19.
# The model divides the population into four compartments: S (susceptible), E (exposed), I (infectious), and R (recovered).
# -----------------------------------------------------------------------------


parser = argparse.ArgumentParser(description="Run experiment")
parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
args = parser.parse_args()

if __name__ == "__main__":
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)


    def seir_eq(v, t, beta, lp, ip):
        """Differential equation of SEIR model
        v: [S, E, I, R] Distribution of people in each state
        t: Time
        beta: Infection rate
        lp: Latent period
        ip: Infectious period
        """
        dS = -beta * v[0] * v[2]
        dE = beta * v[0] * v[2] - (1 / lp) * v[1]
        dI = (1 / lp) * v[1] - (1 / ip) * v[2]
        dR = (1 / ip) * v[2]
        return np.array([dS, dE, dI, dR])


    # Solve SEIR model
    init_state = np.array([3000, 0, 5, 0])
    solution = odeint(
        seir_eq,
        init_state,
        t=np.arange(0, 100, 1),
        args=(0.001, 14, 7),
    )

    means = {
        "infected_peak_day": np.argmax(solution[:, 2]).item(),
        "infected_peak": np.max(solution[:, 2]).item(),
        "total_infected": solution[-1, 3].item(),
    }

    final_info = {
        "SEIR": {
            "means": means,  # means is used in the experiment
            "solution": solution.tolist(),  # solution is used in the visualization
        }
    }
    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump(final_info, f)
