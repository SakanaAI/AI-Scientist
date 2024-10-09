import matplotlib.pyplot as plt
import numpy as np
import json
import os
import os.path as osp

# LOAD FINAL RESULTS:
folders = os.listdir("./")
final_results = {}
for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        with open(osp.join(folder, "final_info.json"), "r") as f:
            final_results[folder] = json.load(f)

# Plot 1: Solution of SEIR model
plt.figure(figsize=(10, 6))
labels = {
    "run_0": "Baseline",
}

for i, run in enumerate(final_results.keys()):
    plt.figure(figsize=(10, 6))
    solution = np.array(final_results[run]["solution"])
    susceptible = solution[:, 0]
    exposed = solution[:, 1]
    infectious = solution[:, 2]
    recovered = solution[:, 3]
    infected_peak_day = final_results[run]["infected_peak_day"]
    infected_peak = final_results[run]["infected_peak"]
    tolal_infected = final_results[run]["tolal_infected"]
    plt.plot(susceptible, label=f"Susceptible", color="blue")
    plt.plot(exposed, label=f"Exposed", color="orange")
    plt.plot(infectious, label=f"Infectious", color="red")
    plt.plot(recovered, label=f"Recovered", color="green")
    plt.axvline(
        infected_peak_day, color="red", linestyle="--", label=f"Infected peak day"
    )
    plt.text(infected_peak_day, infected_peak, f"Peak: {infected_peak}", color="red")

    plt.title(f"Solution of SEIR model (total infected: {tolal_infected})")
    plt.xlabel("Time")
    plt.ylabel("Number of people")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"seir_solution_{run}.png")
    plt.close()
