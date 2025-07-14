import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_results(results_path):
    if not os.path.exists(results_path):
        return None
    with open(results_path, "r") as f:
        return json.load(f)

def plot_orbits(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    if "orbit_correction" not in results or "final_info_dict" not in results["orbit_correction"]:
        return

    for experiment_id, result in results["orbit_correction"]["final_info_dict"].items():
        initial_x = np.array(result["initial_orbit"]["x"])
        initial_y = np.array(result["initial_orbit"]["y"])
        corrected_x = np.array(result["corrected_orbit"]["x"])
        corrected_y = np.array(result["corrected_orbit"]["y"])

        plt.figure(figsize=(12, 6))
        plt.plot(initial_x, label="Initial orb_x", marker="o")
        plt.plot(initial_y, label="Initial orb_y", marker="o")
        plt.plot(corrected_x, label="Corrected orb_x", marker="x")
        plt.plot(corrected_y, label="Corrected orb_y", marker="x")
        plt.title(f"Orbit Trajectories for {experiment_id}")
        plt.xlabel("BPMS Index")
        plt.ylabel("Orbit Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"orbit_{experiment_id}.png"))
        plt.close()

def plot_losses(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    if "orbit_correction_train_info" not in results or "orbit_correction_val_info" not in results:
        return

    train_losses = [info["loss"] for info in results["orbit_correction_train_info"]]
    val_losses = [info["loss"] for info in results["orbit_correction_val_info"]]

    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="x")
    plt.title("Losses Across Experiments")
    plt.xlabel("Experiment Index")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "losses.png"))
    plt.close()

if __name__ == "__main__":
    results_path = "./run_0/final_info.json"
    out_dir = "./run_0/plots"

    results = load_results(results_path)
    if results is not None:
        plot_orbits(results, out_dir)
        plot_losses(results, out_dir)
    else:
        print("Не удалось загрузить результаты. Проверьте путь к файлу.")