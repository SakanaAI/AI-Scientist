import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def get_run_dirs(base_dir='.'):
    return sorted([
        d for d in os.listdir(base_dir)
        if d.startswith('run') and os.path.isdir(os.path.join(base_dir, d))
    ])


def plot_training_curves(run_dirs):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_dirs)))

    for idx, run in enumerate(run_dirs):
        pkl_path = os.path.join(run, 'all_results.pkl')
        if not os.path.exists(pkl_path):
            print(f"[{run}] missing all_results.pkl")
            continue

        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)

        train_losses = results.get('eegnet_train_losses', [])
        val_accuracies = np.array(results.get('eegnet_val_accuracies', [])) * 100

        ax1.plot(train_losses, label=f'{run} - train loss', color=colors[idx], linestyle='-')
        ax2.plot(val_accuracies, label=f'{run} - val acc', color=colors[idx], linestyle='--')

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color='blue')
    ax2.set_ylabel("Validation Accuracy (%)", color='green')

    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='green')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")

    plt.title("Train Loss & Validation Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_training_curves.png")
    plt.close()


def analyze_test_results(run_dirs):
    test_accs = []

    for run in run_dirs:
        final_info_path = os.path.join(run, 'final_info.json')
        all_results_path = os.path.join(run, 'all_results.pkl')

        if not (os.path.exists(final_info_path) and os.path.exists(all_results_path)):
            print(f"[{run}] missing result files")
            continue

        with open(final_info_path, 'r') as f:
            final_info = json.load(f)
        test_acc = final_info['eegnet']['means']['test_accuracy']

        with open(all_results_path, 'rb') as f:
            results = pickle.load(f)
        test_results = results.get('eegnet_test_results', [])

        y_true = [x['label'] for x in test_results]
        y_pred = [x['pred'] for x in test_results]

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f"Confusion Matrix – {run}\nTest Accuracy: {test_acc * 100:.2f}%")
        plt.tight_layout()
        plt.savefig(f"plot_confusion_matrix_{run}.png")
        plt.close()

        print(f"[{run}] Test accuracy: {test_acc * 100:.2f}%")
        test_accs.append((run, test_acc))

    if test_accs:
        test_accs.sort(key=lambda x: x[0])
        run_names, accs = zip(*test_accs)

        plt.figure(figsize=(8, 5))
        plt.bar(run_names, [a * 100 for a in accs], color='teal')
        plt.title("Test Accuracy by Run")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("plot_test_accuracy_by_run.png")
        plt.close()


if __name__ == "__main__":
    run_dirs = get_run_dirs()
    print(f"Found {len(run_dirs)} runs: {run_dirs}")

    plot_training_curves(run_dirs)
    analyze_test_results(run_dirs)
