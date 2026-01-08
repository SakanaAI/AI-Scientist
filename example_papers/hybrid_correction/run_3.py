import numpy as np
import pandas as pd
from cpymad.madx import Madx, TwissFailed
import random
import os
import joblib
from sklearn.neural_network import MLPRegressor

class VEPP5_sample:
    def __init__(self):
        self.init_globals()
        self.num_bpms = 16
        self.hidden_parameters = {'sigma_x':1500e-6, 'sigma_y':1500e-6, 'sigma_s':1500e-6, 'sigma_psi':150e-6, 'seed':1}

        # Precompute the nominal response matrix (without misalignments)
        orig_hidden = self.hidden_parameters.copy()
        self.hidden_parameters = {'sigma_x':0, 'sigma_y':0, 'sigma_s':0, 'sigma_psi':0, 'seed':1}
        self.nominal_response_matrix = self.calculcate_resp_mat()
        self.hidden_parameters = orig_hidden

    @property
    def current_elements(self):
        return self.globals

    def init_globals(self):
        self.globals = {}
        madx = Madx(stdout=False)
        madx.input("option, echo=false, warn=false, info=false, twiss_print=false;")
        madx.call('vepp5_full.seq')
        for name, val in madx.globals.items():
            if 'i_c' in name:
                self.globals[name] = val
        madx.quit()
        del madx

    def start_madx(self):
        madx = Madx(stdout=False)
        madx.input("option, echo=false, warn=false, info=false, twiss_print=false;")
        madx.call('vepp5_full.seq')
        madx.beam(sequence='allrng_ele', particle='electron', energy='0.43', radiate=False)
        madx.use(sequence='allrng_ele')
        madx.select('FLAG = Twiss', 'class = monitor', 'column = x, y;')
        madx.input(f'''
            eoption,seed={self.hidden_parameters['seed']};
            select, flag=error, clear;
            select, flag=error, class=quadrupole;
            ealign, dx:=tgauss(2.5)*{self.hidden_parameters['sigma_x']}, 
                   dy:=tgauss(2.5)*{self.hidden_parameters['sigma_y']}, 
                   ds:=tgauss(2.5)*{self.hidden_parameters['sigma_s']}, 
                   dpsi:=tgauss(2.5)*{self.hidden_parameters['sigma_psi']};
        ''')
        return madx

    def stop_madx(self, madx):
        madx.quit()
        del madx

    def _get_orbit(self, madx):
        for name, val in self.globals.items():
            madx.globals[name] = val

        try:
            madx.twiss(table='twiss', centre=True)
            x = madx.table.twiss.selection().x
            y = madx.table.twiss.selection().y
        except TwissFailed:
            x = np.full(self.num_bpms, np.inf)
            y = np.full(self.num_bpms, np.inf)
            print("Cannot calculate orbits. Try to decrease element values.")

        return x, y

    def get_orbit(self):
        madx = self.start_madx()
        x, y = self._get_orbit(madx)
        self.stop_madx(madx)
        return x, y

    def change_elements(self, elements):
        for name, val in elements.items():
            self.globals[name] = val

    def calculcate_resp_mat(self):
        step = 1
        responses = {}

        madx = self.start_madx()
        for elem, current_val in self.globals.copy().items():
            # Plus step
            self.change_elements({elem: current_val + step})
            x, y = self._get_orbit(madx)

            # Minus step
            self.change_elements({elem: current_val - step})
            x_tmp, y_tmp = self._get_orbit(madx)

            x -= x_tmp
            y -= y_tmp
            x /= step
            y /= step

            # Reset to initial value
            self.change_elements({elem: current_val})

            orbit = np.concatenate((x, y))
            responses[elem] = orbit

        matrix = pd.DataFrame(responses)
        self.stop_madx(madx)
        return matrix

    def correct_orbit(self):
        elem_val_limit = 7
        svd_cutoff = 1e-3
        target_orbit = np.zeros(2 * self.num_bpms)

        # Use precomputed nominal response matrix
        matrix = self.nominal_response_matrix
        inv_mat = np.linalg.pinv(matrix, rcond=svd_cutoff)

        madx = self.start_madx()
        x, y = self._get_orbit(madx)
        current_orbit = np.concatenate((x, y))

        tmp_elem_val = -inv_mat.dot(current_orbit - target_orbit)
        tmp_elem_val = np.clip(tmp_elem_val, -elem_val_limit, elem_val_limit)

        elems_deltas = dict(zip(self.globals.keys(), tmp_elem_val))
        self.change_elements(elems_deltas)
        x, y = self._get_orbit(madx)

        self.stop_madx(madx)
        return x, y, elems_deltas

    def hybrid_correct_orbit(self, model=None):
        # First, do the SVD correction
        x_svd, y_svd, elems_deltas = self.correct_orbit()
        R = np.concatenate((x_svd, y_svd))

        # If no model, then we just return the SVD correction
        if model is None:
            return x_svd, y_svd, elems_deltas, None

        # Predict the additional correction
        dI_nn = model.predict(R.reshape(1, -1))[0]

        # Apply the additional correction to the correctors
        for i, key in enumerate(self.globals.keys()):
            self.globals[key] += dI_nn[i]

        # Get the new orbit
        madx = self.start_madx()
        x_new, y_new = self._get_orbit(madx)
        self.stop_madx(madx)

        return x_new, y_new, elems_deltas, dI_nn
        

import argparse
import json
import numpy as np
import os

def run_experiment(out_dir, num_samples=10):
    os.makedirs(out_dir, exist_ok=True)
    smp = VEPP5_sample()
    init_info = []
    corr_info = []
    results_dict = {}

    # For run_1: training data collection and model training
    if out_dir == "run_1":
        # Collect training data
        X = []
        Y = []
        num_samples_train = 100000

        for i in range(num_samples_train):
            smp.hidden_parameters = {
                'sigma_x': np.random.uniform(1e-4, 1e-3),
                'sigma_y': np.random.uniform(1e-4, 1e-3),
                'sigma_s': np.random.uniform(1e-4, 1e-3),
                'sigma_psi': np.random.uniform(1e-4, 1e-3),
                'seed': np.random.randint(1, 10)
            }

            for key in smp.current_elements:
                smp.change_elements({key: 0.0})

            # Get initial orbit
            initial_x, initial_y = smp.get_orbit()

            # Do SVD correction
            corrected_x, corrected_y, corrected_elements = smp.correct_orbit()

            # Compute residual orbit
            R = np.concatenate((corrected_x, corrected_y))
            # Compute ideal additional correction
            inv_mat = np.linalg.pinv(smp.nominal_response_matrix, rcond=1e-3)
            dI_ideal = -inv_mat.dot(R)

            X.append(R)
            Y.append(dI_ideal)

        # Train the model
        model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=100, verbose=True)
        model.fit(X, Y)

        # Save the model
        model_file = os.path.join(out_dir, "nn_model.joblib")
        joblib.dump(model, model_file)

        # For bookkeeping, we run one sample and save minimal results
        for key in smp.current_elements:
            smp.change_elements({key: 0.0})
        initial_x, initial_y = smp.get_orbit()
        corrected_x, corrected_y, corrected_elements = smp.correct_orbit()

        results_dict["experiment_1"] = {
            'hidden_parameters': smp.hidden_parameters,
            'initial_orbit': {'x': initial_x.tolist(), 'y': initial_y.tolist()},
            'corrected_orbit': {'x': corrected_x.tolist(), 'y': corrected_y.tolist()},
            'corrected_elements': corrected_elements,
        }

        init_info.append({
            "iter": 1,
            "loss": np.sum(initial_x**2 + initial_y**2),
            "phase": "init"
        })
        corr_info.append({
            "iter": 1,
            "loss": np.sum(corrected_x**2 + corrected_y**2),
            "phase": "correction"
        })

    # For runs 2 to 10: hybrid correction
    elif out_dir in [f"run_{i}" for i in range(2,11)]:
        # Load the pre-trained model
        model_file = "run_1/nn_model.joblib"
        if not os.path.exists(model_file):
            print(f"Model file {model_file} not found. Falling back to SVD.")
            model = None
        else:
            model = joblib.load(model_file)

        for i in range(num_samples):
            smp.hidden_parameters = {
                'sigma_x': np.random.uniform(1e-4, 1e-3),
                'sigma_y': np.random.uniform(1e-4, 1e-3),
                'sigma_s': np.random.uniform(1e-4, 1e-3),
                'sigma_psi': np.random.uniform(1e-4, 1e-3),
                'seed': np.random.randint(1, 10)
            }

            for key in smp.current_elements:
                smp.change_elements({key: 0.0})

            initial_x, initial_y = smp.get_orbit()
            # Use hybrid correction
            corrected_x, corrected_y, corrected_elements, dI_nn = smp.hybrid_correct_orbit(model)

            results_dict[f"experiment_{i + 1}"] = {
                'hidden_parameters': smp.hidden_parameters,
                'initial_orbit': {'x': initial_x.tolist(), 'y': initial_y.tolist()},
                'corrected_orbit': {'x': corrected_x.tolist(), 'y': corrected_y.tolist()},
                'corrected_elements': corrected_elements,
            }

            init_info.append({
                "iter": i + 1,
                "loss": np.sum(initial_x**2 + initial_y**2),
                "phase": "init"
            })
            corr_info.append({
                "iter": i + 1,
                "loss": np.sum(corrected_x**2 + corrected_y**2),
                "phase": "correction"
            })

    # For other runs (including run_0): baseline SVD
    else:
        for i in range(num_samples):
            smp.hidden_parameters = {
                'sigma_x': np.random.uniform(1e-4, 1e-3),
                'sigma_y': np.random.uniform(1e-4, 1e-3),
                'sigma_s': np.random.uniform(1e-4, 1e-3),
                'sigma_psi': np.random.uniform(1e-4, 1e-3),
                'seed': np.random.randint(1, 10)
            }

            for key in smp.current_elements:
                smp.change_elements({key: 0.0})

            initial_x, initial_y = smp.get_orbit()
            corrected_x, corrected_y, corrected_elements = smp.correct_orbit()

            results_dict[f"experiment_{i + 1}"] = {
                'hidden_parameters': smp.hidden_parameters,
                'initial_orbit': {'x': initial_x.tolist(), 'y': initial_y.tolist()},
                'corrected_orbit': {'x': corrected_x.tolist(), 'y': corrected_y.tolist()},
                'corrected_elements': corrected_elements,
            }

            init_info.append({
                "iter": i + 1,
                "loss": np.sum(initial_x**2 + initial_y**2),
                "phase": "init"
            })
            corr_info.append({
                "iter": i + 1,
                "loss": np.sum(corrected_x**2 + corrected_y**2),
                "phase": "correction"
            })

    formatted_results = {
        "orbit_correction": {
            "means": {
                "init_loss_mean": np.mean([info["loss"] for info in init_info]),
                "corr_loss_mean": np.mean([info["loss"] for info in corr_info])
            },
            "stderrs": {
                "init_loss_stderr": np.std([info["loss"] for info in init_info]) / np.sqrt(len(init_info)),
                "corr_loss_stderr": np.std([info["loss"] for info in corr_info]) / np.sqrt(len(corr_info))
            },
            "final_info_dict": results_dict
        }
    }

    all_results = {
        "orbit_correction_final_info": formatted_results,
        "orbit_correction_init_info": init_info,
        "orbit_correction_corr_info": corr_info,
    }

    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump(formatted_results, f, indent=4)

    with open(os.path.join(out_dir, "all_results.npy"), "wb") as f:
        np.save(f, all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run orbit correction experiment")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of experiments")
    args = parser.parse_args()

    run_experiment(args.out_dir, args.num_samples)
