import matplotlib.pyplot as plt
import numpy as np
import json
import os
import os.path as osp
import pandas as pd
import statsmodels.api as sm

folders = os.listdir("./")
final_results = {}
for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        with open(osp.join(folder, "final_info.json"), "r") as f:
            final_results[folder] = json.load(f)['MACE']['means']

datasets = ['ethanol', 'benzene', 'uracil','naphthalene','aspirin','salicylic','malonaldehyde','toluene','paracetamol','azobenzene']

# CREATE LEGEND -- PLEASE FILL IN YOUR RUN NAMES HERE
# Keep the names short, as these will be in the legend.
labels = {
    "run_0": "Baseline",
}

# CREATE PLOTS
# create plot of molecular size vs rmsf, to show how size of the molecule affects the stability of the md trajectory
def plot_size_vs_rmfs(datasets, final_results):
    mols = []
    rs = []
    for run_id in final_results.keys():
        for dat in datasets:
            mols.append(final_results[run_id][dat]['md_results']['mol_size'])
            rs.append(final_results[run_id][dat]['md_results']['rmfs'])
    plt.figure()
    plt.scatter(np.array(mols), np.array(rs))
    plt.xlabel('Average size')
    plt.ylabel('RMFS')
    plt.legend()
    plt.title('Mean size vs RMFS')
    plt.savefig('size_vs_rmfs.png', dpi=300)

# create plot of MACE trained model validation MAE vs rmsf
def plot_mae_vs_rmfs(datasets, final_results):
    maes = []
    rs = []
    for run_id in final_results.keys():
       for dat in datasets:
          maes.append(final_results[run_id][dat]['mace_results']['mae_e'])
          rs.append(final_results[run_id][dat]['md_results']['rmfs'])
    plt.figure()
    plt.scatter(np.array(maes), np.array(rs))
    plt.xlabel('MAE')
    plt.ylabel('RMFS')
    plt.legend()
    plt.title('MAE vs RMFS')
    plt.savefig('mae_vs_rmfs.png', dpi=300)

# create plots for the molecules of RMFS vs given parameter we change accross runs (in here an example is the number of interactions)
def plot_parameter_vs_rmfs(datasets,final_results):
    for dat in datasets:
        params = []
        rs = []
        for run_id in final_results.keys():
            params.append(final_results[run_id][dat]['params']['num_interactions'])
            rs.append(final_results[run_id][dat]['md_results']['rmfs'])
        plt.figure()
        plt.scatter(np.array(params), np.array(rs), label=labels[run_id])
        plt.xlabel('number of interactions')
        plt.ylabel('RMFS')
        plt.legend()
        plt.title('number of interactions vs RMFS')
        plt.savefig(dat+'_param_vs_rmfs.png', dpi=300)

# estimating direct effect of the parameter of study (number of interactions in a given example) on to RMFS values
def plot_direct_effect(datasets, final_results):
    datum = {'rmfs':[],
             'mae_e':[],
             'size': [],
             'num_interactions':[],
            }
    for run_id in final_results.keys():
        for dat in datasets:
            datum['rmfs'].append(final_results[run_id][dat]['md_results']['rmfs'])
            datum['mae_e'].append(final_results[run_id][dat]['mace_results']['mae_e'])
            datum['size'].append(final_results[run_id][dat]['md_results']['mol_size'])
            datum['num_interactions'].append(final_results[run_id][dat]['params']['num_interactions'])
    df = pd.DataFrame(datum)

    # Define the target (RMSF) and the independent variables (Param, MAE, Size)
    X = df[['num_interactions', 'mae_e', 'size']]  # Adjustment set: MAE and Size
    y = df['rmfs']  # Dependent variable: RMSF

    # Add a constant to the independent variables (for the intercept)
    X = sm.add_constant(X)

    # Fit the OLS regression model
    model = sm.OLS(y, X).fit()
    # direct effect correlation coefficient
    coef = model.params['num_interactions']
    # standard error interval for it
    # we check how different it is from zero
    std_err = model.bse['num_interactions']

    # Plot the coefficient of 'Param' with error bars (representing standard error)
    plt.figure(figsize=(6, 4))
    plt.errorbar(x=[1], y=[coef], yerr=[std_err], fmt='o', capsize=5, label='num_interactions Effect')
    plt.axhline(y=0, color='r', linestyle='--', label='Zero Effect')  # Add a horizontal line at 0 for comparison

    # Add labels and title
    plt.xticks([1], ['Param'])
    plt.ylabel('Coefficient Estimate')
    plt.title('Direct Effect of num_interactions on RMSF with Standard Error')

    # Add a legend
    plt.legend()
    plt.savefig('direct_param_vs_rmfs.png', dpi=300)


# ploting for the experiment
plot_size_vs_rmfs(datasets, final_results)
plot_mae_vs_rmfs(datasets, final_results)
plot_parameter_vs_rmfs(datasets,final_results)
plot_direct_effect(datasets, final_results)
