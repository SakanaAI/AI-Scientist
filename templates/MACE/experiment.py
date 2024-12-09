import subprocess
import sys
from ase import units
from ase.md.langevin import Langevin
from ase.io import read
from ase.optimize import BFGS
import numpy as np
from mace.calculators import MACECalculator
import argparse
import json
import os
from ase.io import read
import MDAnalysis as mda

# train MACE model with given parameters, given run_id, and given dataset taken from a folder with the same name as dataset

def train_mace_model(run_id, dataset, params):
    # do not run more than 100 max_num_epochs for these molecules

    # if no folder for run_id and dataset create it it
    run_directory = './'+run_id+'/'+dataset
    if not os.path.exists(run_directory):
       os.makedirs(run_directory)
    
    # command to run mace_run_train with given parameters, given run_id, and given dataset taken from a folder with the same name as dataset
    # parameteres related to the model are taken from the params dictionary
    command = ['mace_run_train', '--results_dir', run_directory, 
            '--checkpoints_dir', run_directory, 
            '--name', run_id+'_'+dataset,
            '--seed', '2024', '--train_file', dataset+'/'+dataset+'_train.xyz', '--test_file', dataset+'/'+dataset+'_test.xyz', '--valid_fraction','0.05',
            '--config_type_weights', '{Default: 1.0}', '--num_interactions', str(params['num_interactions']), '--num_channels', '128',
            '--max_L', str(params['max_L']), '--correlation', '3', '--E0s', 'average', '--model', 'MACE', 
            '--hidden_irreps', params['hidden_irreps'], '--r_max', str(params[('r_max')]), '--batch_size','16', '--ema',
            '--ema_decay', '0.99', '--amsgrad', '--max_num_epochs', '50', '--device', 'cuda', '--error_table', 'PerAtomRMSE'
          ]

    # Start the training
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Continuously read and print output in real-time
    for line in process.stdout:
        print(line, end='')

    # Optionally, handle errors in a similar way
    for line in process.stderr:
        print(line, end='', file=sys.stderr)

    # Wait for the process to complete
    process.wait()

# run molecular dynamics for a given run_id, dataset, number of steps and temperature
# after running md simulation RMFS and molecular size are calculated to be saved in a json file
# do not run more than 10000 steps on this machine
def run_md(run_id, dataset, steps=10000, temp=310):
    calculator = MACECalculator(model_paths='./'+run_id+'_'+dataset+'.model', device='cuda')
    init_conf = read(dataset+'/'+dataset+'_i.xyz')
    init_conf.set_calculator(calculator)
    
    print ("run optinization")
    opt = BFGS(init_conf)
    opt.run(fmax=0.01)

    run_directory = run_id+'/'+dataset
    md_file_path = os.path.join(run_directory, 'md.xyz')

    print ('run MD simulation')
    dyn = Langevin(init_conf, 0.5*units.fs, temperature_K=temp, friction=5e-3)
    def write_frame():
            dyn.atoms.write(md_file_path, append=True)
    dyn.attach(write_frame, interval=50)
    dyn.run(steps)
    print("MD finished!")
    results = {'steps':steps, 'temp':temp, 'rmfs':rmfs(run_id,dataset), 'mol_size':mol_size(dataset)}

    # after done with the trajectory remove the trajectory file and model files to save the disk space
    if os.path.exists(md_file_path):
        os.remove(md_file_path)
    if os.path.exists('./'+run_id+'_'+dataset+'.model'):
        os.remove('./'+run_id+'_'+dataset+'.model')
    if os.path.exists('./'+run_id+'_'+dataset+'_compiled.model'):
        os.remove('./'+run_id+'_'+dataset+'_compiled.model')
    return results

def read_and_save_results(run_id, dataset):
    run_directory = run_id+'/'+dataset
    file_path = run_directory+'/'+run_id+'_'+dataset+'_run-2024_train.txt'
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            record = json.loads(line.strip())
            data.append(record)
    for entry in data[::-1]:
        if entry['mode']=='eval':
           results = entry
           break
    return results            

def rmfs(run_id, dataset):
    # rmfs for the trajectory, useful to describe how stable is the trajectory,
    # or how flexible is the molecule

    traj_file = run_id+'/'+dataset+'/md.xyz'
    u = mda.Universe(traj_file, format='XYZ')
    positions = np.zeros((len(u.trajectory), u.atoms.n_atoms, 3))

    for ts in u.trajectory:
        positions[ts.frame] = u.atoms.positions

    average_positions = np.mean(positions, axis=0)
    rmfs_a = np.sqrt(np.mean((positions - average_positions)**2, axis=0))
    return np.std(rmfs_a)/np.mean(rmfs_a)

def load_xyz(file_path):
    # Load XYZ file and return atom coordinates as a NumPy array
    with open(file_path, 'r') as file:
        lines = file.readlines()

    num_atoms = int(lines[0].strip())
    atom_coords = []

    for line in lines[2:2+num_atoms]:
        parts = line.split()
        atom_coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return np.array(atom_coords)

def calculate_distances(coords):
    # Calculate distances between all pairs of atoms in the molecule
    num_atoms = coords.shape[0]
    distances = []

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = np.linalg.norm(coords[i] - coords[j])
            distances.append(distance)

    return np.array(distances)

def mol_size(dat):
    # mean and variance in interatomic distances for the molecule
    # as proxy to describe molecular size
    xyz_file = dat+'/'+dat+"_i.xyz"

    # Load the molecule using ASE's read function
    atoms = read(xyz_file)
    # Get the positions of all atoms
    positions = atoms.get_positions()
    
    # Find the minimum and maximum coordinates along each axis
    min_coords = np.min(positions, axis=0)
    max_coords = np.max(positions, axis=0)
    
    # Calculate the size of the molecule as the diagonal of the bounding box
    size = np.linalg.norm(max_coords - min_coords)
    
    return size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    args = parser.parse_args()
     
    run_id = args.out_dir
    all_results = {}
    final_info = {}

    for dataset in ['ethanol', 'benzene', 'uracil','naphthalene','aspirin','salicylic','malonaldehyde','toluene','paracetamol','azobenzene']:
        all_results[dataset] = {}

        run_directory = run_id+'/'+dataset
        if not os.path.exists('./'+run_directory):
           os.makedirs('./'+run_directory)
        
        # num_interactions is [1,2,3,4,5] r_max radius < 20.0 and > 1.0, max_L is in [1,2,3,4]
        all_results[dataset]['params'] = {'hidden_irreps':'128x0e + 128x1o', 'r_max':5.0, 'num_interactions':2, 'max_L':1}
        train_mace_model(run_id, dataset,all_results[dataset]['params'])
        all_results[dataset]['mace_results'] = read_and_save_results(run_id, dataset)
        all_results[dataset]['md_results'] = run_md(run_id, dataset)
        

    final_info['MACE'] = {'means':all_results}

    with open(run_id+'/final_info.json', 'w') as json_file:
      json.dump(final_info, json_file, indent=4)  

    
