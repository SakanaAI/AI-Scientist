# AI-SCIENTIST requires Python 3.11, and DeepMimic requires Python 3.7
# To circumvent this, 2 separate conda envs are used, and the Python 3.7 conda env
# named "deep_mimic_env" is toggled on/off at the start/end of this file
import os
from environment_handler import run_in_conda_env
if __name__ == '__main__':
    if os.environ.get('CONDA_DEFAULT_ENV') != 'deep_mimic_env':
        run_in_conda_env('deep_mimic_env')

import argparse
import json
import os.path as osp
import pathlib
import pickle
import time
import os
import sys
import subprocess
import numpy as np
import tensorflow as tf  # Added for tf.cast

# Add DeepMimic to Python path
deepmimic_path = osp.join(osp.dirname(__file__), 'DeepMimic')
sys.path.append(deepmimic_path)

from env.deepmimic_env import DeepMimicEnv
from learning.amp_agent import AMPAgent
from learning.rl_world import RLWorld
from util.arg_parser import ArgParser
import util.mpi_util as MPIUtil

# Seed for reproducibility
np.random.seed(42)

# Concatenate the root folder name to file paths in order to run
# experiment.py from ~/templates/amp/
root_folder_name = "DeepMimic"

def calc_pose_error(env, agent_states, motion_states):
    """Calculate pose error between generated and reference motion
    
    Args:
        env: DeepMimic environment
        agent_states: Current agent state from env.record_state()
        motion_states: Reference motion state from env.record_amp_obs_expert()
        
    Returns:
        float: Mean pose error between agent and reference motion
    """
    # The states come from the DeepMimic environment in specific formats:
    # - Root position (3D)
    # - Root rotation (quaternion 4D)
    # - Joint angles
    
    # We know from the DeepMimic codebase these are typically numpy arrays
    if not isinstance(agent_states, np.ndarray):
        agent_states = np.array(agent_states)
    if not isinstance(motion_states, np.ndarray):
        motion_states = np.array(motion_states)

    # In DeepMimic, the states are laid out in a fixed format
    # First 3 values are root position (x,y,z)
    pos_err = np.linalg.norm(agent_states[0:3] - motion_states[0:3])
    
    # Next 4 values are root rotation quaternion (w,x,y,z)
    # For quaternions, dot product gives cosine of angle between rotations
    quat_dot = np.clip(np.abs(np.sum(agent_states[3:7] * motion_states[3:7])), -1.0, 1.0)
    rot_err = 1.0 - quat_dot  # Convert to error where 0 means identical rotations
    
    # Remaining values are joint angles
    joint_err = np.mean(np.abs(agent_states[7:] - motion_states[7:]))
    
    # Weight the errors according to typical values used in character animation
    # These weights prioritize joint angles since they're most visible
    total_err = 0.1 * pos_err + 0.2 * rot_err + 0.7 * joint_err
    
    return total_err
    
def create_deepmimic_env(motion_file, enable_draw=False):
    """
    Create a DeepMimic environment with specified motion file.
    
    Args:
        motion_file (str): Path to the motion file to be used
        enable_draw (bool): Whether to enable drawing (visualization)
    
    Returns:
        DeepMimicEnv: Configured environment
    """
    env_args = [
        "--scene", "imitate_amp",
        
        # Time params
        "--time_lim_min", "0.5",
        "--time_lim_max", "0.5", 
        "--time_end_lim_min", "20",
        "--time_end_lim_max", "20",
        "--anneal_samples", "32000000",
        
        # Simulation params
        "--num_update_substeps", "10",
        "--num_sim_substeps", "2",
        "--world_scale", "4",
        
        # Terrain
        "--terrain_file", f"{root_folder_name}/data/terrain/plane.txt",
        
        # Character config
        "--char_types", "general",
        "--character_files", f"{root_folder_name}/data/characters/humanoid3d.txt",
        "--enable_char_soft_contact", "false",
        "--fall_contact_bodies", "0", "1", "2", "3", "4", "6", "7", "8", "9", "10", "12", "13", "14",
        
        # Controller config  
        "--char_ctrls", "ct_pd",
        "--char_ctrl_files", f"{root_folder_name}/data/controllers/humanoid3d_rot_ctrl.txt",
        "--kin_ctrl", "motion",
        "--motion_file", motion_file,
        "--sync_char_root_pos", "true",
        "--sync_char_root_rot", "false",
        
        # AMP specific settings
        "--enable_amp_obs_local_root", "true",
        "--enable_amp_task_reward", "true"
    ]
    
    return DeepMimicEnv(env_args, enable_draw)

def run_amp_experiment(config, motion_file):
    env = create_deepmimic_env(motion_file)
    
    # Create argument parser and load agent configuration
    arg_parser = ArgParser()

    # Important: Use the same agent config as test.py
    temp_dir = "output/temp" 
    os.makedirs(temp_dir, exist_ok=True)
    
    agent_config = {
        "AgentType": "AMP",
        "ActorNet": "fc_2layers_1024units",
        "CriticNet": "fc_2layers_1024units",
        "DiscNet": "fc_2layers_1024units",
        "ActorInitOutputScale": 0.01,
        "DiscInitOutputScale": 0.01,
        "ReplayBufferSize": 100000,
        "MiniBatchSize": 32,
        "DiscMiniBatchSize": 32,
        "InitSamples": 300,
        "ItersPerUpdate": 2,
        "ActorStepsize": 0.0002,
        "CriticStepsize": 0.001,
        "DiscStepsize": 0.001,
        "Epochs": 3,
        "Discount": 0.95,
        "TDLambda": 0.95,
        "RewardScale": 2.0,
        "RatioClip": 0.2,
        "ExpParams": {
            "Rate": 0.95,
            "Noise": 0.2
        },
        "TaskRewardLerp": 0.7,
        "DiscLogitRegWeight": 0.1,
        "DiscWeightDecay": 0.00001,
        "DiscGradPenalty": 1.0,
        "ExpAnnealSamples": 48000000
    }
    
    agent_file = os.path.join(temp_dir, "test_agent.json")
    with open(agent_file, 'w') as f:
        json.dump(agent_config, f, indent=4)

    arg_parser.load_args([
        "--agent_files", agent_file,
        "--output_path", config.out_dir,
        "--int_output_path", f"{config.out_dir}/intermediate"
    ])

    world = RLWorld(env, arg_parser)
    
    fps = 60
    timestep = 1.0 / fps
    train_steps = config.num_train_steps
    
    # Track metrics throughout training, not just at end
    train_losses = []
    disc_rewards = []
    latest_disc_reward = 0
    latest_disc_reward_std = 0
    
    print(f"Training AMP for motion: {motion_file}")
    global_step = 0

    # Add new tracking variables
    pose_errors = []
    curr_pose_error = 0.0
    
    while global_step < train_steps:
        world.update(timestep)
        
        # Collect metrics periodically (every 100 steps like in test.py)
        if global_step % 100 == 0 and world.agents and world.agents[0] is not None:
            agent = world.agents[0]
            
            # Get discriminator reward mean and std
            if hasattr(agent, '_disc_reward_mean'):
                latest_disc_reward = MPIUtil.reduce_avg(agent._disc_reward_mean)
                disc_rewards.append(latest_disc_reward)
                
            if hasattr(agent, '_disc_reward_std'):
                latest_disc_reward_std = MPIUtil.reduce_avg(agent._disc_reward_std)
                
            # Get training loss
            if hasattr(agent, 'logger') and agent.logger.get_num_keys() > 0:
                loss = agent.logger.log_current_row.get('Disc_Loss', 0.0)
                if loss != 0.0:
                    train_losses.append(loss)

            # Get current agent states
            agent_states = env.record_state(agent.id)
            # Get reference motion states
            motion_states = env.record_amp_obs_expert(agent.id)
            
            # Calculate pose error
            curr_pose_error = calc_pose_error(env, agent_states, motion_states)
            pose_errors.append(curr_pose_error)
                    
            samples_collected = agent.replay_buffer.total_count
            
        if env.is_episode_end():
            world.end_episode()
            world.reset()
            
        global_step += 1
    
    final_metrics = {
        "training_time": global_step * timestep,
        "samples_collected": samples_collected,

        # disc_reward_mean - nested under "means" to conform to AI-Scientist
        "means": {
            "disc_reward_mean": float(latest_disc_reward)
        },
        "stderrs": {
            "disc_reward_std": float(latest_disc_reward_std)
        },
        
        "train_losses": train_losses,
        "pose_errors": pose_errors,
        "final_pose_error": float(curr_pose_error)
    }
    
    return final_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_steps", type=int, default=10000)
    parser.add_argument("--out_dir", type=str, default="run_0")
    config = parser.parse_args()

    # Ensure output directory exists
    pathlib.Path(config.out_dir).mkdir(parents=True, exist_ok=True)

    # List of motion files to experiment with
    motion_files = [
        f"{root_folder_name}/data/motions/humanoid3d_walk.txt",
        f"{root_folder_name}/data/motions/humanoid3d_jog.txt", 
        f"{root_folder_name}/data/motions/humanoid3d_run.txt"
    ]

    # Store results for each motion
    all_results = {}

    # Run experiments for each motion
    for motion_file in motion_files:
        motion_name = os.path.splitext(os.path.basename(motion_file))[0]
        print(f"Running experiment for motion: {motion_name}")
        
        results = run_amp_experiment(config, motion_file)
        all_results[motion_name] = results

    # Save results
    with open(osp.join(config.out_dir, "final_info.json"), "w") as f:
        json.dump(all_results, f, indent=4)

    with open(osp.join(config.out_dir, "all_results.pkl"), "wb") as f:
        pickle.dump(all_results, f)

    print("Experiment completed successfully.")

if __name__ == "__main__":
    main()