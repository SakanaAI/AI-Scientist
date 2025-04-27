import subprocess
import os
import sys
import atexit

def run_in_conda_env(env_name):
    """
    Re-run the current script in the specified conda environment
    """
    conda_path = os.environ.get('CONDA_EXE', 'conda')
    current_env = os.environ.get('CONDA_DEFAULT_ENV')
    print("Current environment:", current_env)
    
    if current_env != env_name:
        script_path = os.path.abspath(sys.argv[0])
        script_args = ' '.join(sys.argv[1:])
        cmd = f'"{conda_path}" run -n {env_name} python "{script_path}" {script_args}'
        
        print(f"Switching to {env_name} environment...")
        try:
            process = subprocess.run(cmd, shell=True, check=True)
            sys.exit(process.returncode)
        except subprocess.CalledProcessError as e:
            print(f"Error running script in {env_name}: {str(e)}")
            sys.exit(1)
        finally:
            print(f"Switching back to {current_env} environment...")
