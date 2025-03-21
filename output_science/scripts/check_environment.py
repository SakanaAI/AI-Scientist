#!/usr/bin/env python
"""
Environment Check Script for AI Scientist

This script checks that the environment is correctly set up for running AI Scientist experiments.
It verifies Python packages, API keys, and GPU availability.
"""

import os
import sys
import importlib
import pkg_resources
import subprocess
from collections import namedtuple

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Define required packages
RequiredPackage = namedtuple('RequiredPackage', ['name', 'import_name', 'min_version', 'optional'])

REQUIRED_PACKAGES = [
    RequiredPackage('torch', 'torch', '2.0.0', False),
    RequiredPackage('transformers', 'transformers', '4.30.0', False),
    RequiredPackage('numpy', 'numpy', '1.20.0', False),
    RequiredPackage('matplotlib', 'matplotlib', '3.5.0', False),
    RequiredPackage('openai', 'openai', '0.27.0', False),
    RequiredPackage('anthropic', 'anthropic', '0.5.0', False),
    RequiredPackage('einops', 'einops', '0.6.0', False),
    RequiredPackage('aider', 'aider', '0.14.0', False),
    RequiredPackage('semantic-scholar-api', 'semanticscholar', '1.0.0', True),
    RequiredPackage('pyalex', 'pyalex', '0.9', True),
]

# Required API keys
REQUIRED_KEYS = [
    ('OPENAI_API_KEY', False),  # (name, required)
    ('ANTHROPIC_API_KEY', False),
    ('S2_API_KEY', True),  # Optional
    ('DEEPSEEK_API_KEY', True),  # Optional
    ('OPENROUTER_API_KEY', True),  # Optional
    ('GEMINI_API_KEY', True),  # Optional
]

def check_python_version():
    """Check Python version"""
    print("\n=== Checking Python Version ===")
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major != 3 or python_version.minor < 9:
        print("❌ Python 3.9+ is required")
        return False
    else:
        print("✅ Python version OK")
        return True

def check_packages():
    """Check required packages"""
    print("\n=== Checking Required Packages ===")
    all_packages_ok = True
    
    for package in REQUIRED_PACKAGES:
        try:
            # Check if package is installed
            installed_version = pkg_resources.get_distribution(package.name).version
            
            # Try importing the package
            importlib.import_module(package.import_name)
            
            # Check version if required
            if package.min_version:
                if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(package.min_version):
                    print(f"❌ {package.name} version {installed_version} is below required {package.min_version}")
                    if not package.optional:
                        all_packages_ok = False
                else:
                    print(f"✅ {package.name} {installed_version} (required: {package.min_version})")
            else:
                print(f"✅ {package.name} {installed_version}")
                
        except (pkg_resources.DistributionNotFound, ImportError):
            if package.optional:
                print(f"⚠️ Optional package {package.name} not found")
            else:
                print(f"❌ Required package {package.name} not found")
                all_packages_ok = False
    
    return all_packages_ok

def check_api_keys():
    """Check required API keys"""
    print("\n=== Checking API Keys ===")
    all_keys_ok = True
    has_one_llm_key = False
    
    for key_name, required in REQUIRED_KEYS:
        key_value = os.environ.get(key_name)
        if key_value:
            print(f"✅ {key_name} found")
            # Check if it's an LLM API key
            if key_name in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'DEEPSEEK_API_KEY', 'OPENROUTER_API_KEY', 'GEMINI_API_KEY']:
                has_one_llm_key = True
        else:
            if required:
                print(f"❌ Required key {key_name} not found")
                all_keys_ok = False
            else:
                print(f"⚠️ Optional key {key_name} not found")
    
    if not has_one_llm_key:
        print("❌ At least one LLM API key is required (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)")
        all_keys_ok = False
    
    return all_keys_ok

def check_gpu():
    """Check GPU availability"""
    print("\n=== Checking GPU Availability ===")
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"✅ {device_count} CUDA device(s) available")
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                print(f"   Device {i}: {device_name}")
            return True
        else:
            print("⚠️ CUDA not available. GPU-accelerated experiments may be very slow.")
            return True  # Not failing the check, just warning
    except Exception as e:
        print(f"⚠️ Error checking GPU: {e}")
        return True  # Not failing the check, just warning

def check_directories():
    """Check required directories"""
    print("\n=== Checking Project Structure ===")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    required_dirs = [
        'data',
        'experiments',
        'papers',
        'models',
        'logs',
        'reviews',
        'workflows',
        'scripts',
        'results',
        'config'
    ]
    
    all_dirs_ok = True
    for dir_name in required_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.isdir(dir_path):
            print(f"✅ Directory {dir_name}/ exists")
        else:
            print(f"❌ Directory {dir_name}/ not found")
            all_dirs_ok = False
    
    return all_dirs_ok

def check_project_access():
    """Check access to the main AI Scientist project"""
    print("\n=== Checking Project Access ===")
    try:
        # Try importing a key module from the project
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.append(project_dir)
        
        from ai_scientist import llm
        print(f"✅ Found main project at {project_dir}")
        print(f"✅ Successfully imported ai_scientist modules")
        return True
    except ImportError as e:
        print(f"❌ Could not import project modules: {e}")
        print("Make sure this script is run from within the AI-Scientist project directory")
        return False

def main():
    """Run all checks"""
    print("AI Scientist Environment Check")
    print("==============================")
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_packages),
        ("API Keys", check_api_keys),
        ("GPU Availability", check_gpu),
        ("Project Structure", check_directories),
        ("Project Access", check_project_access)
    ]
    
    results = []
    for name, check_func in checks:
        result = check_func()
        results.append((name, result))
    
    print("\n=== Summary ===")
    all_ok = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
        if not result:
            all_ok = False
    
    if all_ok:
        print("\n🎉 All checks passed! The environment is correctly set up.")
        return 0
    else:
        print("\n⚠️ Some checks failed. Please address the issues above before running experiments.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 