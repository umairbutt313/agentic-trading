#!/usr/bin/env python3
"""
Environment Variable Loader Utility
Provides a centralized way to load .env file from project root
"""

import os
from dotenv import load_dotenv

def load_project_env():
    """
    Load environment variables from the project root .env file.
    This function should be called by all Python modules that need environment variables.
    
    Returns:
        bool: True if .env file was found and loaded, False otherwise
    """
    # Get the project root directory (parent of utils)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    env_path = os.path.join(project_root, '.env')
    
    # Load the .env file
    success = load_dotenv(env_path)
    
    if success:
        print(f"✓ Environment variables loaded from: {env_path}")
    else:
        print(f"⚠ Warning: .env file not found at: {env_path}")
    
    return success

def get_env_var(key: str, default: str = None, required: bool = False):
    """
    Get an environment variable with better error handling.
    
    Args:
        key (str): Environment variable name
        default (str): Default value if not found
        required (bool): If True, raises error if variable not found
        
    Returns:
        str: Environment variable value
        
    Raises:
        ValueError: If required variable is not found
    """
    value = os.getenv(key, default)
    
    if required and value is None:
        raise ValueError(f"Required environment variable '{key}' not found in .env file")
    
    return value

# Auto-load environment variables when this module is imported
load_project_env()