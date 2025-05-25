"""
Helper module for standardized judge configuration across all tasks.
This module provides utilities to create judge instances with environment-based configuration.
"""

import os
from typing import Optional, Dict, Any
from .judge_utils import SimplifiedJudge
from loguru import logger as eval_logger


def get_judge_model_name(yaml_config: Optional[Dict[str, Any]] = None, 
                         default: str = "gpt-4-0613") -> str:
    """
    Get the judge model name from environment or config.
    
    Priority order:
    1. MODEL_VERSION environment variable
    2. yaml_config["metadata"]["gpt_eval_model_name"] (if provided)
    3. default parameter
    
    Args:
        yaml_config: Optional YAML configuration dict
        default: Default model name if not found elsewhere
        
    Returns:
        Model name to use for evaluation
    """
    # First check environment variable
    model_name = os.getenv("MODEL_VERSION")
    if model_name:
        eval_logger.info(f"Using model from environment: {model_name}")
        return model_name
    
    # Then check YAML config
    if yaml_config:
        model_name = yaml_config.get("metadata", {}).get("gpt_eval_model_name")
        if model_name:
            eval_logger.info(f"Using model from config: {model_name}")
            return model_name
    
    # Finally use default
    eval_logger.info(f"Using default model: {default}")
    return default


def create_judge(yaml_config: Optional[Dict[str, Any]] = None,
                 default_model: str = "gpt-4-0613",
                 temperature: float = 0.0,
                 **kwargs) -> SimplifiedJudge:
    """
    Create a SimplifiedJudge instance with environment-based configuration.
    
    This function automatically uses the following environment variables:
    - API_TYPE: 'openai' or 'azure' (defaults to 'openai')
    - MODEL_VERSION: Model name to use
    - OPENAI_API_KEY: For OpenAI API
    - AZURE_API_KEY, AZURE_ENDPOINT, API_VERSION: For Azure OpenAI
    
    Args:
        yaml_config: Optional YAML configuration dict for fallback
        default_model: Default model name if not found in env or config
        temperature: Temperature for the judge model
        **kwargs: Additional arguments passed to SimplifiedJudge
        
    Returns:
        Configured SimplifiedJudge instance
    """
    model_name = get_judge_model_name(yaml_config, default_model)
    
    # Log configuration for debugging
    api_type = os.getenv("API_TYPE", "openai")
    eval_logger.debug(f"Creating judge with API type: {api_type}, model: {model_name}")
    
    return SimplifiedJudge(
        model_name=model_name,
        temperature=temperature,
        **kwargs
    )


def log_judge_config():
    """Log current judge configuration for debugging."""
    api_type = os.getenv("API_TYPE", "openai")
    model_version = os.getenv("MODEL_VERSION", "Not set")
    
    eval_logger.info("Judge Configuration:")
    eval_logger.info(f"  API_TYPE: {api_type}")
    eval_logger.info(f"  MODEL_VERSION: {model_version}")
    
    if api_type == "azure":
        azure_endpoint = os.getenv("AZURE_ENDPOINT", "Not set")
        api_version = os.getenv("API_VERSION", "Not set")
        eval_logger.info(f"  AZURE_ENDPOINT: {azure_endpoint}")
        eval_logger.info(f"  API_VERSION: {api_version}")
        eval_logger.info(f"  AZURE_API_KEY: {'Set' if os.getenv('AZURE_API_KEY') else 'Not set'}")
    else:
        eval_logger.info(f"  OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")