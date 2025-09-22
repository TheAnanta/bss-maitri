"""Configuration management for BSS Maitri"""

import yaml
import os
from typing import Dict, Any
from pathlib import Path

class Config:
    """Configuration manager for BSS Maitri AI Assistant"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def update(self, key: str, value: Any):
        """Update configuration value using dot notation"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    @property
    def model(self) -> Dict[str, Any]:
        return self._config.get('model', {})
    
    @property
    def emotion_detection(self) -> Dict[str, Any]:
        return self._config.get('emotion_detection', {})
    
    @property
    def conversation(self) -> Dict[str, Any]:
        return self._config.get('conversation', {})
    
    @property
    def system(self) -> Dict[str, Any]:
        return self._config.get('system', {})
    
    @property
    def crew_monitoring(self) -> Dict[str, Any]:
        return self._config.get('crew_monitoring', {})