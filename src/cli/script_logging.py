"""
Script Layer Structured Logging

Provides consistent, structured logging across all scripts with:
  - Multiple severity levels
  - Contextual information
  - Execution timing
  - Progress tracking

Usage:
  logger = get_script_logger('train')
  logger.info('Starting training', extra={'epochs': 100, 'batch_size': 32})
  logger.progress('Training', 50, 100)
  logger.error('Training failed', exc_info=exc)
"""

import logging
import sys
from typing import Optional, Any, Dict
from datetime import datetime


class ProgressTracker:
    """Tracks progress with consistent formatting."""
    
    def __init__(self, name: str, total: int) -> None:
        """
        Initialize progress tracker.
        
        Args:
          name: Operation name
          total: Total items/iterations
        """
        self.name = name
        self.total = total
        self.current = 0
    
    def update(self, current: int) -> str:
        """
        Update progress and return formatted string.
        
        Args:
          current: Current progress count
        
        Returns:
          Formatted progress string like "[50/100] Training... 50.0%"
        """
        self.current = current
        percent = (current / self.total * 100) if self.total > 0 else 0
        return f"[{current}/{self.total}] {self.name}... {percent:.1f}%"


class ScriptLogger(logging.Logger):
    """Custom logger with script-specific methods."""
    
    def progress(self, name: str, current: int, total: int, level: int = logging.INFO) -> None:
        """
        Log progress with consistent formatting.
        
        Args:
          name: Operation name
          current: Current progress count
          total: Total items
          level: Log level (default INFO)
        """
        tracker = ProgressTracker(name, total)
        message = tracker.update(current)
        self.log(level, message)
    
    def step(self, stage: str, message: str, **context: Any) -> None:
        """
        Log a processing step with context.
        
        Args:
          stage: Processing stage name
          message: Step description
          context: Additional contextual information
        """
        context_str = " | ".join(f"{k}={v}" for k, v in context.items())
        full_msg = f"[{stage}] {message}"
        if context_str:
            full_msg += f" ({context_str})"
        self.info(full_msg)
    
    def metric(self, name: str, value: Any, unit: str = "", **tags: Any) -> None:
        """
        Log a metric with unit and tags.
        
        Args:
          name: Metric name
          value: Metric value
          unit: Unit of measurement (optional)
          tags: Additional tags for filtering
        """
        unit_str = f" {unit}" if unit else ""
        tags_str = " | ".join(f"{k}={v}" for k, v in tags.items())
        suffix = f" ({tags_str})" if tags_str else ""
        self.info(f"METRIC: {name}={value}{unit_str}{suffix}")
    
    def configuration(self, config_name: str, **params: Any) -> None:
        """
        Log configuration parameters.
        
        Args:
          config_name: Configuration section name
          params: Configuration parameters
        """
        params_str = " | ".join(f"{k}={v}" for k, v in params.items())
        self.info(f"CONFIG[{config_name}]: {params_str}")


def get_script_logger(name: str) -> ScriptLogger:
    """
    Get or create a script logger with consistent formatting.
    
    Args:
      name: Logger name (typically script name)
    
    Returns:
      ScriptLogger: Configured logger instance
    
    Example:
      logger = get_script_logger('train')
      logger.info('Starting training')
      logger.progress('Training', 50, 100)
      logger.metric('loss', 0.25, unit='cross_entropy')
    """
    logging.setLoggerClass(ScriptLogger)
    logger = logging.getLogger(f"qnim.scripts.{name}")
    
    # Only add handler if not already present
    if not logger.handlers:
        # Console handler with color support
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        
        # Format: [HH:MM:SS] LEVEL | Logger | Message
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


def create_execution_context(script_name: str, **context: Any) -> Dict[str, Any]:
    """
    Create a structured execution context for logging.
    
    Args:
      script_name: Name of executing script
      context: Additional context items
    
    Returns:
      Dictionary with execution metadata
    
    Example:
      ctx = create_execution_context('train', batch_size=32, epochs=100)
      logger.info('Starting', extra=ctx)
    """
    return {
        "script": script_name,
        "timestamp": datetime.now().isoformat(),
        "pid": __import__("os").getpid(),
        **context
    }
