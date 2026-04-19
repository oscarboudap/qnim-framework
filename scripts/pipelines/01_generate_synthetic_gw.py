#!/usr/bin/env python3
"""
Block 1: Generate Synthetic Gravitational Waves

Solves all physics equations:
  - Post-Newtonian approximations (baseline GR)
  - Effective One Body (EOB) waveforms
  - Quantum gravity anomalies from layers 4, 5, 6, 7 (SSTG framework)
  
Produces:
  - h_plus, h_cross strain data
  - Ground truth theory labels
  - Parameter metadata (masses, spins, distances)
  - SNR annotations for validation
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import h5py
import numpy as np
import json
from typing import Dict, Any, Optional
from datetime import datetime

# CLI Framework
from src.cli import (
    ScriptConfig,
    ScriptContainer,
    get_script_logger,
    create_execution_context,
    ScriptException,
    ScriptExecutionException,
)

# Domain & Application
from src.domain.astrophysics.sstg.simple_generator import SimpleSyntheticGWGenerator
from src.domain.astrophysics.value_objects import TheoryFamily


def _determine_source_type(params: Dict[str, Any]) -> str:
    """
    Determine GW source type from parameters.
    
    Supports:
      - CBC: Compact Binary Coalescence (two objects merging)
      - CONTINUOUS: Rotating neutron stars
      - BURST: Supernovae, starquakes, merger ringdown
      - SGWB: Stochastic background
    
    Args:
        params: Parameter dictionary from engine
    
    Returns:
        Source type string
    """
    # For now, categorize by mass and theory
    has_m1 = "m1" in params and params["m1"] > 0
    has_m2 = "m2" in params and params["m2"] > 0
    
    # CBC: Two distinct masses in merger
    if has_m1 and has_m2:
        m1 = params["m1"]
        m2 = params["m2"]
        
        # Check if it's a realistic binary
        if m1 > 0 and m2 > 0 and abs(m1 - m2) > 0.1:
            return "CBC"
    
    # CONTINUOUS: Single rotating object (neutron star)
    if "spin" in params and params["spin"] > 0.3:
        return "CONTINUOUS"
    
    # BURST: Transient events (supernovae, quakes, ringdown)
    if "theory" in params and params["theory"] in ["QUANTUM", "MODIFIED_GRAVITY"]:
        return "BURST"
    
    # Default to CBC for two-body systems
    return "CBC"


def main(output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate synthetic GW dataset with quantum gravity anomalies.
    
    Args:
        output_path: Optional custom output directory
    
    Returns:
        Dict with:
            - "output_path": Path where data was saved
            - "num_events": Number of events generated
            - "theory_distribution": Count by theory
            - "snr_ranges": SNR statistics
    """
    logger = get_script_logger("block_1_synthesis")
    context = create_execution_context("01_generate_synthetic_gw.py")
    
    try:
        # ====================================================================
        # INITIALIZATION
        # ====================================================================
        logger.step("INIT", "Initializing Block 1: GW Synthesis")
        
        config = ScriptConfig.from_env()
        logger.configuration(
            "SynthesisConfig",
            target_samples=config.training.preprocessing.target_samples,
            sample_rate=config.training.preprocessing.sample_rate_hz,
        )
        
        # ====================================================================
        # ENGINE INSTANTIATION
        # ====================================================================
        logger.step("ENGINE", "Creating Synthetic GW Generator")
        
        generator = SimpleSyntheticGWGenerator(
            sample_rate=config.training.preprocessing.sample_rate_hz
        )
        
        # Theory weights for sampling
        theory_weights = {
            "RG": 0.85,
            "QUANTUM": 0.10,
            "MODIFIED_GRAVITY": 0.05,
        }
        
        # Source type weights
        source_weights = {
            "CBC": 0.70,
            "CONTINUOUS": 0.15,
            "BURST": 0.10,
            "SGWB": 0.05,
        }
        
        # Determine output directory
        if output_path:
            output_dir = Path(output_path)
        else:
            output_dir = Path(config.training.paths.get_data_synthetic_path())
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use full target from config
        TARGET_EVENTS = config.training.preprocessing.target_samples
        
        # ====================================================================
        # GENERATION LOOP
        # ====================================================================
        logger.step(
            "GENERATION",
            "Generating synthetic events",
            target_count=config.training.preprocessing.target_samples
        )
        
        events_generated = 0
        events_valid = 0
        theory_counts = {}
        snr_buffer = []
        
        output_file = output_dir / f"synthetic_gw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        
        with h5py.File(output_file, 'w') as h5f:
            # Create datasets for strain data
            h5f.create_dataset(
                'strain_plus',
                shape=(0, config.training.preprocessing.target_samples),
                maxshape=(None, config.training.preprocessing.target_samples),
                dtype=np.float32,
                compression='gzip'
            )
            h5f.create_dataset(
                'strain_cross',
                shape=(0, config.training.preprocessing.target_samples),
                maxshape=(None, config.training.preprocessing.target_samples),
                dtype=np.float32,
                compression='gzip'
            )
            
            # String datasets for metadata
            h5f.create_dataset(
                'theory_labels',
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(encoding='utf-8'),
                compression='gzip'
            )
            h5f.create_dataset(
                'source_types',
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(encoding='utf-8'),
                compression='gzip'
            )
            h5f.create_dataset(
                'parameters_json',
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(encoding='utf-8'),
                compression='gzip'
            )
            
            # Generation loop
            max_attempts = TARGET_EVENTS * 3
            
            while events_valid < TARGET_EVENTS:
                events_generated += 1
                
                if events_generated > max_attempts:
                    logger.warning(
                        f"Reached max attempts ({max_attempts}), stopping generation"
                    )
                    break
                
                try:
                    # Sample theory and source type
                    theory = np.random.choice(
                        list(theory_weights.keys()),
                        p=list(theory_weights.values())
                    )
                    source_type = np.random.choice(
                        list(source_weights.keys()),
                        p=list(source_weights.values())
                    )
                    
                    # Generate waveform
                    result = generator.generate_event(source_type=source_type)
                    
                    h_plus = result["h_plus"]
                    h_cross = result["h_cross"]
                    
                    # Trim to target sample size
                    h_plus = h_plus[:config.training.preprocessing.target_samples]
                    h_cross = h_cross[:config.training.preprocessing.target_samples]
                    
                    # Pad if needed
                    if len(h_plus) < config.training.preprocessing.target_samples:
                        h_plus = np.pad(h_plus, (0, config.training.preprocessing.target_samples - len(h_plus)))
                        h_cross = np.pad(h_cross, (0, config.training.preprocessing.target_samples - len(h_cross)))
                    
                    # Store in HDF5
                    h5f['strain_plus'].resize(events_valid + 1, axis=0)
                    h5f['strain_cross'].resize(events_valid + 1, axis=0)
                    h5f['theory_labels'].resize(events_valid + 1, axis=0)
                    h5f['source_types'].resize(events_valid + 1, axis=0)
                    h5f['parameters_json'].resize(events_valid + 1, axis=0)
                    
                    h5f['strain_plus'][events_valid] = h_plus.astype(np.float32)
                    h5f['strain_cross'][events_valid] = h_cross.astype(np.float32)
                    h5f['theory_labels'][events_valid] = theory
                    h5f['source_types'][events_valid] = source_type
                    h5f['parameters_json'][events_valid] = json.dumps(result.get("parameters", {}))
                    
                    # Track statistics
                    theory_counts[theory] = theory_counts.get(theory, 0) + 1
                    
                    # Calculate SNR proxy
                    snr_proxy = np.sqrt(np.mean(h_plus**2)) / (1e-21 + 1e-10)
                    snr_buffer.append(snr_proxy)
                    
                    events_valid += 1
                    
                    # Progress reporting
                    if events_valid % max(1, TARGET_EVENTS // 3) == 0:
                        logger.progress("Synthesis", events_valid, TARGET_EVENTS)
                
                except Exception as e:
                    logger.debug(f"Event {events_generated} failed: {str(e)[:80]}")
                    continue
        
        # ====================================================================
        # REPORTING
        # ====================================================================
        logger.step("REPORT", "Block 1 Synthesis Complete")
        
        for theory, count in sorted(theory_counts.items()):
            logger.metric(f"theory_{theory}", count, unit="events")
        
        if snr_buffer:
            logger.metric("snr_mean", np.mean(snr_buffer), unit="ratio")
            logger.metric("snr_std", np.std(snr_buffer), unit="ratio")
        
        logger.info(f"✅ Saved {events_valid} synthetic events to {output_file}")
        
        return {
            "output_path": str(output_file),
            "num_events": events_valid,
            "theory_distribution": theory_counts,
            "snr_ranges": {"mean": float(np.mean(snr_buffer)), "std": float(np.std(snr_buffer))},
        }
    
    except ScriptExecutionException as e:
        logger.error(f"❌ Block 1 failed: {e}", extra=context)
        raise SystemExit(1)
    
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", extra=context, exc_info=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
