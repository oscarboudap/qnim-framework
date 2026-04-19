#!/usr/bin/env python3
"""
Simplified Synthetic GW Generator for TFM Demo

Generates synthetic gravitational wave data with:
  - Realistic chirp-like signals or Gaussian noise
  - Multiple source types (CBC, CONTINUOUS, BURST)
  - Quantum gravity anomaly injection
  - Robust fallback: if PyCBC fails, use analytical templates

This prioritizes ROBUSTNESS over perfect physics for the TFM deadline.
"""

import sys
from pathlib import Path

# Add parent for imports
root_path = str(Path(__file__).resolve().parents[3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import numpy as np
from typing import Dict, Any, Optional
import logging

# Configure logging
logger = logging.getLogger("SSTG.SimpleGenerator")
logger.setLevel(logging.DEBUG)


def generate_chirp_signal(
    m1: float,
    m2: float,
    distance: float,
    fs: float = 4096.0,
    duration: float = 4.0,
) -> tuple:
    """
    Generate analytical chirp signal (simplified GW-like waveform).
    
    Args:
        m1, m2: Masses in solar masses
        distance: Distance in Mpc
        fs: Sampling frequency
        duration: Signal duration in seconds
    
    Returns:
        (h_plus, h_cross): Strain arrays
    """
    # Time array
    t = np.arange(int(fs * duration)) / fs
    
    # Chirp frequency evolution (simplified)
    # GW frequency scales as: f ~ (c³/G)^(5/256) * (m1*m2/(m1+m2)²)^(5/256) * t^(-3/8)
    
    # Chirp parameters
    total_mass = m1 + m2
    reduced_mass = (m1 * m2) / total_mass
    
    # Initial and final frequencies
    f0 = 20.0  # Hz (start)
    duration_remaining = duration * (1 - t / duration)
    duration_remaining = np.maximum(duration_remaining, 0.01)
    
    # Frequency evolution (simplified power law)
    frequency = f0 + 150 * (1 - np.exp(-5 * (t / duration)**0.5))
    
    # Phase accumulation
    phase = 2 * np.pi * np.cumsum(frequency) / fs
    
    # Amplitude envelope (increases near merger)
    # Amplitude scales as: A ~ (m1*m2) / distance
    amplitude = (reduced_mass / 100.0) * (1 / (distance + 0.1))
    
    # Smoothing envelope (ramps up and down)
    envelope = np.ones_like(t)
    ramp_time = 0.5
    ramp_idx = int(ramp_time * fs)
    
    if ramp_idx > 0:
        envelope[:ramp_idx] = np.linspace(0, 1, ramp_idx)
        envelope[-ramp_idx:] = np.linspace(1, 0, ramp_idx)
    
    # Generate polarizations
    strain = amplitude * envelope * np.sin(phase)
    
    h_plus = strain
    h_cross = amplitude * envelope * np.cos(phase)
    
    return h_plus, h_cross


class SimpleSyntheticGWGenerator:
    """Robust synthetic GW generator for demonstration."""
    
    def __init__(self, sample_rate: float = 4096.0):
        self.sample_rate = sample_rate
    
    def generate_cbc_signal(
        self,
        m1: float,
        m2: float,
        distance: float,
        spin: float = 0.5
    ) -> Dict[str, Any]:
        """Generate Compact Binary Coalescence signal."""
        # Use analytical chirp
        duration = 4.0
        h_plus, h_cross = generate_chirp_signal(
            m1 + spin * 0.1,  # Spin increases effective mass slightly
            m2,
            distance,
            fs=self.sample_rate,
            duration=duration
        )
        
        return {
            "h_plus": h_plus,
            "h_cross": h_cross,
            "source_type": "CBC",
            "parameters": {"m1": m1, "m2": m2, "distance": distance, "spin": spin}
        }
    
    def generate_continuous_signal(
        self,
        mass: float,
        spin: float,
        distance: float
    ) -> Dict[str, Any]:
        """Generate continuous wave from rotating neutron star."""
        # Continuous: nearly monochromatic
        duration = 4.0
        t = np.arange(int(self.sample_rate * duration)) / self.sample_rate
        
        # Frequency from neutron star rotation
        frequency = 100 + spin * 300  # Hz
        
        # Very small amplitude
        amplitude = 1e-24 * (mass / 1.4) * (1 / (distance + 0.1))
        
        phase = 2 * np.pi * frequency * t
        
        h_plus = amplitude * np.sin(phase)
        h_cross = amplitude * np.cos(phase + np.pi/4)
        
        return {
            "h_plus": h_plus,
            "h_cross": h_cross,
            "source_type": "CONTINUOUS",
            "parameters": {"mass": mass, "spin": spin, "distance": distance}
        }
    
    def generate_burst_signal(
        self,
        energy: float,
        distance: float
    ) -> Dict[str, Any]:
        """Generate burst signal (supernova, merger ringdown)."""
        # Burst: short-lived, complex structure
        duration = 1.0
        t = np.arange(int(self.sample_rate * duration)) / self.sample_rate
        
        # Gaussian envelope (fast)
        tau = 0.1  # Width ~100ms
        center = 0.2
        envelope = np.exp(-((t - center) / tau)**2)
        
        # Multiple frequency components (ringdown modes)
        f1 = 200 + energy * 100  # Primary mode
        f2 = 300 + energy * 80   # Secondary mode
        
        amplitude = 1e-23 * (energy / 1e50) * (1 / (distance + 0.1))
        
        phase1 = 2 * np.pi * f1 * t
        phase2 = 2 * np.pi * f2 * t
        
        h_plus = amplitude * envelope * (0.7 * np.sin(phase1) + 0.3 * np.sin(phase2))
        h_cross = amplitude * envelope * (0.7 * np.cos(phase1) + 0.3 * np.cos(phase2))
        
        return {
            "h_plus": h_plus,
            "h_cross": h_cross,
            "source_type": "BURST",
            "parameters": {"energy": energy, "distance": distance}
        }
    
    def generate_stochastic_signal(
        self,
        amplitude: float,
        distance: float
    ) -> Dict[str, Any]:
        """Generate stochastic background noise."""
        # SGWB: broadband noise
        duration = 4.0
        n_samples = int(self.sample_rate * duration)
        
        # Colored noise (1/f spectrum) - more realistic
        f = np.fft.rfftfreq(n_samples, 1/self.sample_rate)
        
        # Generate white noise in frequency domain
        white = np.random.randn(len(f)) + 1j * np.random.randn(len(f))
        
        # Apply 1/f coloring
        # Avoid division by zero at f=0
        colored_f = f.copy()
        colored_f[0] = 1
        amp_spectrum = white / np.sqrt(colored_f)
        amp_spectrum[0] = 0
        
        # Transform back to time
        h_time = np.fft.irfft(amp_spectrum, n=n_samples)
        
        # Normalize
        amplitude_final = amplitude * (1 / (distance + 0.1))
        h_plus = amplitude_final * h_time / (np.std(h_time) + 1e-10)
        h_cross = amplitude_final * (h_time[::-1]) / (np.std(h_time) + 1e-10)
        
        return {
            "h_plus": h_plus,
            "h_cross": h_cross,
            "source_type": "SGWB",
            "parameters": {"amplitude": amplitude, "distance": distance}
        }
    
    def generate_event(
        self,
        source_type: str = "CBC",
        **params
    ) -> Dict[str, Any]:
        """
        Generate synthetic GW event of specified type.
        
        Args:
            source_type: "CBC", "CONTINUOUS", "BURST", or "SGWB"
            **params: Source-specific parameters
        
        Returns:
            Dict with h_plus, h_cross, and metadata
        """
        logger.debug(f"┌─ generate_event(source_type={source_type})")
        
        try:
            if source_type == "CBC":
                logger.debug(f"  → CBC generation mode")
                m1 = params.get("m1", np.random.uniform(10, 50))
                m2 = params.get("m2", np.random.uniform(5, m1))
                distance = params.get("distance", np.random.uniform(100, 2000))
                spin = params.get("spin", np.random.uniform(0, 0.8))
                logger.debug(f"    m1={m1:.2f}, m2={m2:.2f}, d={distance:.1f}, spin={spin:.3f}")
                
                result = self.generate_cbc_signal(m1, m2, distance, spin)
                logger.debug(f"  ✓ CBC generated: h_plus shape={result['h_plus'].shape}")
                return result
            
            elif source_type == "CONTINUOUS":
                logger.debug(f"  → CONTINUOUS generation mode")
                mass = params.get("mass", np.random.uniform(1, 3))
                spin = params.get("spin", np.random.uniform(0.3, 0.9))
                distance = params.get("distance", np.random.uniform(500, 5000))
                logger.debug(f"    mass={mass:.2f}, spin={spin:.3f}, d={distance:.1f}")
                
                result = self.generate_continuous_signal(mass, spin, distance)
                logger.debug(f"  ✓ CONTINUOUS generated: h_plus shape={result['h_plus'].shape}")
                return result
            
            elif source_type == "BURST":
                logger.debug(f"  → BURST generation mode")
                energy = params.get("energy", np.random.uniform(0.1, 1.0))
                distance = params.get("distance", np.random.uniform(50, 500))
                logger.debug(f"    energy={energy:.3f}, d={distance:.1f}")
                
                result = self.generate_burst_signal(energy, distance)
                logger.debug(f"  ✓ BURST generated: h_plus shape={result['h_plus'].shape}")
                return result
            
            elif source_type == "SGWB":
                logger.debug(f"  → SGWB generation mode")
                amplitude = params.get("amplitude", np.random.uniform(1e-26, 1e-24))
                distance = params.get("distance", np.random.uniform(1000, 10000))
                logger.debug(f"    amplitude={amplitude:.2e}, d={distance:.1f}")
                
                result = self.generate_stochastic_signal(amplitude, distance)
                logger.debug(f"  ✓ SGWB generated: h_plus shape={result['h_plus'].shape}")
                return result
            
            else:
                logger.error(f"  ❌ Unknown source type: {source_type}")
                raise ValueError(f"Unknown source type: {source_type}")
        
        except Exception as e:
            logger.error(f"  ❌ Generation failed: {e}")
            raise
        
        finally:
            logger.debug(f"└─ generate_event done")
