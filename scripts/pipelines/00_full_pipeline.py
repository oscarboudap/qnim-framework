#!/usr/bin/env python3
"""
QNIM Master Orchestrator: Full Pipeline Execution (Blocks 1→2→3)

Executes the complete TFM workflow:
  Block 1: Generate synthetic GW with quantum gravity injection
  Block 2: Train Quantum VQC on synthetic dataset
  Block 3: Exhaustive statistical validation with Monte Carlo sweeps

This orchestrator manages data flow, error handling, and unified reporting.
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
from typing import Dict, Any

# CLI Framework
from src.cli import (
    ScriptConfig,
    ScriptContainer,
    get_script_logger,
    create_execution_context,
    ScriptException,
    ScriptExecutionException,
)


def main() -> None:
    """
    Master pipeline orchestrator.
    
    Executes Blocks 1→2→3 with unified error handling and reporting.
    """
    logger = get_script_logger("full_pipeline")
    context = create_execution_context("00_full_pipeline.py")
    
    try:
        # ====================================================================
        # INITIALIZATION
        # ====================================================================
        logger.step("INIT", "Starting QNIM Master Pipeline", total_blocks=3)
        
        config = ScriptConfig.from_env()
        container = ScriptContainer(config)
        
        pipeline_results = {
            "block_1_synthesis": None,
            "block_2_training": None,
            "block_3_validation": None,
            "execution_times": {},
        }
        
        # ====================================================================
        # BLOCK 1: GENERATE SYNTHETIC GW
        # ====================================================================
        logger.step("PIPELINE", "Executing Block 1: Synthetic GW Generation")
        block_1_start = time.time()
        
        try:
            from scripts.pipelines import generate_synthetic_gw
            result_1 = generate_synthetic_gw.main()
            pipeline_results["block_1_synthesis"] = result_1
            pipeline_results["execution_times"]["block_1"] = time.time() - block_1_start
            
            logger.metric(
                "block_1_duration",
                pipeline_results["execution_times"]["block_1"],
                unit="seconds"
            )
            logger.info("✅ Block 1 completed successfully")
        
        except Exception as e:
            logger.error(f"❌ Block 1 failed: {e}", exc_info=True)
            raise ScriptExecutionException(
                "Block 1 (Synthetic GW Generation) failed",
                context={"error": str(e)}
            )
        
        # ====================================================================
        # BLOCK 2: TRAIN VQC
        # ====================================================================
        logger.step("PIPELINE", "Executing Block 2: VQC Model Training")
        block_2_start = time.time()
        
        try:
            from scripts.pipelines import train_vqc_model
            
            # Pass Block 1 results as input
            result_2 = train_vqc_model.main(
                synthetic_data_path=result_1.get("output_path") if result_1 else None
            )
            pipeline_results["block_2_training"] = result_2
            pipeline_results["execution_times"]["block_2"] = time.time() - block_2_start
            
            logger.metric(
                "block_2_duration",
                pipeline_results["execution_times"]["block_2"],
                unit="seconds"
            )
            logger.info("✅ Block 2 completed successfully")
        
        except Exception as e:
            logger.error(f"❌ Block 2 failed: {e}", exc_info=True)
            raise ScriptExecutionException(
                "Block 2 (VQC Training) failed",
                context={"error": str(e)}
            )
        
        # ====================================================================
        # BLOCK 3: EXHAUSTIVE VALIDATION
        # ====================================================================
        logger.step("PIPELINE", "Executing Block 3: Statistical Validation")
        block_3_start = time.time()
        
        try:
            from scripts.pipelines import validate_exhaustive
            
            # Pass Block 2 results as input
            result_3 = validate_exhaustive.main(
                trained_weights_path=result_2.get("weights_path") if result_2 else None,
                trained_model=result_2.get("model") if result_2 else None
            )
            pipeline_results["block_3_validation"] = result_3
            pipeline_results["execution_times"]["block_3"] = time.time() - block_3_start
            
            logger.metric(
                "block_3_duration",
                pipeline_results["execution_times"]["block_3"],
                unit="seconds"
            )
            logger.info("✅ Block 3 completed successfully")
        
        except Exception as e:
            logger.error(f"❌ Block 3 failed: {e}", exc_info=True)
            raise ScriptExecutionException(
                "Block 3 (Statistical Validation) failed",
                context={"error": str(e)}
            )
        
        # ====================================================================
        # UNIFIED REPORTING
        # ====================================================================
        logger.step(
            "SUMMARY",
            "Pipeline Execution Complete",
            total_time_seconds=sum(pipeline_results["execution_times"].values()),
            block_count=3
        )
        
        # Report execution times
        for block_name, duration in pipeline_results["execution_times"].items():
            logger.metric(block_name, duration, unit="seconds")
        
        # Report Block 3 final statistics (if available)
        if result_3 and "final_significance" in result_3:
            significance = result_3["final_significance"]
            logger.metric("significance_level", significance.get("sigma", 0), unit="σ")
        
        logger.info("✅ QNIM Master Pipeline completed successfully")
        return None
    
    except ScriptExecutionException as e:
        logger.error(f"❌ Pipeline execution failed: {e}", extra=context)
        raise SystemExit(1)
    
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", extra=context, exc_info=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
