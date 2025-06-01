"""
Main entry point for the OpenAlpha_Evolve application.
Orchestrates autonomous development with intelligent task management.
"""
import asyncio
import logging
import sys
import os
import yaml
import argparse
                                               
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from autonomous_task_manager import AutonomousTaskManager
from task_manager.agent import TaskManagerAgent
from core.interfaces import TaskDefinition
from config import settings

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.LOG_FILE, mode="a")
    ]
)
logger = logging.getLogger(__name__)

def load_task_from_yaml(yaml_path: str) -> tuple[list, str, str, str, list]:
    """Load task configuration and test cases from a YAML file."""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            # Get task configuration
            task_id = data.get('task_id')
            task_description = data.get('task_description')
            function_name = data.get('function_name')
            allowed_imports = data.get('allowed_imports', [])
            
            # Convert test cases from YAML format to input_output_examples format
            input_output_examples = []
            for test_group in data.get('tests', []):
                for test_case in test_group.get('test_cases', []):
                    if 'output' in test_case:
                        input_output_examples.append({
                            'input': test_case['input'],
                            'output': test_case['output']
                        })
                    elif 'validation_func' in test_case:
                        input_output_examples.append({
                            'input': test_case['input'],
                            'validation_func': test_case['validation_func']
                        })
            
            return input_output_examples, task_id, task_description, function_name, allowed_imports
    except Exception as e:
        logger.error(f"Error loading task from YAML: {e}")
        return [], "", "", "", []

async def main():
    parser = argparse.ArgumentParser(description="Run OpenAlpha_Evolve with a specified YAML configuration file.")
    parser.add_argument("yaml_path", type=str, help="Path to the YAML configuration file")
    parser.add_argument("--autonomous", action="store_true", help="Enable full autonomous development mode")
    parser.add_argument("--disable-learning", action="store_true", help="Disable learning system")
    parser.add_argument("--disable-debugging", action="store_true", help="Disable auto-debugging")
    parser.add_argument("--disable-context", action="store_true", help="Disable context analysis")
    parser.add_argument("--fallback-mode", action="store_true", help="Use original task manager as fallback")
    args = parser.parse_args()
    yaml_path = args.yaml_path

    logger.info("Starting OpenAlpha_Evolve autonomous algorithmic evolution")
    logger.info(f"Configuration: Population Size={settings.POPULATION_SIZE}, Generations={settings.GENERATIONS}")
    
    if args.autonomous:
        logger.info("🤖 Autonomous development mode enabled")

    # Load task configuration and test cases from YAML file
    test_cases, task_id, task_description, function_name, allowed_imports = load_task_from_yaml(yaml_path)
    
    if not test_cases or not task_id or not task_description or not function_name:
        logger.error("Missing required task configuration in YAML file. Exiting.")
        return

    task = TaskDefinition(
        id=task_id,
        description=task_description,
        function_name_to_evolve=function_name,
        input_output_examples=test_cases,
        allowed_imports=allowed_imports
    )

    # Configure autonomous task manager
    autonomous_config = {
        'use_autonomous_pipeline': args.autonomous and not args.fallback_mode,
        'enable_learning': not args.disable_learning,
        'enable_auto_debugging': not args.disable_debugging,
        'enable_context_analysis': not args.disable_context
    }

    task_manager = AutonomousTaskManager(
        task_definition=task,
        config=autonomous_config
    )
    
    # Validate autonomous setup if enabled
    if args.autonomous:
        logger.info("Validating autonomous components...")
        validation_results = await task_manager.validate_autonomous_setup()
        
        failed_components = [comp for comp, valid in validation_results.items() if not valid]
        if failed_components:
            logger.warning(f"Some autonomous components failed validation: {failed_components}")
            logger.info("Continuing with available components...")
        else:
            logger.info("✅ All autonomous components validated successfully")

    # Execute task
    best_programs = await task_manager.execute()

    if best_programs:
        logger.info(f"🎉 Evolution completed successfully! Best program(s) found: {len(best_programs)}")
        for i, program in enumerate(best_programs):
            logger.info(f"Final Best Program {i+1} ID: {program.id}")
            logger.info(f"Final Best Program {i+1} Fitness: {program.fitness_scores}")
            logger.info(f"Final Best Program {i+1} Code:\n{program.code}")
            
            # Log autonomous features if used
            if hasattr(program, 'learning_tags') and program.learning_tags:
                logger.info(f"Learning tags applied: {program.learning_tags}")
            if hasattr(program, 'optimization_applied') and program.optimization_applied:
                logger.info(f"Optimizations applied: {program.optimization_applied}")
    else:
        logger.info("❌ Evolution completed, but no suitable programs were found.")
    
    # Display autonomous execution statistics
    if args.autonomous:
        try:
            stats = await task_manager.get_execution_statistics()
            logger.info("📊 Autonomous Execution Statistics:")
            for key, value in stats.items():
                if isinstance(value, dict):
                    logger.info(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        logger.info(f"    {sub_key}: {sub_value}")
                else:
                    logger.info(f"  {key}: {value}")
        except Exception as e:
            logger.warning(f"Could not retrieve execution statistics: {e}")

    logger.info("OpenAlpha_Evolve run finished.")

if __name__ == "__main__":
    asyncio.run(main())
