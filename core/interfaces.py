from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import time

@dataclass
class Program:
    id: str
    code: str
    fitness_scores: Dict[str, float] = field(default_factory=dict)                                                 
    generation: int = 0
    parent_id: Optional[str] = None
    island_id: Optional[int] = None
    errors: List[str] = field(default_factory=list)
    status: str = "unevaluated"
    created_at: float = field(default_factory=lambda: time.time())  # Track program age
    task_id: Optional[str] = None
    # New fields for autonomous development
    context_analysis: Optional[Dict[str, Any]] = None
    debug_history: List[Dict[str, Any]] = field(default_factory=list)
    learning_tags: List[str] = field(default_factory=list)
    optimization_applied: List[str] = field(default_factory=list)

@dataclass
class TaskDefinition:
    id: str
    description: str                                              
    function_name_to_evolve: Optional[str] = None  # Can be used if evolving a single function
    target_file_path: Optional[str] = None # Path to the file containing code to be evolved
    evolve_blocks: Optional[List[Dict[str, Any]]] = None # Defines specific blocks within the target_file_path to evolve
                                                        # e.g., [{'block_id': 'optimizer_logic', 'start_marker': '# EVOLVE-BLOCK-START optimizer', 'end_marker': '# EVOLVE-BLOCK-END optimizer'}]
    input_output_examples: Optional[List[Dict[str, Any]]] = None                                                    
    evaluation_criteria: Optional[Dict[str, Any]] = None                                                            
    initial_code_prompt: Optional[str] = "Provide an initial Python solution for the following problem:"
    allowed_imports: Optional[List[str]] = None
    tests: Optional[List[Dict[str, Any]]] = None # List of test groups. Each group is a dict, can include 'name', 'description', 'level' (for cascade), and 'test_cases'.
    expert_knowledge: Optional[str] = None # Relevant expert knowledge, equations, or snippets
    # New fields for autonomous development
    complexity_level: Optional[str] = None  # 'low', 'medium', 'high'
    domain_tags: List[str] = field(default_factory=list)  # e.g., ['algorithms', 'data_structures']
    success_criteria: Optional[Dict[str, Any]] = None
    context_requirements: Optional[Dict[str, Any]] = None

class BaseAgent(ABC):
    """Base class for all agents."""
    @abstractmethod
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Main execution method for an agent."""
        pass

class TaskManagerInterface(BaseAgent):
    @abstractmethod
    async def manage_evolutionary_cycle(self):
        pass

class PromptDesignerInterface(BaseAgent):
    @abstractmethod
    def design_initial_prompt(self, task: TaskDefinition) -> str:
        pass

    @abstractmethod
    def design_mutation_prompt(self, task: TaskDefinition, parent_program: Program, evaluation_feedback: Optional[Dict] = None) -> str:
        pass

    @abstractmethod
    def design_bug_fix_prompt(self, task: TaskDefinition, program: Program, error_info: Dict) -> str:
        pass

class CodeGeneratorInterface(BaseAgent):
    @abstractmethod
    async def generate_code(self, prompt: str, model_name: Optional[str] = None, temperature: Optional[float] = 0.7, output_format: str = "code") -> str:
        pass

class EvaluatorAgentInterface(BaseAgent):
    @abstractmethod
    async def evaluate_program(self, program: Program, task: TaskDefinition) -> Program:
        pass

class DatabaseAgentInterface(BaseAgent):
    @abstractmethod
    async def save_program(self, program: Program):
        pass

    @abstractmethod
    async def get_program(self, program_id: str) -> Optional[Program]:
        pass

    @abstractmethod
    async def get_best_programs(self, task_id: str, limit: int = 10, objective: Optional[str] = None) -> List[Program]:
        pass
    
    @abstractmethod
    async def get_programs_for_next_generation(self, task_id: str, generation_size: int) -> List[Program]:
        pass

class SelectionControllerInterface(BaseAgent):
    @abstractmethod
    def select_parents(self, evaluated_programs: List[Program], num_parents: int) -> List[Program]:
        pass

    @abstractmethod
    def select_survivors(self, current_population: List[Program], offspring_population: List[Program], population_size: int) -> List[Program]:
        pass

    @abstractmethod
    def initialize_islands(self, initial_programs: List[Program]) -> None:
        pass

class RLFineTunerInterface(BaseAgent):
    @abstractmethod
    async def update_policy(self, experience_data: List[Dict]):
        pass

class MonitoringAgentInterface(BaseAgent):
    @abstractmethod
    async def log_metrics(self, metrics: Dict):
        pass

    @abstractmethod
    async def report_status(self):
        pass

# New interfaces for autonomous development

class ContextAnalysisInterface(BaseAgent):
    @abstractmethod
    async def analyze_codebase(self, target_path: Optional[str] = None) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_context_for_element(self, element_name: str) -> Optional[Dict[str, Any]]:
        pass

class ErrorAnalysisInterface(BaseAgent):
    @abstractmethod
    async def classify_error(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def analyze_root_cause(self, error_classification: Dict[str, Any]) -> Dict[str, Any]:
        pass

class AutoDebuggingInterface(BaseAgent):
    @abstractmethod
    async def debug_program(self, program: Program, error_info: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def suggest_fixes(self, error_classification: Dict[str, Any]) -> List[str]:
        pass

class LearningInterface(BaseAgent):
    @abstractmethod
    async def learn_from_data(self, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_recommendations(self, context: Dict[str, Any]) -> List[str]:
        pass
    
    @abstractmethod
    def get_patterns(self, pattern_type: Optional[str] = None) -> List[Dict[str, Any]]:
        pass

class PipelineOrchestratorInterface(BaseAgent):
    @abstractmethod
    async def run_pipeline(self, task_definition: TaskDefinition) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def add_stage_callback(self, stage: str, callback: Any):
        pass
    
    @abstractmethod
    def set_progress_callback(self, callback: Any):
        pass

class ValidationEngineInterface(BaseAgent):
    @abstractmethod
    async def validate_solution(self, program: Program, task_definition: TaskDefinition) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def validate_pipeline_stage(self, stage_name: str, stage_data: Dict[str, Any]) -> bool:
        pass
