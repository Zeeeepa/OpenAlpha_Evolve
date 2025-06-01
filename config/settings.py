import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
FLASH_API_KEY = os.getenv("FLASH_API_KEY")
FLASH_BASE_URL = os.getenv("FLASH_BASE_URL", None)
FLASH_MODEL = os.getenv("FLASH_MODEL")

# PRO_API_KEY = os.getenv("PRO_API_KEY")
# PRO_BASE_URL = os.getenv("PRO_BASE_URL", None)
# PRO_MODEL = os.getenv("PRO_MODEL")

EVALUATION_API_KEY = os.getenv("EVALUATION_API_KEY")
EVALUATION_BASE_URL = os.getenv("EVALUATION_BASE_URL", None)
EVALUATION_MODEL = os.getenv("EVALUATION_MODEL")

# LiteLLM Configuration
LITELLM_DEFAULT_MODEL = os.getenv("LITELLM_DEFAULT_MODEL", "gpt-3.5-turbo")
LITELLM_DEFAULT_BASE_URL = os.getenv("LITELLM_DEFAULT_BASE_URL", None)
LITELLM_MAX_TOKENS = os.getenv("LITELLM_MAX_TOKENS")
LITELLM_TEMPERATURE = os.getenv("LITELLM_TEMPERATURE")
LITELLM_TOP_P = os.getenv("LITELLM_TOP_P")
LITELLM_TOP_K = os.getenv("LITELLM_TOP_K")

# Specific model names for strategic use (can be same as LITELLM_DEFAULT_MODEL if only one is used)
LLM_PRIMARY_MODEL = os.getenv("LLM_PRIMARY_MODEL", LITELLM_DEFAULT_MODEL)
LLM_SECONDARY_MODEL = os.getenv("LLM_SECONDARY_MODEL", FLASH_MODEL if FLASH_MODEL else LLM_PRIMARY_MODEL)

# Linear Integration Configuration
LINEAR_API_KEY = os.getenv("LINEAR_API_KEY")
LINEAR_WEBHOOK_SECRET = os.getenv("LINEAR_WEBHOOK_SECRET")
LINEAR_BOT_USER_ID = os.getenv("LINEAR_BOT_USER_ID")
LINEAR_BOT_EMAIL = os.getenv("LINEAR_BOT_EMAIL")
LINEAR_BOT_NAMES = os.getenv("LINEAR_BOT_NAMES", "codegen,openalpha,bot").split(",")

# Linear Integration Settings
LINEAR_ENABLED = os.getenv("LINEAR_ENABLED", "False").lower() == "true"
LINEAR_AUTO_ASSIGN_LABELS = os.getenv("LINEAR_AUTO_ASSIGN_LABELS", "ai,automation,codegen").split(",")
LINEAR_AUTO_ASSIGN_KEYWORDS = os.getenv("LINEAR_AUTO_ASSIGN_KEYWORDS", "generate,evolve,optimize,automate").split(",")
LINEAR_MAX_ASSIGNMENTS_PER_HOUR = int(os.getenv("LINEAR_MAX_ASSIGNMENTS_PER_HOUR", "10"))
LINEAR_AUTO_START_TASKS = os.getenv("LINEAR_AUTO_START_TASKS", "True").lower() == "true"
LINEAR_AUTO_UPDATE_STATUS = os.getenv("LINEAR_AUTO_UPDATE_STATUS", "True").lower() == "true"
LINEAR_MONITORING_INTERVAL = int(os.getenv("LINEAR_MONITORING_INTERVAL", "60"))

# Linear API Configuration
LINEAR_API_TIMEOUT = int(os.getenv("LINEAR_API_TIMEOUT", "30"))
LINEAR_API_MAX_RETRIES = int(os.getenv("LINEAR_API_MAX_RETRIES", "3"))
LINEAR_RATE_LIMIT_REQUESTS = int(os.getenv("LINEAR_RATE_LIMIT_REQUESTS", "100"))
LINEAR_RATE_LIMIT_WINDOW = int(os.getenv("LINEAR_RATE_LIMIT_WINDOW", "60"))
LINEAR_CACHE_TTL = int(os.getenv("LINEAR_CACHE_TTL", "300"))

# Linear Webhook Configuration
LINEAR_WEBHOOK_MAX_RETRIES = int(os.getenv("LINEAR_WEBHOOK_MAX_RETRIES", "3"))
LINEAR_WEBHOOK_RETRY_DELAY = int(os.getenv("LINEAR_WEBHOOK_RETRY_DELAY", "5"))
LINEAR_WEBHOOK_MAX_PAYLOAD_SIZE = int(os.getenv("LINEAR_WEBHOOK_MAX_PAYLOAD_SIZE", "1048576"))  # 1MB

# Linear Event Management
LINEAR_EVENT_QUEUE_SIZE = int(os.getenv("LINEAR_EVENT_QUEUE_SIZE", "1000"))
LINEAR_EVENT_BATCH_SIZE = int(os.getenv("LINEAR_EVENT_BATCH_SIZE", "10"))
LINEAR_EVENT_PROCESSING_INTERVAL = int(os.getenv("LINEAR_EVENT_PROCESSING_INTERVAL", "5"))
LINEAR_EVENT_RETRY_INTERVAL = int(os.getenv("LINEAR_EVENT_RETRY_INTERVAL", "60"))
LINEAR_EVENT_PERSISTENCE_ENABLED = os.getenv("LINEAR_EVENT_PERSISTENCE_ENABLED", "True").lower() == "true"
LINEAR_EVENT_PERSISTENCE_FILE = os.getenv("LINEAR_EVENT_PERSISTENCE_FILE", "linear_events.json")

# Evolutionary Algorithm Settings
POPULATION_SIZE = 5
GENERATIONS = 2
# Threshold for switching to bug-fix prompt
# If a program has errors and its correctness score is below this, a bug-fix prompt will be used.
BUG_FIX_CORRECTNESS_THRESHOLD = float(os.getenv("BUG_FIX_CORRECTNESS_THRESHOLD", "0.1"))
# Threshold for using the primary (potentially more powerful/expensive) LLM for mutation
HIGH_FITNESS_THRESHOLD_FOR_PRIMARY_LLM = float(os.getenv("HIGH_FITNESS_THRESHOLD_FOR_PRIMARY_LLM", "0.8"))
ELITISM_COUNT = 1
MUTATION_RATE = 0.7
CROSSOVER_RATE = 0.2

# Island Model Settings
NUM_ISLANDS = 4  # Number of subpopulations
MIGRATION_INTERVAL = 4  # Number of generations between migrations
ISLAND_POPULATION_SIZE = POPULATION_SIZE // NUM_ISLANDS  # Programs per island
MIN_ISLAND_SIZE = 2  # Minimum number of programs per island
MIGRATION_RATE = 0.2  # Rate at which programs migrate between islands

# Debug Settings
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
EVALUATION_TIMEOUT_SECONDS = 800

# Docker Execution Settings
DOCKER_IMAGE_NAME = os.getenv("DOCKER_IMAGE_NAME", "code-evaluator:latest")
DOCKER_NETWORK_DISABLED = os.getenv("DOCKER_NETWORK_DISABLED", "True").lower() == "true"

DATABASE_TYPE = "json"
DATABASE_PATH = "program_database.json"

# Logging Configuration
LOG_LEVEL = "DEBUG" if DEBUG else "INFO"
LOG_FILE = "alpha_evolve.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

API_MAX_RETRIES = 5
API_RETRY_DELAY_SECONDS = 10

RL_TRAINING_INTERVAL_GENERATIONS = 50
RL_MODEL_PATH = "rl_finetuner_model.pth"

MONITORING_DASHBOARD_URL = "http://localhost:8080"

def get_setting(key, default=None):
    """
    Retrieves a setting value.
    For LLM models, it specifically checks if the primary choice is available,
    otherwise falls back to a secondary/default if defined.
    """
    return globals().get(key, default)

def get_llm_model(model_type="default"):
    if model_type == "default":
        return LITELLM_DEFAULT_MODEL
    elif model_type == "flash":
        # Assuming FLASH_MODEL might still be a specific, different model.
        # If FLASH_MODEL is also meant to be covered by litellm's general handling,
        # this could also return LITELLM_DEFAULT_MODEL or a specific flash model string.
        # For now, keep FLASH_MODEL if it's distinct.
        return FLASH_MODEL if FLASH_MODEL else LITELLM_DEFAULT_MODEL # Return default if FLASH_MODEL is not set
    # Fallback for any other model_type not explicitly handled
    return LITELLM_DEFAULT_MODEL

def get_linear_config():
    """
    Get Linear integration configuration
    """
    return {
        "api_key": LINEAR_API_KEY,
        "webhook_secret": LINEAR_WEBHOOK_SECRET,
        "bot_user_id": LINEAR_BOT_USER_ID,
        "bot_email": LINEAR_BOT_EMAIL,
        "bot_names": LINEAR_BOT_NAMES,
        "enabled": LINEAR_ENABLED,
        "auto_assign_labels": LINEAR_AUTO_ASSIGN_LABELS,
        "auto_assign_keywords": LINEAR_AUTO_ASSIGN_KEYWORDS,
        "max_assignments_per_hour": LINEAR_MAX_ASSIGNMENTS_PER_HOUR,
        "auto_start_tasks": LINEAR_AUTO_START_TASKS,
        "auto_update_status": LINEAR_AUTO_UPDATE_STATUS,
        "monitoring_interval": LINEAR_MONITORING_INTERVAL,
        "timeout": LINEAR_API_TIMEOUT,
        "max_retries": LINEAR_API_MAX_RETRIES,
        "rate_limit_requests": LINEAR_RATE_LIMIT_REQUESTS,
        "rate_limit_window": LINEAR_RATE_LIMIT_WINDOW,
        "cache_ttl": LINEAR_CACHE_TTL,
        "webhook_max_retries": LINEAR_WEBHOOK_MAX_RETRIES,
        "webhook_retry_delay": LINEAR_WEBHOOK_RETRY_DELAY,
        "webhook_max_payload_size": LINEAR_WEBHOOK_MAX_PAYLOAD_SIZE,
        "event_queue_size": LINEAR_EVENT_QUEUE_SIZE,
        "event_batch_size": LINEAR_EVENT_BATCH_SIZE,
        "event_processing_interval": LINEAR_EVENT_PROCESSING_INTERVAL,
        "event_retry_interval": LINEAR_EVENT_RETRY_INTERVAL,
        "event_persistence_enabled": LINEAR_EVENT_PERSISTENCE_ENABLED,
        "event_persistence_file": LINEAR_EVENT_PERSISTENCE_FILE
    }
