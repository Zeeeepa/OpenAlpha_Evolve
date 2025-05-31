# Linear API Integration System

A comprehensive Linear API integration system for OpenAlpha_Evolve that provides real-time webhook processing, assignment detection, and automated workflow management.

## 🎯 Overview

This integration enables OpenAlpha_Evolve to automatically process Linear issues assigned to the bot, creating evolutionary code generation tasks and providing real-time progress updates back to Linear.

## 🏗️ Architecture

### Core Components

1. **Linear GraphQL Client** (`client/`)
   - Authenticated GraphQL API communication
   - Rate limiting and response caching
   - Comprehensive query and mutation support
   - Error handling and retry mechanisms

2. **Webhook Processing System** (`webhook/`)
   - Real-time webhook event processing
   - Signature validation and payload sanitization
   - Event routing and handler management
   - Retry mechanisms for failed events

3. **Assignment Detection** (`assignment/`)
   - Bot assignment tracking and detection
   - Auto-assignment rule processing
   - Assignment event generation
   - Rate limiting for assignment processing

4. **Workflow Automation** (`workflow/`)
   - Task creation from Linear issues
   - Status synchronization with Linear
   - Progress tracking and reporting
   - Automated comment generation

5. **Event Management** (`events/`)
   - Asynchronous event queue processing
   - Event persistence and replay
   - Background processing and retry logic
   - Event tracking and statistics

## 🚀 Features

### Real-time Integration
- **Webhook Processing**: Instant processing of Linear webhook events
- **Assignment Detection**: Automatic detection when bot is assigned to issues
- **Status Synchronization**: Real-time status updates between systems
- **Progress Reporting**: Live progress updates during code evolution

### Intelligent Automation
- **Auto-assignment**: Automatic assignment based on labels and keywords
- **Task Creation**: Intelligent parsing of Linear issues into evolution tasks
- **Test Case Extraction**: Automatic extraction of test cases from issue descriptions
- **Comment Generation**: Automated progress comments and status updates

### Robust Infrastructure
- **Rate Limiting**: Compliance with Linear API rate limits
- **Caching**: Response caching for improved performance
- **Error Handling**: Comprehensive error handling and recovery
- **Event Persistence**: Persistent event storage for reliability

## 📋 Configuration

### Environment Variables

```bash
# Linear API Configuration
LINEAR_API_KEY=your_linear_api_key
LINEAR_WEBHOOK_SECRET=your_webhook_secret
LINEAR_BOT_USER_ID=your_bot_user_id
LINEAR_BOT_EMAIL=your_bot_email

# Integration Settings
LINEAR_ENABLED=true
LINEAR_AUTO_ASSIGN_LABELS=ai,automation,codegen
LINEAR_AUTO_ASSIGN_KEYWORDS=generate,evolve,optimize,automate
LINEAR_MAX_ASSIGNMENTS_PER_HOUR=10
LINEAR_AUTO_START_TASKS=true
LINEAR_AUTO_UPDATE_STATUS=true
LINEAR_MONITORING_INTERVAL=60

# API Configuration
LINEAR_API_TIMEOUT=30
LINEAR_API_MAX_RETRIES=3
LINEAR_RATE_LIMIT_REQUESTS=100
LINEAR_RATE_LIMIT_WINDOW=60
LINEAR_CACHE_TTL=300

# Webhook Configuration
LINEAR_WEBHOOK_MAX_RETRIES=3
LINEAR_WEBHOOK_RETRY_DELAY=5
LINEAR_WEBHOOK_MAX_PAYLOAD_SIZE=1048576

# Event Management
LINEAR_EVENT_QUEUE_SIZE=1000
LINEAR_EVENT_BATCH_SIZE=10
LINEAR_EVENT_PROCESSING_INTERVAL=5
LINEAR_EVENT_RETRY_INTERVAL=60
LINEAR_EVENT_PERSISTENCE_ENABLED=true
LINEAR_EVENT_PERSISTENCE_FILE=linear_events.json
```

### Configuration Loading

```python
from config.settings import get_linear_config

config = get_linear_config()
agent = LinearIntegrationAgent(config)
```

## 🔧 Usage

### Basic Setup

```python
from integrations.linear import LinearIntegrationAgent
from config.settings import get_linear_config

# Initialize agent
config = get_linear_config()
agent = LinearIntegrationAgent(config)

# Initialize the integration
await agent.initialize()

# Start monitoring for assignments
await agent.monitor_assignments()
```

### Webhook Handling

```python
# Handle incoming webhook
webhook_payload = request.json
webhook_signature = request.headers.get('Linear-Signature')

success = await agent.handle_webhook(webhook_payload, webhook_signature)
```

### Manual Task Creation

```python
from integrations.linear.interfaces import LinearIssue

# Create issue object
issue = LinearIssue(
    id="issue_id",
    title="Implement fibonacci function",
    description="Create a function that calculates fibonacci numbers",
    assignee_id="bot_user_id"
)

# Create task
task_id = await agent.workflow_automation.create_task_from_issue(issue)
```

## 📊 Monitoring and Status

### Integration Status

```python
# Get comprehensive status
status = await agent.get_integration_status()

print(f"Initialized: {status['initialized']}")
print(f"Monitoring Active: {status['monitoring_active']}")
print(f"Last Sync: {status['last_sync']}")
```

### Component Statistics

```python
# Webhook processing stats
webhook_stats = await agent.webhook_processor.get_processing_stats()

# Assignment detection stats  
assignment_stats = await agent.assignment_detector.get_assignment_stats()

# Workflow automation stats
workflow_stats = await agent.workflow_automation.get_workflow_stats()
```

## 🧪 Testing

### Running Tests

```bash
# Run integration tests
python -m integrations.linear.tests.test_integration

# Run specific component tests
python -m integrations.linear.tests.test_client
python -m integrations.linear.tests.test_webhook
python -m integrations.linear.tests.test_assignment
python -m integrations.linear.tests.test_workflow
```

### Test Coverage

The test suite includes:
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Mock Testing**: API interaction testing
- **Error Handling**: Failure scenario testing
- **Performance Tests**: Rate limiting and caching tests

## 🔄 Workflow Examples

### Issue Assignment Workflow

1. **Issue Created/Updated**: Linear webhook triggered
2. **Assignment Detection**: Bot assignment detected
3. **Task Creation**: OpenAlpha_Evolve task created from issue
4. **Evolution Process**: Code evolution begins
5. **Progress Updates**: Real-time progress comments in Linear
6. **Completion**: Final results posted to Linear issue

### Auto-assignment Workflow

1. **Issue Created**: New issue with specific labels/keywords
2. **Auto-assignment Check**: Labels and keywords evaluated
3. **Bot Assignment**: Bot automatically assigned to issue
4. **Workflow Trigger**: Standard assignment workflow begins

## 🛠️ API Reference

### LinearIntegrationAgent

Main orchestrator class for Linear integration.

```python
class LinearIntegrationAgent:
    async def initialize() -> bool
    async def handle_webhook(payload: Dict, signature: str) -> bool
    async def monitor_assignments() -> None
    async def sync_with_linear() -> bool
    async def get_integration_status() -> Dict[str, Any]
    async def cleanup() -> None
```

### LinearGraphQLClient

GraphQL client for Linear API communication.

```python
class LinearGraphQLClient:
    async def authenticate(api_key: str) -> bool
    async def execute_query(query: str, variables: Dict = None) -> Dict
    async def get_issue(issue_id: str) -> Optional[LinearIssue]
    async def update_issue(issue_id: str, updates: Dict) -> bool
    async def create_comment(issue_id: str, body: str) -> Optional[str]
```

### WebhookProcessor

Webhook event processing and validation.

```python
class WebhookProcessor:
    async def process_webhook(payload: bytes, signature: str) -> bool
    async def process_event(event: WebhookEvent) -> bool
    def register_handler(event_type: LinearEventType, handler: Callable) -> None
```

## 🔒 Security

### Webhook Validation
- **Signature Verification**: HMAC-SHA256 signature validation
- **Timestamp Validation**: Replay attack prevention
- **Payload Sanitization**: Input sanitization and validation
- **Size Limits**: Payload size restrictions

### API Security
- **Authentication**: Bearer token authentication
- **Rate Limiting**: API rate limit compliance
- **Error Handling**: Secure error message handling
- **Input Validation**: Comprehensive input validation

## 🚨 Error Handling

### Retry Mechanisms
- **Webhook Processing**: Automatic retry for failed webhook events
- **API Requests**: Exponential backoff for API failures
- **Event Processing**: Queue-based retry for event processing
- **Assignment Detection**: Graceful handling of detection failures

### Logging and Monitoring
- **Structured Logging**: Comprehensive logging throughout the system
- **Error Tracking**: Detailed error tracking and reporting
- **Performance Metrics**: Performance monitoring and statistics
- **Health Checks**: System health monitoring

## 📈 Performance

### Optimization Features
- **Response Caching**: Intelligent caching of API responses
- **Rate Limiting**: Efficient rate limit management
- **Batch Processing**: Batch processing for improved throughput
- **Async Processing**: Asynchronous processing throughout

### Scalability
- **Event Queue**: Scalable event processing queue
- **Background Processing**: Background task processing
- **Resource Management**: Efficient resource utilization
- **Cleanup Operations**: Automatic cleanup of old data

## 🤝 Contributing

### Development Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your Linear API credentials
   ```

3. **Run Tests**:
   ```bash
   python -m integrations.linear.tests.test_integration
   ```

### Code Style
- Follow PEP 8 style guidelines
- Use type hints throughout
- Include comprehensive docstrings
- Add unit tests for new features

## 📝 License

This Linear integration system is part of the OpenAlpha_Evolve project and follows the same licensing terms.

## 🆘 Support

For issues, questions, or contributions:
1. Check the test suite for examples
2. Review the configuration documentation
3. Examine the error logs for debugging information
4. Create an issue with detailed reproduction steps

---

**Note**: This integration requires valid Linear API credentials and proper webhook configuration. Ensure all environment variables are properly set before initializing the system.

