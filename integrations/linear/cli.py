"""
Linear Integration CLI Tool

Command-line interface for testing and managing the Linear integration.
"""

import asyncio
import argparse
import json
import sys
from typing import Dict, Any

from .agent import LinearIntegrationAgent
from .interfaces import LinearIssue
from config.settings import get_linear_config


async def cmd_status(args):
    """Get integration status"""
    config = get_linear_config()
    agent = LinearIntegrationAgent(config)
    
    if await agent.initialize():
        status = await agent.get_integration_status()
        print(json.dumps(status, indent=2, default=str))
    else:
        print("Failed to initialize integration")
        sys.exit(1)


async def cmd_test_webhook(args):
    """Test webhook processing"""
    config = get_linear_config()
    agent = LinearIntegrationAgent(config)
    
    if not await agent.initialize():
        print("Failed to initialize integration")
        sys.exit(1)
    
    # Create test webhook payload
    webhook_payload = {
        "action": "update",
        "type": "Issue", 
        "data": {
            "id": args.issue_id,
            "title": args.title or "Test Issue",
            "description": args.description or "Test issue description",
            "assignee": {
                "id": config.get("bot_user_id"),
                "name": "Test Bot",
                "email": config.get("bot_email")
            }
        }
    }
    
    # Process webhook
    success = await agent.handle_webhook(webhook_payload, args.signature or "test_signature")
    
    if success:
        print("✅ Webhook processed successfully")
    else:
        print("❌ Webhook processing failed")


async def cmd_create_task(args):
    """Create task from issue"""
    config = get_linear_config()
    agent = LinearIntegrationAgent(config)
    
    if not await agent.initialize():
        print("Failed to initialize integration")
        sys.exit(1)
    
    # Create issue object
    issue = LinearIssue(
        id=args.issue_id,
        title=args.title,
        description=args.description or "",
        assignee_id=config.get("bot_user_id"),
        assignee_name="CLI Bot"
    )
    
    # Create task
    task_id = await agent.workflow_automation.create_task_from_issue(issue)
    
    if task_id:
        print(f"✅ Task created: {task_id}")
    else:
        print("❌ Failed to create task")


async def cmd_sync_progress(args):
    """Sync progress to Linear"""
    config = get_linear_config()
    agent = LinearIntegrationAgent(config)
    
    if not await agent.initialize():
        print("Failed to initialize integration")
        sys.exit(1)
    
    # Create progress data
    progress = {
        "status": args.status,
        "generation": args.generation,
        "best_fitness": args.fitness,
        "details": {
            "tests_passed": args.tests_passed,
            "tests_total": args.tests_total
        }
    }
    
    # Sync progress
    success = await agent.workflow_automation.sync_progress(args.issue_id, progress)
    
    if success:
        print("✅ Progress synced successfully")
    else:
        print("❌ Failed to sync progress")


async def cmd_monitor(args):
    """Monitor assignments"""
    config = get_linear_config()
    agent = LinearIntegrationAgent(config)
    
    if not await agent.initialize():
        print("Failed to initialize integration")
        sys.exit(1)
    
    print("🔍 Starting assignment monitoring...")
    print("Press Ctrl+C to stop")
    
    try:
        # Start monitoring in background
        monitor_task = asyncio.create_task(agent.monitor_assignments())
        
        # Print status updates
        while True:
            await asyncio.sleep(10)
            status = await agent.get_integration_status()
            active_tasks = status.get('statistics', {}).get('workflows', {}).get('total_active_tasks', 0)
            print(f"📊 Active tasks: {active_tasks}")
            
    except KeyboardInterrupt:
        print("\\n⏹️ Stopping monitoring...")
        await agent.stop_monitoring()
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass


async def cmd_cleanup(args):
    """Clean up old data"""
    config = get_linear_config()
    agent = LinearIntegrationAgent(config)
    
    if not await agent.initialize():
        print("Failed to initialize integration")
        sys.exit(1)
    
    print("🧹 Cleaning up old data...")
    await agent.cleanup()
    print("✅ Cleanup completed")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="Linear Integration CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get integration status")
    
    # Test webhook command
    webhook_parser = subparsers.add_parser("test-webhook", help="Test webhook processing")
    webhook_parser.add_argument("--issue-id", required=True, help="Issue ID")
    webhook_parser.add_argument("--title", help="Issue title")
    webhook_parser.add_argument("--description", help="Issue description")
    webhook_parser.add_argument("--signature", help="Webhook signature")
    
    # Create task command
    task_parser = subparsers.add_parser("create-task", help="Create task from issue")
    task_parser.add_argument("--issue-id", required=True, help="Issue ID")
    task_parser.add_argument("--title", required=True, help="Issue title")
    task_parser.add_argument("--description", help="Issue description")
    
    # Sync progress command
    progress_parser = subparsers.add_parser("sync-progress", help="Sync progress to Linear")
    progress_parser.add_argument("--issue-id", required=True, help="Issue ID")
    progress_parser.add_argument("--status", required=True, help="Status")
    progress_parser.add_argument("--generation", type=int, default=0, help="Generation number")
    progress_parser.add_argument("--fitness", type=float, default=0.0, help="Fitness score")
    progress_parser.add_argument("--tests-passed", type=int, default=0, help="Tests passed")
    progress_parser.add_argument("--tests-total", type=int, default=0, help="Total tests")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor assignments")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old data")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Map commands to functions
    commands = {
        "status": cmd_status,
        "test-webhook": cmd_test_webhook,
        "create-task": cmd_create_task,
        "sync-progress": cmd_sync_progress,
        "monitor": cmd_monitor,
        "cleanup": cmd_cleanup
    }
    
    # Run command
    if args.command in commands:
        try:
            asyncio.run(commands[args.command](args))
        except KeyboardInterrupt:
            print("\\n⏹️ Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()

