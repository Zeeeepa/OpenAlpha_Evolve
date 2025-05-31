"""
Progress Sync

Synchronizes OpenAlpha_Evolve task progress back to Linear issues.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from ..interfaces import LinearGraphQLClientInterface

logger = logging.getLogger(__name__)


class ProgressSync:
    """Synchronizes progress between OpenAlpha_Evolve and Linear"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.linear_client: Optional[LinearGraphQLClientInterface] = None
        
        # Progress sync configuration
        self.sync_interval = self.config.get("sync_interval", 30)  # seconds
        self.include_generation_details = self.config.get("include_generation_details", True)
        self.include_fitness_scores = self.config.get("include_fitness_scores", True)
        self.max_comment_length = self.config.get("max_comment_length", 2000)
        
        logger.info("ProgressSync initialized")
    
    def set_linear_client(self, client: LinearGraphQLClientInterface) -> None:
        """Set Linear GraphQL client"""
        self.linear_client = client
    
    async def sync_progress(self, issue_id: str, progress: Dict[str, Any]) -> bool:
        """Sync progress to Linear issue"""
        try:
            if not self.linear_client:
                logger.error("Linear client not set")
                return False
            
            # Generate progress comment
            comment_body = self._generate_progress_comment(progress)
            
            # Post comment to Linear issue
            comment_id = await self.linear_client.create_comment(issue_id, comment_body)
            
            if comment_id:
                logger.info(f"Posted progress update {comment_id} to issue {issue_id}")
                return True
            else:
                logger.error(f"Failed to post progress update to issue {issue_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error syncing progress for issue {issue_id}: {e}")
            return False
    
    def _generate_progress_comment(self, progress: Dict[str, Any]) -> str:
        """Generate progress comment for Linear"""
        try:
            status = progress.get("status", "unknown")
            last_update = progress.get("last_update", datetime.now().isoformat())
            generation = progress.get("generation", 0)
            best_fitness = progress.get("best_fitness", 0.0)
            
            # Start building comment
            comment_parts = []
            
            # Status header with emoji
            status_emoji = {
                "created": "🆕",
                "started": "🚀", 
                "running": "⚡",
                "evolving": "🧬",
                "testing": "🧪",
                "completed": "✅",
                "failed": "❌",
                "stopped": "⏹️"
            }
            
            emoji = status_emoji.get(status, "🔄")
            comment_parts.append(f"{emoji} **Evolution Progress Update**")
            
            # Status and timing
            comment_parts.append(f"**Status**: {status.title()}")
            comment_parts.append(f"**Last Update**: {last_update}")
            
            # Generation details if available
            if self.include_generation_details and generation > 0:
                comment_parts.append(f"**Generation**: {generation}")
                
                if self.include_fitness_scores and best_fitness > 0:
                    comment_parts.append(f"**Best Fitness**: {best_fitness:.4f}")
            
            # Additional progress details
            if "details" in progress:
                details = progress["details"]
                if isinstance(details, dict):
                    if "tests_passed" in details:
                        comment_parts.append(f"**Tests Passed**: {details['tests_passed']}")
                    if "tests_total" in details:
                        comment_parts.append(f"**Total Tests**: {details['tests_total']}")
                    if "success_rate" in details:
                        success_rate = details["success_rate"] * 100
                        comment_parts.append(f"**Success Rate**: {success_rate:.1f}%")
            
            # Error information if failed
            if status == "failed" and "error" in progress:
                error_msg = progress["error"]
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                comment_parts.append(f"**Error**: {error_msg}")
            
            # Completion information
            if status == "completed":
                if "solution" in progress:
                    solution_info = progress["solution"]
                    if isinstance(solution_info, dict):
                        if "function_name" in solution_info:
                            comment_parts.append(f"**Solution Function**: `{solution_info['function_name']}`")
                        if "lines_of_code" in solution_info:
                            comment_parts.append(f"**Lines of Code**: {solution_info['lines_of_code']}")
                
                comment_parts.append("\\n🎉 **Code evolution completed successfully!**")
                comment_parts.append("The evolved solution has been generated and tested.")
            
            # Join all parts
            comment_body = "\\n".join(comment_parts)
            
            # Truncate if too long
            if len(comment_body) > self.max_comment_length:
                comment_body = comment_body[:self.max_comment_length - 3] + "..."
            
            return comment_body
            
        except Exception as e:
            logger.error(f"Error generating progress comment: {e}")
            return f"🔄 **Evolution Progress Update**\\n\\nStatus: {progress.get('status', 'unknown')}"
    
    async def sync_completion(self, issue_id: str, result: Dict[str, Any]) -> bool:
        """Sync completion results to Linear issue"""
        try:
            if not self.linear_client:
                logger.error("Linear client not set")
                return False
            
            # Generate completion comment
            comment_body = self._generate_completion_comment(result)
            
            # Post comment to Linear issue
            comment_id = await self.linear_client.create_comment(issue_id, comment_body)
            
            if comment_id:
                logger.info(f"Posted completion update {comment_id} to issue {issue_id}")
                
                # Update issue status if successful
                if result.get("success", False):
                    # TODO: Update issue state to "Done" or similar
                    pass
                
                return True
            else:
                logger.error(f"Failed to post completion update to issue {issue_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error syncing completion for issue {issue_id}: {e}")
            return False
    
    def _generate_completion_comment(self, result: Dict[str, Any]) -> str:
        """Generate completion comment for Linear"""
        try:
            success = result.get("success", False)
            
            if success:
                comment_parts = [
                    "🎉 **Code Evolution Completed Successfully!**",
                    "",
                    "The evolutionary algorithm has successfully generated a solution for this issue."
                ]
                
                # Add solution details
                if "solution" in result:
                    solution = result["solution"]
                    if "code" in solution:
                        code = solution["code"]
                        if len(code) > 500:
                            code = code[:500] + "\\n... (truncated)"
                        
                        comment_parts.extend([
                            "",
                            "**Generated Solution:**",
                            "```python",
                            code,
                            "```"
                        ])
                    
                    if "fitness_score" in solution:
                        comment_parts.append(f"**Final Fitness Score**: {solution['fitness_score']:.4f}")
                    
                    if "test_results" in solution:
                        test_results = solution["test_results"]
                        if "passed" in test_results and "total" in test_results:
                            passed = test_results["passed"]
                            total = test_results["total"]
                            comment_parts.append(f"**Tests Passed**: {passed}/{total}")
                
                # Add evolution statistics
                if "statistics" in result:
                    stats = result["statistics"]
                    comment_parts.append("")
                    comment_parts.append("**Evolution Statistics:**")
                    
                    if "generations" in stats:
                        comment_parts.append(f"- Generations: {stats['generations']}")
                    if "total_evaluations" in stats:
                        comment_parts.append(f"- Total Evaluations: {stats['total_evaluations']}")
                    if "execution_time" in stats:
                        comment_parts.append(f"- Execution Time: {stats['execution_time']:.2f}s")
                
            else:
                comment_parts = [
                    "❌ **Code Evolution Failed**",
                    "",
                    "The evolutionary algorithm was unable to generate a successful solution."
                ]
                
                # Add error details
                if "error" in result:
                    error_msg = result["error"]
                    if len(error_msg) > 300:
                        error_msg = error_msg[:300] + "..."
                    
                    comment_parts.extend([
                        "",
                        f"**Error**: {error_msg}"
                    ])
                
                # Add suggestions
                comment_parts.extend([
                    "",
                    "**Suggestions:**",
                    "- Review the issue description for clarity",
                    "- Ensure test cases are well-defined",
                    "- Consider breaking down complex requirements",
                    "- Check if additional context or constraints are needed"
                ])
            
            return "\\n".join(comment_parts)
            
        except Exception as e:
            logger.error(f"Error generating completion comment: {e}")
            return "🔄 **Evolution process completed** - See logs for details."

