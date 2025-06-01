"""
Linear GraphQL Client

Provides GraphQL client implementation for Linear API communication with
authentication, rate limiting, caching, and error handling.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import json

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..interfaces import LinearGraphQLClientInterface, LinearIssue, LinearComment
from .queries import LinearQueries
from .mutations import LinearMutations

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, max_requests: int = 100, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire rate limit permission"""
        async with self._lock:
            now = time.time()
            # Remove old requests outside the time window
            self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
            
            if len(self.requests) >= self.max_requests:
                # Calculate wait time
                oldest_request = min(self.requests)
                wait_time = self.time_window - (now - oldest_request)
                if wait_time > 0:
                    logger.warning(f"Rate limit reached. Waiting {wait_time:.2f} seconds")
                    await asyncio.sleep(wait_time)
                    return await self.acquire()
            
            self.requests.append(now)


class ResponseCache:
    """Simple in-memory cache for GraphQL responses"""
    
    def __init__(self, default_ttl: int = 300):  # 5 minutes default TTL
        self.cache = {}
        self.default_ttl = default_ttl
        self._lock = asyncio.Lock()
    
    def _generate_key(self, query: str, variables: Optional[Dict] = None) -> str:
        """Generate cache key from query and variables"""
        key_data = {"query": query, "variables": variables or {}}
        return str(hash(json.dumps(key_data, sort_keys=True)))
    
    async def get(self, query: str, variables: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        async with self._lock:
            key = self._generate_key(query, variables)
            if key in self.cache:
                cached_data, expiry = self.cache[key]
                if time.time() < expiry:
                    logger.debug(f"Cache hit for query: {query[:50]}...")
                    return cached_data
                else:
                    del self.cache[key]
            return None
    
    async def set(self, query: str, variables: Optional[Dict], response: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Cache response"""
        async with self._lock:
            key = self._generate_key(query, variables)
            expiry = time.time() + (ttl or self.default_ttl)
            self.cache[key] = (response, expiry)
            logger.debug(f"Cached response for query: {query[:50]}...")
    
    async def clear(self) -> None:
        """Clear all cached responses"""
        async with self._lock:
            self.cache.clear()
            logger.info("Response cache cleared")


class LinearGraphQLClient(LinearGraphQLClientInterface):
    """Linear GraphQL client with authentication, rate limiting, and caching"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api_url = self.config.get("api_url", "https://api.linear.app/graphql")
        self.api_key = self.config.get("api_key")
        self.timeout = self.config.get("timeout", 30)
        self.max_retries = self.config.get("max_retries", 3)
        self.rate_limit_requests = self.config.get("rate_limit_requests", 100)
        self.rate_limit_window = self.config.get("rate_limit_window", 60)
        self.cache_ttl = self.config.get("cache_ttl", 300)
        
        # Initialize components
        self.rate_limiter = RateLimiter(self.rate_limit_requests, self.rate_limit_window)
        self.cache = ResponseCache(self.cache_ttl)
        self.queries = LinearQueries()
        self.mutations = LinearMutations()
        
        # Setup HTTP session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "POST"],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.authenticated = False
        self.user_info = None
        
        logger.info("LinearGraphQLClient initialized")
    
    async def authenticate(self, api_key: str) -> bool:
        """Authenticate with Linear API"""
        try:
            self.api_key = api_key
            self.session.headers.update({
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "OpenAlpha_Evolve-Linear-Integration/1.0"
            })
            
            # Test authentication by getting viewer info
            response = await self.execute_query(self.queries.GET_VIEWER)
            if response and "data" in response and "viewer" in response["data"]:
                self.authenticated = True
                self.user_info = response["data"]["viewer"]
                logger.info(f"Successfully authenticated as {self.user_info.get('name', 'Unknown')}")
                return True
            else:
                logger.error("Authentication failed: Invalid response")
                return False
                
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    async def execute_query(self, query: str, variables: Optional[Dict] = None, use_cache: bool = True) -> Dict[str, Any]:
        """Execute GraphQL query with rate limiting and caching"""
        if not self.api_key:
            raise ValueError("API key not set. Call authenticate() first.")
        
        # Check cache for read queries
        if use_cache and "mutation" not in query.lower():
            cached_response = await self.cache.get(query, variables)
            if cached_response:
                return cached_response
        
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        try:
            payload = {
                "query": query,
                "variables": variables or {}
            }
            
            logger.debug(f"Executing GraphQL query: {query[:100]}...")
            
            response = self.session.post(
                self.api_url,
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Check for GraphQL errors
            if "errors" in result:
                logger.error(f"GraphQL errors: {result['errors']}")
                raise Exception(f"GraphQL errors: {result['errors']}")
            
            # Cache successful read queries
            if use_cache and "mutation" not in query.lower() and "data" in result:
                await self.cache.set(query, variables, result)
            
            logger.debug("GraphQL query executed successfully")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"GraphQL query execution failed: {e}")
            raise
    
    async def get_issue(self, issue_id: str) -> Optional[LinearIssue]:
        """Get issue by ID"""
        try:
            response = await self.execute_query(
                self.queries.GET_ISSUE,
                {"id": issue_id}
            )
            
            if response and "data" in response and "issue" in response["data"]:
                issue_data = response["data"]["issue"]
                return self._parse_issue(issue_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get issue {issue_id}: {e}")
            return None
    
    async def update_issue(self, issue_id: str, updates: Dict[str, Any]) -> bool:
        """Update issue"""
        try:
            response = await self.execute_query(
                self.mutations.UPDATE_ISSUE,
                {"id": issue_id, "input": updates},
                use_cache=False
            )
            
            if response and "data" in response and "issueUpdate" in response["data"]:
                success = response["data"]["issueUpdate"]["success"]
                if success:
                    logger.info(f"Successfully updated issue {issue_id}")
                    # Clear cache for this issue
                    await self.cache.clear()
                return success
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update issue {issue_id}: {e}")
            return False
    
    async def create_comment(self, issue_id: str, body: str) -> Optional[str]:
        """Create comment on issue"""
        try:
            response = await self.execute_query(
                self.mutations.CREATE_COMMENT,
                {
                    "input": {
                        "issueId": issue_id,
                        "body": body
                    }
                },
                use_cache=False
            )
            
            if response and "data" in response and "commentCreate" in response["data"]:
                comment_data = response["data"]["commentCreate"]
                if comment_data["success"]:
                    comment_id = comment_data["comment"]["id"]
                    logger.info(f"Successfully created comment {comment_id} on issue {issue_id}")
                    return comment_id
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create comment on issue {issue_id}: {e}")
            return None
    
    async def get_team_issues(self, team_id: str, limit: int = 50, filters: Optional[Dict] = None) -> List[LinearIssue]:
        """Get issues for a team"""
        try:
            variables = {
                "teamId": team_id,
                "first": limit
            }
            if filters:
                variables["filter"] = filters
            
            response = await self.execute_query(
                self.queries.GET_TEAM_ISSUES,
                variables
            )
            
            if response and "data" in response and "team" in response["data"]:
                team_data = response["data"]["team"]
                if team_data and "issues" in team_data:
                    issues_data = team_data["issues"]["nodes"]
                    return [self._parse_issue(issue_data) for issue_data in issues_data]
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get team issues for team {team_id}: {e}")
            return []
    
    async def get_user_assigned_issues(self, user_id: str, limit: int = 50) -> List[LinearIssue]:
        """Get issues assigned to a user"""
        try:
            response = await self.execute_query(
                self.queries.GET_USER_ASSIGNED_ISSUES,
                {"userId": user_id, "first": limit}
            )
            
            if response and "data" in response and "user" in response["data"]:
                user_data = response["data"]["user"]
                if user_data and "assignedIssues" in user_data:
                    issues_data = user_data["assignedIssues"]["nodes"]
                    return [self._parse_issue(issue_data) for issue_data in issues_data]
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get assigned issues for user {user_id}: {e}")
            return []
    
    async def search_issues(self, query: str, limit: int = 50) -> List[LinearIssue]:
        """Search issues"""
        try:
            response = await self.execute_query(
                self.queries.SEARCH_ISSUES,
                {"query": query, "first": limit}
            )
            
            if response and "data" in response and "searchIssues" in response["data"]:
                issues_data = response["data"]["searchIssues"]["nodes"]
                return [self._parse_issue(issue_data) for issue_data in issues_data]
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to search issues with query '{query}': {e}")
            return []
    
    def _parse_issue(self, issue_data: Dict[str, Any]) -> LinearIssue:
        """Parse issue data from GraphQL response"""
        assignee = issue_data.get("assignee")
        team = issue_data.get("team")
        project = issue_data.get("project")
        cycle = issue_data.get("cycle")
        state = issue_data.get("state")
        labels = issue_data.get("labels", {}).get("nodes", [])
        
        return LinearIssue(
            id=issue_data["id"],
            title=issue_data["title"],
            description=issue_data.get("description"),
            state=state["name"] if state else None,
            assignee_id=assignee["id"] if assignee else None,
            assignee_name=assignee["name"] if assignee else None,
            team_id=team["id"] if team else None,
            team_name=team["name"] if team else None,
            project_id=project["id"] if project else None,
            cycle_id=cycle["id"] if cycle else None,
            priority=issue_data.get("priority"),
            labels=[label["name"] for label in labels],
            created_at=datetime.fromisoformat(issue_data["createdAt"].replace("Z", "+00:00")) if issue_data.get("createdAt") else None,
            updated_at=datetime.fromisoformat(issue_data["updatedAt"].replace("Z", "+00:00")) if issue_data.get("updatedAt") else None,
            url=issue_data.get("url"),
            branch_name=issue_data.get("branchName")
        )
    
    async def get_teams(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get teams"""
        try:
            response = await self.execute_query(
                self.queries.GET_TEAMS,
                {"first": limit}
            )
            
            if response and "data" in response and "teams" in response["data"]:
                return response["data"]["teams"]["nodes"]
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get teams: {e}")
            return []
    
    async def get_projects(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get projects"""
        try:
            response = await self.execute_query(
                self.queries.GET_PROJECTS,
                {"first": limit}
            )
            
            if response and "data" in response and "projects" in response["data"]:
                return response["data"]["projects"]["nodes"]
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get projects: {e}")
            return []
    
    async def execute(self, *args, **kwargs) -> Any:
        """Execute method for BaseAgent interface"""
        # This could be used for health checks or general operations
        if not self.authenticated:
            return {"status": "error", "message": "Not authenticated"}
        
        return {
            "status": "ready",
            "authenticated": self.authenticated,
            "user": self.user_info,
            "cache_size": len(self.cache.cache)
        }

