"""
Linear GraphQL Queries

Contains all GraphQL queries for Linear API operations.
"""


class LinearQueries:
    """Collection of Linear GraphQL queries"""
    
    # Issue queries
    GET_ISSUE = """
    query GetIssue($id: String!) {
        issue(id: $id) {
            id
            title
            description
            state {
                id
                name
                type
            }
            assignee {
                id
                name
                email
            }
            team {
                id
                name
                key
            }
            project {
                id
                name
            }
            cycle {
                id
                name
            }
            priority
            labels {
                nodes {
                    id
                    name
                    color
                }
            }
            createdAt
            updatedAt
            url
            branchName
            comments {
                nodes {
                    id
                    body
                    user {
                        id
                        name
                    }
                    createdAt
                    updatedAt
                }
            }
        }
    }
    """
    
    GET_TEAM_ISSUES = """
    query GetTeamIssues($teamId: String!, $first: Int = 50, $filter: IssueFilter) {
        team(id: $teamId) {
            issues(first: $first, filter: $filter) {
                nodes {
                    id
                    title
                    description
                    state {
                        id
                        name
                        type
                    }
                    assignee {
                        id
                        name
                        email
                    }
                    project {
                        id
                        name
                    }
                    cycle {
                        id
                        name
                    }
                    priority
                    labels {
                        nodes {
                            id
                            name
                            color
                        }
                    }
                    createdAt
                    updatedAt
                    url
                    branchName
                }
                pageInfo {
                    hasNextPage
                    endCursor
                }
            }
        }
    }
    """
    
    GET_USER_ASSIGNED_ISSUES = """
    query GetUserAssignedIssues($userId: String!, $first: Int = 50) {
        user(id: $userId) {
            assignedIssues(first: $first) {
                nodes {
                    id
                    title
                    description
                    state {
                        id
                        name
                        type
                    }
                    team {
                        id
                        name
                        key
                    }
                    project {
                        id
                        name
                    }
                    cycle {
                        id
                        name
                    }
                    priority
                    labels {
                        nodes {
                            id
                            name
                            color
                        }
                    }
                    createdAt
                    updatedAt
                    url
                    branchName
                }
                pageInfo {
                    hasNextPage
                    endCursor
                }
            }
        }
    }
    """
    
    # Team and organization queries
    GET_TEAMS = """
    query GetTeams($first: Int = 50) {
        teams(first: $first) {
            nodes {
                id
                name
                key
                description
                private
                issueCount
                members {
                    nodes {
                        id
                        name
                        email
                    }
                }
            }
            pageInfo {
                hasNextPage
                endCursor
            }
        }
    }
    """
    
    GET_TEAM = """
    query GetTeam($id: String!) {
        team(id: $id) {
            id
            name
            key
            description
            private
            issueCount
            members {
                nodes {
                    id
                    name
                    email
                }
            }
            states {
                nodes {
                    id
                    name
                    type
                    color
                    position
                }
            }
            labels {
                nodes {
                    id
                    name
                    color
                    description
                }
            }
        }
    }
    """
    
    # User queries
    GET_VIEWER = """
    query GetViewer {
        viewer {
            id
            name
            email
            avatarUrl
            organization {
                id
                name
                urlKey
            }
        }
    }
    """
    
    GET_USER = """
    query GetUser($id: String!) {
        user(id: $id) {
            id
            name
            email
            avatarUrl
            active
            admin
            guest
        }
    }
    """
    
    # Project queries
    GET_PROJECTS = """
    query GetProjects($first: Int = 50) {
        projects(first: $first) {
            nodes {
                id
                name
                description
                state
                progress
                startDate
                targetDate
                lead {
                    id
                    name
                }
                teams {
                    nodes {
                        id
                        name
                        key
                    }
                }
                issues {
                    nodes {
                        id
                        title
                        state {
                            name
                            type
                        }
                    }
                }
            }
            pageInfo {
                hasNextPage
                endCursor
            }
        }
    }
    """
    
    # Cycle queries
    GET_CYCLES = """
    query GetCycles($teamId: String!, $first: Int = 50) {
        team(id: $teamId) {
            cycles(first: $first) {
                nodes {
                    id
                    name
                    description
                    startsAt
                    endsAt
                    progress
                    issues {
                        nodes {
                            id
                            title
                            state {
                                name
                                type
                            }
                        }
                    }
                }
                pageInfo {
                    hasNextPage
                    endCursor
                }
            }
        }
    }
    """
    
    # Comment queries
    GET_ISSUE_COMMENTS = """
    query GetIssueComments($issueId: String!, $first: Int = 50) {
        issue(id: $issueId) {
            comments(first: $first) {
                nodes {
                    id
                    body
                    user {
                        id
                        name
                        email
                    }
                    createdAt
                    updatedAt
                }
                pageInfo {
                    hasNextPage
                    endCursor
                }
            }
        }
    }
    """
    
    # Webhook queries
    GET_WEBHOOKS = """
    query GetWebhooks($first: Int = 50) {
        webhooks(first: $first) {
            nodes {
                id
                label
                url
                enabled
                secret
                resourceTypes
                createdAt
                updatedAt
            }
            pageInfo {
                hasNextPage
                endCursor
            }
        }
    }
    """
    
    # Search queries
    SEARCH_ISSUES = """
    query SearchIssues($query: String!, $first: Int = 50) {
        searchIssues(query: $query, first: $first) {
            nodes {
                id
                title
                description
                state {
                    id
                    name
                    type
                }
                assignee {
                    id
                    name
                }
                team {
                    id
                    name
                    key
                }
                priority
                createdAt
                updatedAt
                url
            }
            pageInfo {
                hasNextPage
                endCursor
            }
        }
    }
    """

