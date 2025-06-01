"""
Linear GraphQL Mutations

Contains all GraphQL mutations for Linear API operations.
"""


class LinearMutations:
    """Collection of Linear GraphQL mutations"""
    
    # Issue mutations
    CREATE_ISSUE = """
    mutation CreateIssue($input: IssueCreateInput!) {
        issueCreate(input: $input) {
            success
            issue {
                id
                title
                description
                state {
                    id
                    name
                }
                assignee {
                    id
                    name
                }
                team {
                    id
                    name
                }
                url
            }
            lastSyncId
        }
    }
    """
    
    UPDATE_ISSUE = """
    mutation UpdateIssue($id: String!, $input: IssueUpdateInput!) {
        issueUpdate(id: $id, input: $input) {
            success
            issue {
                id
                title
                description
                state {
                    id
                    name
                }
                assignee {
                    id
                    name
                }
                priority
                updatedAt
            }
            lastSyncId
        }
    }
    """
    
    DELETE_ISSUE = """
    mutation DeleteIssue($id: String!) {
        issueDelete(id: $id) {
            success
            lastSyncId
        }
    }
    """
    
    ASSIGN_ISSUE = """
    mutation AssignIssue($id: String!, $assigneeId: String!) {
        issueUpdate(id: $id, input: { assigneeId: $assigneeId }) {
            success
            issue {
                id
                assignee {
                    id
                    name
                    email
                }
                updatedAt
            }
            lastSyncId
        }
    }
    """
    
    UNASSIGN_ISSUE = """
    mutation UnassignIssue($id: String!) {
        issueUpdate(id: $id, input: { assigneeId: null }) {
            success
            issue {
                id
                assignee {
                    id
                    name
                }
                updatedAt
            }
            lastSyncId
        }
    }
    """
    
    UPDATE_ISSUE_STATE = """
    mutation UpdateIssueState($id: String!, $stateId: String!) {
        issueUpdate(id: $id, input: { stateId: $stateId }) {
            success
            issue {
                id
                state {
                    id
                    name
                    type
                }
                updatedAt
            }
            lastSyncId
        }
    }
    """
    
    UPDATE_ISSUE_PRIORITY = """
    mutation UpdateIssuePriority($id: String!, $priority: Int!) {
        issueUpdate(id: $id, input: { priority: $priority }) {
            success
            issue {
                id
                priority
                updatedAt
            }
            lastSyncId
        }
    }
    """
    
    ADD_ISSUE_LABELS = """
    mutation AddIssueLabels($id: String!, $labelIds: [String!]!) {
        issueUpdate(id: $id, input: { labelIds: $labelIds }) {
            success
            issue {
                id
                labels {
                    nodes {
                        id
                        name
                        color
                    }
                }
                updatedAt
            }
            lastSyncId
        }
    }
    """
    
    # Comment mutations
    CREATE_COMMENT = """
    mutation CreateComment($input: CommentCreateInput!) {
        commentCreate(input: $input) {
            success
            comment {
                id
                body
                user {
                    id
                    name
                }
                issue {
                    id
                    title
                }
                createdAt
            }
            lastSyncId
        }
    }
    """
    
    UPDATE_COMMENT = """
    mutation UpdateComment($id: String!, $input: CommentUpdateInput!) {
        commentUpdate(id: $id, input: $input) {
            success
            comment {
                id
                body
                updatedAt
            }
            lastSyncId
        }
    }
    """
    
    DELETE_COMMENT = """
    mutation DeleteComment($id: String!) {
        commentDelete(id: $id) {
            success
            lastSyncId
        }
    }
    """
    
    # Project mutations
    CREATE_PROJECT = """
    mutation CreateProject($input: ProjectCreateInput!) {
        projectCreate(input: $input) {
            success
            project {
                id
                name
                description
                state
                lead {
                    id
                    name
                }
                createdAt
            }
            lastSyncId
        }
    }
    """
    
    UPDATE_PROJECT = """
    mutation UpdateProject($id: String!, $input: ProjectUpdateInput!) {
        projectUpdate(id: $id, input: $input) {
            success
            project {
                id
                name
                description
                state
                progress
                updatedAt
            }
            lastSyncId
        }
    }
    """
    
    # Cycle mutations
    CREATE_CYCLE = """
    mutation CreateCycle($input: CycleCreateInput!) {
        cycleCreate(input: $input) {
            success
            cycle {
                id
                name
                description
                startsAt
                endsAt
                team {
                    id
                    name
                }
                createdAt
            }
            lastSyncId
        }
    }
    """
    
    UPDATE_CYCLE = """
    mutation UpdateCycle($id: String!, $input: CycleUpdateInput!) {
        cycleUpdate(id: $id, input: $input) {
            success
            cycle {
                id
                name
                description
                startsAt
                endsAt
                progress
                updatedAt
            }
            lastSyncId
        }
    }
    """
    
    # Webhook mutations
    CREATE_WEBHOOK = """
    mutation CreateWebhook($input: WebhookCreateInput!) {
        webhookCreate(input: $input) {
            success
            webhook {
                id
                label
                url
                enabled
                secret
                resourceTypes
                createdAt
            }
            lastSyncId
        }
    }
    """
    
    UPDATE_WEBHOOK = """
    mutation UpdateWebhook($id: String!, $input: WebhookUpdateInput!) {
        webhookUpdate(id: $id, input: $input) {
            success
            webhook {
                id
                label
                url
                enabled
                resourceTypes
                updatedAt
            }
            lastSyncId
        }
    }
    """
    
    DELETE_WEBHOOK = """
    mutation DeleteWebhook($id: String!) {
        webhookDelete(id: $id) {
            success
            lastSyncId
        }
    }
    """
    
    # Team mutations
    CREATE_TEAM = """
    mutation CreateTeam($input: TeamCreateInput!) {
        teamCreate(input: $input) {
            success
            team {
                id
                name
                key
                description
                private
                createdAt
            }
            lastSyncId
        }
    }
    """
    
    UPDATE_TEAM = """
    mutation UpdateTeam($id: String!, $input: TeamUpdateInput!) {
        teamUpdate(id: $id, input: $input) {
            success
            team {
                id
                name
                key
                description
                private
                updatedAt
            }
            lastSyncId
        }
    }
    """
    
    # Label mutations
    CREATE_ISSUE_LABEL = """
    mutation CreateIssueLabel($input: IssueLabelCreateInput!) {
        issueLabelCreate(input: $input) {
            success
            issueLabel {
                id
                name
                color
                description
                team {
                    id
                    name
                }
                createdAt
            }
            lastSyncId
        }
    }
    """
    
    UPDATE_ISSUE_LABEL = """
    mutation UpdateIssueLabel($id: String!, $input: IssueLabelUpdateInput!) {
        issueLabelUpdate(id: $id, input: $input) {
            success
            issueLabel {
                id
                name
                color
                description
                updatedAt
            }
            lastSyncId
        }
    }
    """

