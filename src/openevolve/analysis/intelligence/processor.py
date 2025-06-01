"""
Requirement Processor for natural language requirement analysis.

Processes natural language requirements and converts them into structured
requirement contexts for further analysis and code mapping.
"""

import re
import logging
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass

from ..core.interfaces import (
    RequirementProcessorInterface, RequirementContext, CodeContext,
    AnalysisConfig
)


logger = logging.getLogger(__name__)


@dataclass
class RequirementKeywords:
    """Keywords for different requirement types."""
    feature_keywords: Set[str]
    bug_keywords: Set[str]
    enhancement_keywords: Set[str]
    performance_keywords: Set[str]
    security_keywords: Set[str]


class RequirementProcessor(RequirementProcessorInterface):
    """
    Processor for natural language requirements analysis.
    """
    
    def __init__(self, config: AnalysisConfig):
        """Initialize the requirement processor."""
        self.config = config
        
        # Define keyword sets for requirement classification
        self.keywords = RequirementKeywords(
            feature_keywords={
                'add', 'create', 'implement', 'build', 'develop', 'new', 'feature',
                'functionality', 'capability', 'support', 'enable', 'allow'
            },
            bug_keywords={
                'fix', 'bug', 'error', 'issue', 'problem', 'broken', 'incorrect',
                'wrong', 'fail', 'crash', 'exception', 'defect'
            },
            enhancement_keywords={
                'improve', 'enhance', 'optimize', 'better', 'upgrade', 'refactor',
                'modernize', 'update', 'modify', 'change'
            },
            performance_keywords={
                'performance', 'speed', 'fast', 'slow', 'optimize', 'efficient',
                'memory', 'cpu', 'latency', 'throughput', 'scalability'
            },
            security_keywords={
                'security', 'secure', 'authentication', 'authorization', 'encrypt',
                'vulnerability', 'attack', 'protection', 'safe', 'privacy'
            }
        )
        
        # Priority indicators
        self.priority_indicators = {
            'critical': {'critical', 'urgent', 'emergency', 'blocker', 'high priority'},
            'high': {'important', 'high', 'significant', 'major'},
            'medium': {'medium', 'normal', 'standard', 'moderate'},
            'low': {'low', 'minor', 'nice to have', 'optional', 'future'}
        }
        
        # Complexity indicators
        self.complexity_indicators = {
            'simple': {'simple', 'easy', 'basic', 'straightforward', 'quick'},
            'medium': {'medium', 'moderate', 'standard', 'typical'},
            'complex': {'complex', 'complicated', 'difficult', 'challenging', 'advanced'},
            'very_complex': {'very complex', 'extremely difficult', 'major refactor', 'architectural'}
        }
        
        logger.info("Requirement processor initialized")
    
    async def process_requirements(self, requirements: str) -> RequirementContext:
        """
        Process natural language requirements into structured context.
        
        Args:
            requirements: Natural language requirements text
            
        Returns:
            RequirementContext with analyzed requirements
        """
        logger.debug("Processing requirements")
        
        try:
            # Generate unique ID
            req_id = f"req_{int(time.time())}"
            
            # Clean and normalize text
            cleaned_text = await self._clean_text(requirements)
            
            # Extract requirement type
            req_type = await self._classify_requirement_type(cleaned_text)
            
            # Extract priority
            priority = await self._extract_priority(cleaned_text)
            
            # Estimate complexity
            complexity = await self._estimate_complexity(cleaned_text)
            
            # Extract components and dependencies
            components = await self._extract_affected_components(cleaned_text)
            dependencies = await self._extract_dependencies(cleaned_text)
            
            # Extract acceptance criteria
            acceptance_criteria = await self._extract_acceptance_criteria(cleaned_text)
            
            # Extract technical constraints
            constraints = await self._extract_technical_constraints(cleaned_text)
            
            # Generate suggested approach
            approach = await self._suggest_approach(cleaned_text, req_type)
            
            # Estimate effort
            effort = await self._estimate_effort(complexity, len(components))
            
            # Identify risk factors
            risks = await self._identify_risk_factors(cleaned_text, complexity)
            
            requirement_context = RequirementContext(
                id=req_id,
                description=requirements,
                type=req_type,
                priority=priority,
                complexity_estimate=complexity,
                affected_components=components,
                dependencies=dependencies,
                acceptance_criteria=acceptance_criteria,
                technical_constraints=constraints,
                suggested_approach=approach,
                estimated_effort=effort,
                risk_factors=risks,
                metadata={
                    'processed_text': cleaned_text,
                    'word_count': len(cleaned_text.split()),
                    'sentence_count': len(re.split(r'[.!?]+', cleaned_text))
                }
            )
            
            logger.debug(f"Processed requirement: {req_id} ({req_type})")
            return requirement_context
            
        except Exception as e:
            logger.error(f"Requirement processing failed", exc_info=e)
            return RequirementContext(
                id=f"error_{int(time.time())}",
                description=requirements,
                type="unknown",
                metadata={'error': str(e)}
            )
    
    async def map_to_code(self, requirement: RequirementContext,
                         code_contexts: List[CodeContext]) -> Dict[str, Any]:
        """
        Map requirements to existing code components.
        
        Args:
            requirement: Analyzed requirement context
            code_contexts: List of code contexts to map against
            
        Returns:
            Mapping results with relevance scores
        """
        logger.debug(f"Mapping requirement {requirement.id} to code")
        
        try:
            mappings = []
            
            for code_context in code_contexts:
                relevance_score = await self._calculate_relevance_score(
                    requirement, code_context
                )
                
                if relevance_score > 0.1:  # Threshold for relevance
                    mappings.append({
                        'file_path': code_context.file_path,
                        'relevance_score': relevance_score,
                        'matching_elements': await self._find_matching_elements(
                            requirement, code_context
                        ),
                        'suggested_changes': await self._suggest_code_changes(
                            requirement, code_context
                        )
                    })
            
            # Sort by relevance score
            mappings.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return {
                'requirement_id': requirement.id,
                'total_mappings': len(mappings),
                'high_relevance_count': len([m for m in mappings if m['relevance_score'] > 0.7]),
                'mappings': mappings[:10],  # Top 10 most relevant
                'coverage_score': await self._calculate_coverage_score(requirement, mappings)
            }
            
        except Exception as e:
            logger.error(f"Requirement mapping failed", exc_info=e)
            return {
                'requirement_id': requirement.id,
                'error': str(e),
                'mappings': []
            }
    
    async def decompose_task(self, requirement: RequirementContext) -> List[RequirementContext]:
        """
        Decompose complex requirements into subtasks.
        
        Args:
            requirement: Complex requirement to decompose
            
        Returns:
            List of subtask requirements
        """
        logger.debug(f"Decomposing requirement {requirement.id}")
        
        try:
            subtasks = []
            
            # Simple decomposition based on keywords and structure
            text = requirement.description.lower()
            
            # Look for enumerated items
            enumerated_items = re.findall(r'(?:^|\n)\s*(?:\d+\.|\*|-)\s*(.+)', text, re.MULTILINE)
            
            if enumerated_items:
                for i, item in enumerate(enumerated_items):
                    subtask = RequirementContext(
                        id=f"{requirement.id}_subtask_{i+1}",
                        description=item.strip(),
                        type=requirement.type,
                        priority=requirement.priority,
                        complexity_estimate=requirement.complexity_estimate * 0.3,  # Assume subtasks are simpler
                        metadata={
                            'parent_id': requirement.id,
                            'subtask_index': i + 1,
                            'is_subtask': True
                        }
                    )
                    subtasks.append(subtask)
            
            # Look for logical sections (and, then, also, etc.)
            logical_sections = re.split(r'\b(?:and|then|also|additionally|furthermore)\b', text)
            
            if len(logical_sections) > 1 and not enumerated_items:
                for i, section in enumerate(logical_sections):
                    section = section.strip()
                    if len(section) > 20:  # Minimum meaningful length
                        subtask = RequirementContext(
                            id=f"{requirement.id}_section_{i+1}",
                            description=section,
                            type=requirement.type,
                            priority=requirement.priority,
                            complexity_estimate=requirement.complexity_estimate * 0.4,
                            metadata={
                                'parent_id': requirement.id,
                                'section_index': i + 1,
                                'is_subtask': True
                            }
                        )
                        subtasks.append(subtask)
            
            # If no clear decomposition found, create logical subtasks based on type
            if not subtasks:
                subtasks = await self._create_default_subtasks(requirement)
            
            logger.debug(f"Decomposed requirement into {len(subtasks)} subtasks")
            return subtasks
            
        except Exception as e:
            logger.error(f"Task decomposition failed", exc_info=e)
            return [requirement]  # Return original if decomposition fails
    
    async def _clean_text(self, text: str) -> str:
        """Clean and normalize requirement text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        return text.strip()
    
    async def _classify_requirement_type(self, text: str) -> str:
        """Classify the type of requirement."""
        text_lower = text.lower()
        
        # Count keyword matches for each type
        type_scores = {}
        
        for keyword in self.keywords.feature_keywords:
            if keyword in text_lower:
                type_scores['feature'] = type_scores.get('feature', 0) + 1
        
        for keyword in self.keywords.bug_keywords:
            if keyword in text_lower:
                type_scores['bug_fix'] = type_scores.get('bug_fix', 0) + 1
        
        for keyword in self.keywords.enhancement_keywords:
            if keyword in text_lower:
                type_scores['enhancement'] = type_scores.get('enhancement', 0) + 1
        
        for keyword in self.keywords.performance_keywords:
            if keyword in text_lower:
                type_scores['performance'] = type_scores.get('performance', 0) + 1
        
        for keyword in self.keywords.security_keywords:
            if keyword in text_lower:
                type_scores['security'] = type_scores.get('security', 0) + 1
        
        # Return type with highest score
        if type_scores:
            return max(type_scores, key=type_scores.get)
        else:
            return 'feature'  # Default
    
    async def _extract_priority(self, text: str) -> int:
        """Extract priority from requirement text."""
        text_lower = text.lower()
        
        for priority_level, keywords in self.priority_indicators.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if priority_level == 'critical':
                        return 5
                    elif priority_level == 'high':
                        return 4
                    elif priority_level == 'medium':
                        return 3
                    elif priority_level == 'low':
                        return 1
        
        return 2  # Default medium-low priority
    
    async def _estimate_complexity(self, text: str) -> float:
        """Estimate complexity based on text analysis."""
        text_lower = text.lower()
        
        # Base complexity
        complexity = 1.0
        
        # Adjust based on complexity indicators
        for level, keywords in self.complexity_indicators.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if level == 'simple':
                        complexity *= 0.5
                    elif level == 'medium':
                        complexity *= 1.0
                    elif level == 'complex':
                        complexity *= 2.0
                    elif level == 'very_complex':
                        complexity *= 3.0
                    break
        
        # Adjust based on text length and structure
        word_count = len(text.split())
        if word_count > 100:
            complexity *= 1.5
        elif word_count > 200:
            complexity *= 2.0
        
        # Adjust based on technical terms
        technical_terms = [
            'algorithm', 'database', 'api', 'integration', 'architecture',
            'framework', 'library', 'protocol', 'encryption', 'optimization'
        ]
        
        tech_count = sum(1 for term in technical_terms if term in text_lower)
        complexity += tech_count * 0.3
        
        return min(complexity, 10.0)  # Cap at 10
    
    async def _extract_affected_components(self, text: str) -> List[str]:
        """Extract affected components from requirement text."""
        components = []
        
        # Common component patterns
        component_patterns = [
            r'\b(?:user interface|ui|frontend)\b',
            r'\b(?:backend|server|api)\b',
            r'\b(?:database|db|storage)\b',
            r'\b(?:authentication|auth|login)\b',
            r'\b(?:payment|billing)\b',
            r'\b(?:notification|email|messaging)\b',
            r'\b(?:search|indexing)\b',
            r'\b(?:reporting|analytics)\b',
            r'\b(?:configuration|settings)\b',
            r'\b(?:security|permissions)\b'
        ]
        
        text_lower = text.lower()
        for pattern in component_patterns:
            matches = re.findall(pattern, text_lower)
            components.extend(matches)
        
        # Remove duplicates and clean up
        components = list(set(components))
        return [comp.replace(' ', '_') for comp in components]
    
    async def _extract_dependencies(self, text: str) -> List[str]:
        """Extract dependencies from requirement text."""
        dependencies = []
        
        # Look for dependency indicators
        dependency_patterns = [
            r'depends on (.+?)(?:\.|,|$)',
            r'requires (.+?)(?:\.|,|$)',
            r'needs (.+?)(?:\.|,|$)',
            r'after (.+?)(?:\.|,|$)',
            r'before (.+?)(?:\.|,|$)'
        ]
        
        for pattern in dependency_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dependencies.extend(matches)
        
        return [dep.strip() for dep in dependencies]
    
    async def _extract_acceptance_criteria(self, text: str) -> List[str]:
        """Extract acceptance criteria from requirement text."""
        criteria = []
        
        # Look for criteria patterns
        criteria_patterns = [
            r'(?:should|must|shall|will)\s+(.+?)(?:\.|,|$)',
            r'(?:given|when|then)\s+(.+?)(?:\.|,|$)',
            r'(?:acceptance criteria|criteria):\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in criteria_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            criteria.extend(matches)
        
        return [criterion.strip() for criterion in criteria if len(criterion.strip()) > 10]
    
    async def _extract_technical_constraints(self, text: str) -> List[str]:
        """Extract technical constraints from requirement text."""
        constraints = []
        
        # Look for constraint patterns
        constraint_patterns = [
            r'(?:constraint|limitation|restriction):\s*(.+?)(?:\.|,|$)',
            r'(?:cannot|must not|should not)\s+(.+?)(?:\.|,|$)',
            r'(?:within|under|less than)\s+(\d+\s*(?:ms|seconds|minutes|mb|gb))',
            r'(?:compatible with|supports?)\s+(.+?)(?:\.|,|$)'
        ]
        
        for pattern in constraint_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            constraints.extend(matches)
        
        return [constraint.strip() for constraint in constraints]
    
    async def _suggest_approach(self, text: str, req_type: str) -> str:
        """Suggest implementation approach based on requirement."""
        approaches = {
            'feature': 'Implement new functionality with proper testing and documentation',
            'bug_fix': 'Identify root cause, implement fix, and add regression tests',
            'enhancement': 'Analyze current implementation and improve incrementally',
            'performance': 'Profile current performance, identify bottlenecks, and optimize',
            'security': 'Conduct security review and implement necessary safeguards'
        }
        
        base_approach = approaches.get(req_type, 'Analyze requirements and implement solution')
        
        # Add specific suggestions based on text content
        text_lower = text.lower()
        
        if 'test' in text_lower:
            base_approach += '. Ensure comprehensive test coverage.'
        
        if 'documentation' in text_lower:
            base_approach += '. Update relevant documentation.'
        
        if 'backward compatible' in text_lower:
            base_approach += '. Maintain backward compatibility.'
        
        return base_approach
    
    async def _estimate_effort(self, complexity: float, component_count: int) -> float:
        """Estimate effort in story points or hours."""
        # Base effort from complexity
        effort = complexity * 2
        
        # Adjust for number of components
        effort += component_count * 0.5
        
        # Add overhead for coordination
        if component_count > 3:
            effort *= 1.2
        
        return round(effort, 1)
    
    async def _identify_risk_factors(self, text: str, complexity: float) -> List[str]:
        """Identify potential risk factors."""
        risks = []
        
        text_lower = text.lower()
        
        # High complexity risk
        if complexity > 5:
            risks.append('High complexity may lead to implementation challenges')
        
        # Integration risks
        if any(term in text_lower for term in ['integration', 'third party', 'external']):
            risks.append('External dependencies may cause integration issues')
        
        # Performance risks
        if any(term in text_lower for term in ['performance', 'scale', 'large data']):
            risks.append('Performance requirements may be challenging to meet')
        
        # Security risks
        if any(term in text_lower for term in ['security', 'authentication', 'sensitive']):
            risks.append('Security requirements need careful implementation')
        
        # Timeline risks
        if any(term in text_lower for term in ['urgent', 'asap', 'immediately']):
            risks.append('Tight timeline may impact quality')
        
        return risks
    
    async def _calculate_relevance_score(self, requirement: RequirementContext,
                                       code_context: CodeContext) -> float:
        """Calculate relevance score between requirement and code."""
        score = 0.0
        
        req_text = requirement.description.lower()
        code_content = code_context.content.lower()
        
        # Check for direct keyword matches
        req_words = set(re.findall(r'\b\w+\b', req_text))
        code_words = set(re.findall(r'\b\w+\b', code_content))
        
        common_words = req_words.intersection(code_words)
        if req_words:
            keyword_score = len(common_words) / len(req_words)
            score += keyword_score * 0.4
        
        # Check for component matches
        for component in requirement.affected_components:
            if component.lower() in code_content:
                score += 0.2
        
        # Check file path relevance
        file_path_lower = code_context.file_path.lower()
        for component in requirement.affected_components:
            if component.lower() in file_path_lower:
                score += 0.3
        
        # Check for function/class name matches
        for element in code_context.elements:
            if element.name.lower() in req_text:
                score += 0.1
        
        return min(score, 1.0)
    
    async def _find_matching_elements(self, requirement: RequirementContext,
                                    code_context: CodeContext) -> List[Dict[str, Any]]:
        """Find code elements that match the requirement."""
        matching_elements = []
        
        req_text = requirement.description.lower()
        
        for element in code_context.elements:
            relevance = 0.0
            
            # Check name similarity
            if element.name.lower() in req_text:
                relevance += 0.5
            
            # Check content similarity
            element_content = element.content.lower()
            req_words = set(re.findall(r'\b\w+\b', req_text))
            element_words = set(re.findall(r'\b\w+\b', element_content))
            
            common_words = req_words.intersection(element_words)
            if req_words:
                relevance += (len(common_words) / len(req_words)) * 0.3
            
            if relevance > 0.1:
                matching_elements.append({
                    'name': element.name,
                    'type': element.type,
                    'line': element.start_line,
                    'relevance': relevance
                })
        
        return sorted(matching_elements, key=lambda x: x['relevance'], reverse=True)
    
    async def _suggest_code_changes(self, requirement: RequirementContext,
                                  code_context: CodeContext) -> List[str]:
        """Suggest specific code changes based on requirement."""
        suggestions = []
        
        req_type = requirement.type
        
        if req_type == 'feature':
            suggestions.append('Add new function or method to implement the feature')
            suggestions.append('Update existing interfaces if needed')
            suggestions.append('Add appropriate error handling')
        
        elif req_type == 'bug_fix':
            suggestions.append('Identify and fix the root cause of the issue')
            suggestions.append('Add validation to prevent similar issues')
            suggestions.append('Update error messages if applicable')
        
        elif req_type == 'enhancement':
            suggestions.append('Refactor existing code for better performance')
            suggestions.append('Improve code readability and maintainability')
            suggestions.append('Add additional configuration options')
        
        elif req_type == 'performance':
            suggestions.append('Optimize algorithms and data structures')
            suggestions.append('Add caching where appropriate')
            suggestions.append('Profile and eliminate bottlenecks')
        
        elif req_type == 'security':
            suggestions.append('Add input validation and sanitization')
            suggestions.append('Implement proper authentication checks')
            suggestions.append('Add security logging and monitoring')
        
        return suggestions
    
    async def _calculate_coverage_score(self, requirement: RequirementContext,
                                      mappings: List[Dict[str, Any]]) -> float:
        """Calculate how well the mappings cover the requirement."""
        if not mappings:
            return 0.0
        
        # Simple coverage based on number of high-relevance mappings
        high_relevance_count = len([m for m in mappings if m['relevance_score'] > 0.7])
        medium_relevance_count = len([m for m in mappings if 0.4 <= m['relevance_score'] <= 0.7])
        
        coverage = (high_relevance_count * 0.8 + medium_relevance_count * 0.4) / len(requirement.affected_components or [1])
        
        return min(coverage, 1.0)
    
    async def _create_default_subtasks(self, requirement: RequirementContext) -> List[RequirementContext]:
        """Create default subtasks based on requirement type."""
        subtasks = []
        
        base_tasks = {
            'feature': [
                'Design and plan the feature implementation',
                'Implement core functionality',
                'Add comprehensive testing',
                'Update documentation'
            ],
            'bug_fix': [
                'Investigate and reproduce the issue',
                'Identify root cause',
                'Implement fix',
                'Add regression tests'
            ],
            'enhancement': [
                'Analyze current implementation',
                'Design improvements',
                'Implement enhancements',
                'Validate improvements'
            ]
        }
        
        tasks = base_tasks.get(requirement.type, base_tasks['feature'])
        
        for i, task_desc in enumerate(tasks):
            subtask = RequirementContext(
                id=f"{requirement.id}_default_{i+1}",
                description=task_desc,
                type=requirement.type,
                priority=requirement.priority,
                complexity_estimate=requirement.complexity_estimate * 0.25,
                metadata={
                    'parent_id': requirement.id,
                    'is_subtask': True,
                    'is_default_subtask': True
                }
            )
            subtasks.append(subtask)
        
        return subtasks

