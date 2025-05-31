#!/usr/bin/env python3
"""
Context Analysis Engine Demo

Demonstrates the capabilities of the Context Analysis Engine including:
- Code analysis and semantic understanding
- Requirement processing and mapping
- Recommendation generation
- Dependency graph construction
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from openevolve.analysis import (
    ContextAnalysisEngine, AnalysisConfig, AnalysisType, LanguageType
)


async def demo_code_analysis():
    """Demonstrate code analysis capabilities."""
    print("🔍 Context Analysis Engine Demo")
    print("=" * 50)
    
    # Create configuration
    config = AnalysisConfig(
        enabled_analyses=[
            AnalysisType.SEMANTIC,
            AnalysisType.COMPLEXITY,
            AnalysisType.QUALITY,
            AnalysisType.PATTERN,
            AnalysisType.DEPENDENCY
        ],
        supported_languages=[LanguageType.PYTHON, LanguageType.JAVASCRIPT],
        enable_caching=True,
        parallel_processing=True,
        verbose_output=True
    )
    
    # Initialize the engine
    engine = ContextAnalysisEngine(config)
    
    # Sample Python code for analysis
    sample_code = '''
import math
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeometryCalculator:
    """A calculator for geometric operations."""
    
    def __init__(self):
        self.calculations_count = 0
        logger.info("GeometryCalculator initialized")
    
    def calculate_circle_area(self, radius: float) -> float:
        """Calculate the area of a circle."""
        if radius < 0:
            raise ValueError("Radius cannot be negative")
        
        self.calculations_count += 1
        area = math.pi * radius ** 2
        logger.debug(f"Calculated circle area: {area}")
        return area
    
    def calculate_rectangle_area(self, width: float, height: float) -> float:
        """Calculate the area of a rectangle."""
        if width < 0 or height < 0:
            raise ValueError("Dimensions cannot be negative")
        
        self.calculations_count += 1
        area = width * height
        logger.debug(f"Calculated rectangle area: {area}")
        return area
    
    def calculate_triangle_area(self, base: float, height: float) -> float:
        """Calculate the area of a triangle."""
        if base < 0 or height < 0:
            raise ValueError("Dimensions cannot be negative")
        
        self.calculations_count += 1
        area = 0.5 * base * height
        logger.debug(f"Calculated triangle area: {area}")
        return area
    
    def get_statistics(self) -> dict:
        """Get calculation statistics."""
        return {
            "total_calculations": self.calculations_count,
            "calculator_type": "GeometryCalculator"
        }

def batch_calculate_areas(shapes: List[dict]) -> List[float]:
    """Calculate areas for multiple shapes."""
    calculator = GeometryCalculator()
    results = []
    
    for shape in shapes:
        try:
            if shape["type"] == "circle":
                area = calculator.calculate_circle_area(shape["radius"])
            elif shape["type"] == "rectangle":
                area = calculator.calculate_rectangle_area(shape["width"], shape["height"])
            elif shape["type"] == "triangle":
                area = calculator.calculate_triangle_area(shape["base"], shape["height"])
            else:
                logger.warning(f"Unknown shape type: {shape['type']}")
                continue
            
            results.append(area)
            
        except ValueError as e:
            logger.error(f"Error calculating area for {shape}: {e}")
            results.append(0.0)
    
    return results

# Example usage
if __name__ == "__main__":
    calculator = GeometryCalculator()
    
    # Test calculations
    circle_area = calculator.calculate_circle_area(5.0)
    rectangle_area = calculator.calculate_rectangle_area(4.0, 6.0)
    triangle_area = calculator.calculate_triangle_area(3.0, 8.0)
    
    print(f"Circle area: {circle_area}")
    print(f"Rectangle area: {rectangle_area}")
    print(f"Triangle area: {triangle_area}")
    
    # Batch calculations
    shapes = [
        {"type": "circle", "radius": 3.0},
        {"type": "rectangle", "width": 5.0, "height": 4.0},
        {"type": "triangle", "base": 6.0, "height": 7.0}
    ]
    
    areas = batch_calculate_areas(shapes)
    print(f"Batch calculation results: {areas}")
    
    # Print statistics
    stats = calculator.get_statistics()
    print(f"Calculator statistics: {stats}")
'''
    
    print("\n📝 Analyzing Sample Python Code...")
    print("-" * 30)
    
    # Perform analysis
    result = await engine.analyze(sample_code, "geometry_calculator.py")
    
    # Display results
    print(f"✅ Analysis completed in {result.execution_time:.2f} seconds")
    print(f"📊 Analysis ID: {result.analysis_id}")
    
    if result.errors:
        print(f"❌ Errors: {len(result.errors)}")
        for error in result.errors:
            print(f"   - {error}")
    
    if result.warnings:
        print(f"⚠️  Warnings: {len(result.warnings)}")
        for warning in result.warnings:
            print(f"   - {warning}")
    
    # Code context information
    code_context = result.code_contexts[0]
    print(f"\n🔤 Language: {code_context.language.value}")
    print(f"📄 Elements found: {len(code_context.elements)}")
    
    # Display extracted elements
    print("\n🏗️  Code Elements:")
    for element in code_context.elements[:10]:  # Show first 10
        print(f"   - {element.type}: {element.name} (line {element.start_line})")
    
    # Display patterns
    if code_context.patterns:
        print(f"\n🎨 Detected Patterns: {', '.join(code_context.patterns)}")
    
    # Display complexity metrics
    if code_context.complexity_metrics:
        print(f"\n📈 Complexity Metrics:")
        for metric, value in code_context.complexity_metrics.items():
            print(f"   - {metric}: {value:.2f}")
    
    # Display quality metrics
    if code_context.quality_metrics:
        print(f"\n⭐ Quality Metrics:")
        for metric, value in code_context.quality_metrics.items():
            print(f"   - {metric}: {value:.2f}")
    
    # Display dependency graph
    if result.dependency_graph:
        print(f"\n🕸️  Dependency Graph:")
        print(f"   - Nodes: {len(result.dependency_graph.nodes)}")
        print(f"   - Edges: {len(result.dependency_graph.edges)}")
        
        # Show some dependencies
        if result.dependency_graph.edges:
            print("   - Sample dependencies:")
            for edge in result.dependency_graph.edges[:5]:
                print(f"     {edge.source} → {edge.target} ({edge.relationship_type})")
    
    # Display recommendations
    if result.recommendations:
        print(f"\n💡 Recommendations ({len(result.recommendations)}):")
        for rec in result.recommendations[:5]:  # Show top 5
            print(f"   - [{rec.type}] {rec.title} (Priority: {rec.priority})")
            print(f"     {rec.description}")
    
    return result


async def demo_requirement_analysis(engine):
    """Demonstrate requirement analysis capabilities."""
    print("\n" + "=" * 50)
    print("📋 Requirement Analysis Demo")
    print("=" * 50)
    
    # Sample requirements
    requirements = [
        """
        Add a new caching mechanism to improve the performance of the GeometryCalculator.
        The cache should store previously calculated results and return them for identical inputs.
        This is a high priority enhancement that should reduce computation time by at least 50%.
        The implementation should be thread-safe and configurable.
        """,
        
        """
        Fix the bug where negative values cause the application to crash.
        Users report that entering negative dimensions results in unhandled exceptions.
        This is a critical issue that needs immediate attention.
        Add proper input validation and user-friendly error messages.
        """,
        
        """
        Implement a new feature for calculating the volume of 3D shapes.
        Support for sphere, cube, and cylinder volumes is required.
        The feature should integrate seamlessly with the existing GeometryCalculator class.
        Include comprehensive unit tests and documentation.
        """
    ]
    
    for i, req_text in enumerate(requirements, 1):
        print(f"\n📝 Analyzing Requirement #{i}...")
        print("-" * 30)
        
        # Process requirement
        requirement = await engine.analyze_requirements(req_text.strip())
        
        print(f"🆔 ID: {requirement.id}")
        print(f"📊 Type: {requirement.type}")
        print(f"⭐ Priority: {requirement.priority}/5")
        print(f"🔧 Complexity: {requirement.complexity_estimate:.1f}/10")
        print(f"⏱️  Estimated Effort: {requirement.estimated_effort:.1f} points")
        
        if requirement.affected_components:
            print(f"🎯 Affected Components: {', '.join(requirement.affected_components)}")
        
        if requirement.dependencies:
            print(f"🔗 Dependencies: {', '.join(requirement.dependencies)}")
        
        if requirement.acceptance_criteria:
            print(f"✅ Acceptance Criteria:")
            for criterion in requirement.acceptance_criteria:
                print(f"   - {criterion}")
        
        if requirement.risk_factors:
            print(f"⚠️  Risk Factors:")
            for risk in requirement.risk_factors:
                print(f"   - {risk}")
        
        if requirement.suggested_approach:
            print(f"💡 Suggested Approach: {requirement.suggested_approach}")


async def demo_requirement_mapping(engine, analysis_result):
    """Demonstrate requirement to code mapping."""
    print("\n" + "=" * 50)
    print("🗺️  Requirement-to-Code Mapping Demo")
    print("=" * 50)
    
    # Create a requirement related to the analyzed code
    from openevolve.analysis.core.interfaces import RequirementContext
    
    requirement = RequirementContext(
        id="perf_req_001",
        description="Optimize the GeometryCalculator to cache calculation results for better performance",
        type="enhancement",
        priority=4,
        affected_components=["GeometryCalculator", "calculation", "performance"],
        acceptance_criteria=[
            "Cache should store results of previous calculations",
            "Performance improvement of at least 50% for repeated calculations",
            "Cache should be configurable and thread-safe"
        ]
    )
    
    print("📋 Mapping requirement to existing code...")
    print(f"Requirement: {requirement.description}")
    
    # Map requirement to code
    mapping_result = await engine.map_requirements_to_code(
        requirement, analysis_result.code_contexts
    )
    
    print(f"\n📊 Mapping Results:")
    print(f"   - Total mappings found: {mapping_result['total_mappings']}")
    print(f"   - High relevance mappings: {mapping_result['high_relevance_count']}")
    print(f"   - Coverage score: {mapping_result['coverage_score']:.2f}")
    
    if mapping_result['mappings']:
        print(f"\n🎯 Top Relevant Code Sections:")
        for mapping in mapping_result['mappings'][:3]:
            print(f"   - File: {mapping['file_path']}")
            print(f"     Relevance: {mapping['relevance_score']:.2f}")
            
            if mapping['matching_elements']:
                print(f"     Matching elements:")
                for element in mapping['matching_elements'][:3]:
                    print(f"       • {element['type']}: {element['name']} (line {element['line']})")
            
            if mapping['suggested_changes']:
                print(f"     Suggested changes:")
                for change in mapping['suggested_changes'][:2]:
                    print(f"       • {change}")


async def demo_impact_analysis(engine):
    """Demonstrate impact analysis capabilities."""
    print("\n" + "=" * 50)
    print("📊 Impact Analysis Demo")
    print("=" * 50)
    
    # Simulate proposed changes
    proposed_changes = [
        "Modify GeometryCalculator.__init__ to add caching functionality",
        "Update calculate_circle_area method to check cache before calculation",
        "Add new cache management methods to GeometryCalculator class",
        "Update unit tests to verify caching behavior"
    ]
    
    print("🔍 Analyzing impact of proposed changes...")
    for change in proposed_changes:
        print(f"   - {change}")
    
    # Create a simple dependency graph for demo
    from openevolve.analysis.core.interfaces import DependencyGraph, DependencyNode, DependencyEdge
    
    graph = DependencyGraph()
    
    # Add nodes
    nodes = [
        DependencyNode("calc_init", "GeometryCalculator.__init__", "method", "geometry_calculator.py", 15),
        DependencyNode("calc_circle", "calculate_circle_area", "method", "geometry_calculator.py", 20),
        DependencyNode("calc_rect", "calculate_rectangle_area", "method", "geometry_calculator.py", 30),
        DependencyNode("batch_calc", "batch_calculate_areas", "function", "geometry_calculator.py", 60)
    ]
    
    for node in nodes:
        graph.add_node(node)
    
    # Add edges
    edges = [
        DependencyEdge("batch_calc", "calc_circle", "calls"),
        DependencyEdge("batch_calc", "calc_rect", "calls"),
        DependencyEdge("calc_circle", "calc_init", "uses"),
        DependencyEdge("calc_rect", "calc_init", "uses")
    ]
    
    for edge in edges:
        graph.add_edge(edge)
    
    # Perform impact analysis
    impact = await engine.recommendation_engine.analyze_impact(proposed_changes, graph)
    
    print(f"\n📈 Impact Analysis Results:")
    print(f"   - Risk Level: {impact.risk_level}")
    print(f"   - Confidence: {impact.confidence_score:.2f}")
    print(f"   - Affected Files: {len(impact.affected_files)}")
    print(f"   - Affected Functions: {len(impact.affected_functions)}")
    
    if impact.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in impact.recommendations:
            print(f"   - {rec}")
    
    if impact.test_suggestions:
        print(f"\n🧪 Test Suggestions:")
        for suggestion in impact.test_suggestions:
            print(f"   - {suggestion}")


async def main():
    """Main demo function."""
    try:
        # Run code analysis demo
        analysis_result = await demo_code_analysis()
        
        # Create engine for other demos
        config = AnalysisConfig(enable_caching=False)
        engine = ContextAnalysisEngine(config)
        
        # Run requirement analysis demo
        await demo_requirement_analysis(engine)
        
        # Run requirement mapping demo
        await demo_requirement_mapping(engine, analysis_result)
        
        # Run impact analysis demo
        await demo_impact_analysis(engine)
        
        print("\n" + "=" * 50)
        print("✅ Context Analysis Engine Demo Completed!")
        print("=" * 50)
        
        print("\n📚 Summary of Capabilities Demonstrated:")
        print("   ✓ Semantic code analysis and element extraction")
        print("   ✓ Complexity and quality metrics calculation")
        print("   ✓ Code pattern recognition")
        print("   ✓ Dependency graph construction")
        print("   ✓ Intelligent recommendation generation")
        print("   ✓ Natural language requirement processing")
        print("   ✓ Requirement-to-code mapping")
        print("   ✓ Impact analysis for proposed changes")
        
        print("\n🚀 The Context Analysis Engine is ready for integration!")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

