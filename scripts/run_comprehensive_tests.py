#!/usr/bin/env python3
"""
Comprehensive test runner for OpenEvolve autonomous development pipeline.

This script runs all tests, validates system integrity, and generates
comprehensive reports on system health and performance.
"""

import os
import sys
import subprocess
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestRunner:
    """Comprehensive test runner for the OpenEvolve system."""
    
    def __init__(self, verbose: bool = False, output_dir: str = "test_results"):
        """Initialize the test runner."""
        self.verbose = verbose
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {
            "timestamp": time.time(),
            "test_suites": {},
            "summary": {},
            "errors": []
        }
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] [{level}] {message}"
        print(formatted_message)
        
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(formatted_message)
    
    def run_command(self, command: List[str], cwd: Optional[str] = None) -> Dict[str, Any]:
        """Run a command and capture output."""
        self.log(f"Running command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                cwd=cwd or project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(command)
            }
        
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": "Command timed out after 5 minutes",
                "command": " ".join(command)
            }
        
        except Exception as e:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "command": " ".join(command)
            }
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests using pytest."""
        self.log("Running unit tests...")
        
        # Find all test files
        test_files = list(Path(project_root).rglob("test_*.py"))
        unit_test_files = [
            str(f) for f in test_files 
            if "integration" not in str(f) and "performance" not in str(f) and "validation" not in str(f)
        ]
        
        if not unit_test_files:
            return {
                "success": True,
                "tests_run": 0,
                "failures": 0,
                "errors": 0,
                "skipped": 0,
                "output": "No unit tests found"
            }
        
        # Run pytest with coverage
        command = [
            sys.executable, "-m", "pytest",
            "--verbose",
            "--tb=short",
            "--junit-xml=" + str(self.output_dir / "unit_tests.xml"),
            "--cov=.",
            "--cov-report=html:" + str(self.output_dir / "coverage_html"),
            "--cov-report=json:" + str(self.output_dir / "coverage.json")
        ] + unit_test_files
        
        result = self.run_command(command)
        
        # Parse pytest output for summary
        summary = self._parse_pytest_output(result["stdout"])
        
        return {
            "success": result["success"],
            "output": result["stdout"],
            "errors": result["stderr"],
            **summary
        }
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        self.log("Running integration tests...")
        
        integration_test_dir = Path(project_root) / "tests" / "integration"
        if not integration_test_dir.exists():
            return {
                "success": True,
                "tests_run": 0,
                "output": "No integration tests directory found"
            }
        
        command = [
            sys.executable, "-m", "pytest",
            str(integration_test_dir),
            "--verbose",
            "--tb=short",
            "--junit-xml=" + str(self.output_dir / "integration_tests.xml")
        ]
        
        result = self.run_command(command)
        summary = self._parse_pytest_output(result["stdout"])
        
        return {
            "success": result["success"],
            "output": result["stdout"],
            "errors": result["stderr"],
            **summary
        }
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        self.log("Running performance tests...")
        
        performance_test_dir = Path(project_root) / "tests" / "performance"
        if not performance_test_dir.exists():
            return {
                "success": True,
                "tests_run": 0,
                "output": "No performance tests directory found"
            }
        
        command = [
            sys.executable, "-m", "pytest",
            str(performance_test_dir),
            "--verbose",
            "--tb=short",
            "--junit-xml=" + str(self.output_dir / "performance_tests.xml")
        ]
        
        result = self.run_command(command)
        summary = self._parse_pytest_output(result["stdout"])
        
        return {
            "success": result["success"],
            "output": result["stdout"],
            "errors": result["stderr"],
            **summary
        }
    
    def run_validation_tests(self) -> Dict[str, Any]:
        """Run validation tests."""
        self.log("Running validation tests...")
        
        validation_test_dir = Path(project_root) / "tests" / "validation"
        if not validation_test_dir.exists():
            return {
                "success": True,
                "tests_run": 0,
                "output": "No validation tests directory found"
            }
        
        command = [
            sys.executable, "-m", "pytest",
            str(validation_test_dir),
            "--verbose",
            "--tb=short",
            "--junit-xml=" + str(self.output_dir / "validation_tests.xml")
        ]
        
        result = self.run_command(command)
        summary = self._parse_pytest_output(result["stdout"])
        
        return {
            "success": result["success"],
            "output": result["stdout"],
            "errors": result["stderr"],
            **summary
        }
    
    def run_code_quality_checks(self) -> Dict[str, Any]:
        """Run code quality checks."""
        self.log("Running code quality checks...")
        
        quality_results = {}
        
        # Run flake8 for style checking
        flake8_result = self.run_command([
            sys.executable, "-m", "flake8",
            "--max-line-length=120",
            "--ignore=E203,W503",
            "--output-file=" + str(self.output_dir / "flake8_report.txt"),
            "."
        ])
        quality_results["flake8"] = flake8_result
        
        # Run mypy for type checking
        mypy_result = self.run_command([
            sys.executable, "-m", "mypy",
            "--ignore-missing-imports",
            "--no-strict-optional",
            "."
        ])
        quality_results["mypy"] = mypy_result
        
        # Run bandit for security checks
        bandit_result = self.run_command([
            sys.executable, "-m", "bandit",
            "-r", ".",
            "-f", "json",
            "-o", str(self.output_dir / "bandit_report.json")
        ])
        quality_results["bandit"] = bandit_result
        
        # Calculate overall success
        overall_success = all(result["success"] for result in quality_results.values())
        
        return {
            "success": overall_success,
            "checks": quality_results,
            "summary": f"Code quality checks: {len([r for r in quality_results.values() if r['success']])}/{len(quality_results)} passed"
        }
    
    def run_dead_code_detection(self) -> Dict[str, Any]:
        """Run dead code detection."""
        self.log("Running dead code detection...")
        
        try:
            # Import and run the code quality validator
            from tests.validation.test_code_quality_validation import CodeQualityValidator
            
            validator = CodeQualityValidator(project_root)
            validator.analyze_codebase()
            
            dead_code = validator.find_dead_code()
            quality_metrics = validator.check_code_quality()
            
            # Save results
            with open(self.output_dir / "dead_code_report.json", "w") as f:
                json.dump({
                    "dead_code": dead_code,
                    "quality_metrics": quality_metrics
                }, f, indent=2)
            
            total_dead_items = (
                len(dead_code["unused_functions"]) +
                len(dead_code["unused_classes"])
            )
            
            return {
                "success": True,
                "dead_code_items": total_dead_items,
                "unused_functions": len(dead_code["unused_functions"]),
                "unused_classes": len(dead_code["unused_classes"]),
                "quality_issues": len(quality_metrics["issues"]),
                "summary": f"Found {total_dead_items} potentially dead code items"
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "summary": "Dead code detection failed"
            }
    
    def run_security_scan(self) -> Dict[str, Any]:
        """Run security scanning."""
        self.log("Running security scan...")
        
        security_results = {}
        
        # Run safety check for known vulnerabilities
        safety_result = self.run_command([
            sys.executable, "-m", "safety",
            "check",
            "--json",
            "--output", str(self.output_dir / "safety_report.json")
        ])
        security_results["safety"] = safety_result
        
        # Run pip-audit for dependency vulnerabilities
        pip_audit_result = self.run_command([
            sys.executable, "-m", "pip_audit",
            "--format=json",
            "--output=" + str(self.output_dir / "pip_audit_report.json")
        ])
        security_results["pip_audit"] = pip_audit_result
        
        overall_success = all(result["success"] for result in security_results.values())
        
        return {
            "success": overall_success,
            "scans": security_results,
            "summary": f"Security scans: {len([r for r in security_results.values() if r['success']])}/{len(security_results)} passed"
        }
    
    def _parse_pytest_output(self, output: str) -> Dict[str, Any]:
        """Parse pytest output to extract test statistics."""
        lines = output.split("\\n")
        
        # Look for the summary line
        summary_line = ""
        for line in lines:
            if "passed" in line or "failed" in line or "error" in line:
                if any(word in line for word in ["test", "passed", "failed", "error", "skipped"]):
                    summary_line = line
                    break
        
        # Default values
        result = {
            "tests_run": 0,
            "failures": 0,
            "errors": 0,
            "skipped": 0
        }
        
        if summary_line:
            # Parse numbers from summary line
            import re
            
            # Extract numbers followed by keywords
            passed_match = re.search(r"(\\d+)\\s+passed", summary_line)
            failed_match = re.search(r"(\\d+)\\s+failed", summary_line)
            error_match = re.search(r"(\\d+)\\s+error", summary_line)
            skipped_match = re.search(r"(\\d+)\\s+skipped", summary_line)
            
            if passed_match:
                result["tests_run"] += int(passed_match.group(1))
            if failed_match:
                result["failures"] = int(failed_match.group(1))
                result["tests_run"] += result["failures"]
            if error_match:
                result["errors"] = int(error_match.group(1))
                result["tests_run"] += result["errors"]
            if skipped_match:
                result["skipped"] = int(skipped_match.group(1))
        
        return result
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report."""
        self.log("Generating comprehensive test report...")
        
        # Calculate summary statistics
        total_tests = sum(
            suite.get("tests_run", 0) 
            for suite in self.results["test_suites"].values()
            if isinstance(suite, dict)
        )
        
        total_failures = sum(
            suite.get("failures", 0) + suite.get("errors", 0)
            for suite in self.results["test_suites"].values()
            if isinstance(suite, dict)
        )
        
        success_rate = ((total_tests - total_failures) / total_tests * 100) if total_tests > 0 else 0
        
        self.results["summary"] = {
            "total_tests": total_tests,
            "total_failures": total_failures,
            "success_rate": success_rate,
            "overall_success": total_failures == 0 and all(
                suite.get("success", False) 
                for suite in self.results["test_suites"].values()
                if isinstance(suite, dict)
            )
        }
        
        # Generate HTML report
        html_report = self._generate_html_report()
        
        # Save reports
        with open(self.output_dir / "test_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        with open(self.output_dir / "test_report.html", "w") as f:
            f.write(html_report)
        
        # Generate summary
        summary = f"""
OpenEvolve Test Results Summary
==============================

Total Tests Run: {total_tests}
Failures: {total_failures}
Success Rate: {success_rate:.1f}%
Overall Status: {'PASS' if self.results['summary']['overall_success'] else 'FAIL'}

Test Suites:
"""
        
        for suite_name, suite_results in self.results["test_suites"].items():
            if isinstance(suite_results, dict):
                status = "PASS" if suite_results.get("success", False) else "FAIL"
                tests = suite_results.get("tests_run", 0)
                failures = suite_results.get("failures", 0) + suite_results.get("errors", 0)
                summary += f"  {suite_name}: {status} ({tests} tests, {failures} failures)\\n"
        
        summary += f"\\nDetailed reports saved to: {self.output_dir}\\n"
        
        return summary
    
    def _generate_html_report(self) -> str:
        """Generate an HTML test report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>OpenEvolve Test Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .success {{ color: green; }}
        .failure {{ color: red; }}
        .warning {{ color: orange; }}
        .suite {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .suite-header {{ font-weight: bold; font-size: 18px; margin-bottom: 10px; }}
        .metric {{ margin: 5px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>OpenEvolve Test Results</h1>
        <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.results['timestamp']))}</p>
        <p class="{'success' if self.results['summary']['overall_success'] else 'failure'}">
            Overall Status: {'PASS' if self.results['summary']['overall_success'] else 'FAIL'}
        </p>
    </div>
    
    <div class="suite">
        <div class="suite-header">Summary</div>
        <div class="metric">Total Tests: {self.results['summary']['total_tests']}</div>
        <div class="metric">Failures: {self.results['summary']['total_failures']}</div>
        <div class="metric">Success Rate: {self.results['summary']['success_rate']:.1f}%</div>
    </div>
"""
        
        for suite_name, suite_results in self.results["test_suites"].items():
            if isinstance(suite_results, dict):
                status_class = "success" if suite_results.get("success", False) else "failure"
                html += f"""
    <div class="suite">
        <div class="suite-header {status_class}">{suite_name.replace('_', ' ').title()}</div>
        <div class="metric">Status: {'PASS' if suite_results.get('success', False) else 'FAIL'}</div>
        <div class="metric">Tests Run: {suite_results.get('tests_run', 0)}</div>
        <div class="metric">Failures: {suite_results.get('failures', 0)}</div>
        <div class="metric">Errors: {suite_results.get('errors', 0)}</div>
        <div class="metric">Skipped: {suite_results.get('skipped', 0)}</div>
"""
                
                if suite_results.get("summary"):
                    html += f'<div class="metric">Summary: {suite_results["summary"]}</div>'
                
                html += "</div>"
        
        html += """
</body>
</html>
"""
        return html
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites."""
        self.log("Starting comprehensive test suite...")
        
        # Run all test suites
        test_suites = [
            ("unit_tests", self.run_unit_tests),
            ("integration_tests", self.run_integration_tests),
            ("performance_tests", self.run_performance_tests),
            ("validation_tests", self.run_validation_tests),
            ("code_quality", self.run_code_quality_checks),
            ("dead_code_detection", self.run_dead_code_detection),
            ("security_scan", self.run_security_scan)
        ]
        
        for suite_name, suite_func in test_suites:
            try:
                self.log(f"Running {suite_name}...")
                result = suite_func()
                self.results["test_suites"][suite_name] = result
                
                if result.get("success"):
                    self.log(f"{suite_name} completed successfully", "INFO")
                else:
                    self.log(f"{suite_name} failed", "ERROR")
                    
            except Exception as e:
                self.log(f"Error running {suite_name}: {e}", "ERROR")
                self.results["test_suites"][suite_name] = {
                    "success": False,
                    "error": str(e)
                }
                self.results["errors"].append(f"{suite_name}: {e}")
        
        # Generate final report
        summary = self.generate_report()
        self.log("Test suite completed")
        
        return {
            "results": self.results,
            "summary": summary
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run comprehensive tests for OpenEvolve")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output-dir", "-o", default="test_results", help="Output directory for results")
    parser.add_argument("--suite", "-s", choices=[
        "unit", "integration", "performance", "validation", 
        "quality", "dead-code", "security", "all"
    ], default="all", help="Test suite to run")
    
    args = parser.parse_args()
    
    runner = TestRunner(verbose=args.verbose, output_dir=args.output_dir)
    
    if args.suite == "all":
        result = asyncio.run(runner.run_all_tests())
    else:
        # Run specific suite
        suite_map = {
            "unit": runner.run_unit_tests,
            "integration": runner.run_integration_tests,
            "performance": runner.run_performance_tests,
            "validation": runner.run_validation_tests,
            "quality": runner.run_code_quality_checks,
            "dead-code": runner.run_dead_code_detection,
            "security": runner.run_security_scan
        }
        
        if args.suite in suite_map:
            result = suite_map[args.suite]()
            runner.results["test_suites"][args.suite] = result
            summary = runner.generate_report()
            result = {"results": runner.results, "summary": summary}
        else:
            print(f"Unknown test suite: {args.suite}")
            sys.exit(1)
    
    # Print summary
    print(result["summary"])
    
    # Exit with appropriate code
    overall_success = result["results"]["summary"].get("overall_success", False)
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()

