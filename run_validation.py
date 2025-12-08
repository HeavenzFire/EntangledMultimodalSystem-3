"""
Validation Script for Quantum Unity Integration

This script runs the validation process for the Quantum Unity Integration system,
including testing and preparing for GitHub push.
"""

import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Validation")


def run_tests():
    """
    Run the tests for the Quantum Unity Integration system.
    
    Returns:
        bool: True if the tests passed, False otherwise
    """
    logger.info("Running tests for the Quantum Unity Integration system")
    
    try:
        # Run the test script
        result = subprocess.run([sys.executable, "test_quantum_unity.py"], check=True)
        
        # Check if the tests passed
        if result.returncode == 0:
            logger.info("Tests passed successfully")
            return True
        else:
            logger.error("Tests failed")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running tests: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running tests: {e}")
        return False


def run_visualization_demo():
    """
    Run the visualization demo for the Quantum Unity Integration system.
    
    Returns:
        bool: True if the demo ran successfully, False otherwise
    """
    logger.info("Running visualization demo for the Quantum Unity Integration system")
    
    try:
        # Run the quantum_unity_integration.py script
        result = subprocess.run([sys.executable, "quantum_unity_integration.py"], check=True)
        
        # Check if the demo ran successfully
        if result.returncode == 0:
            logger.info("Visualization demo ran successfully")
            return True
        else:
            logger.error("Visualization demo failed")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running visualization demo: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running visualization demo: {e}")
        return False


def prepare_github_push(repo_name, repo_url=None, branch="main", commit_message=None):
    """
    Prepare for pushing the Quantum Unity Integration system to GitHub.
    
    Args:
        repo_name: The name of the GitHub repository
        repo_url: The URL of the GitHub repository (if None, will be constructed from repo_name)
        branch: The branch to push to
        commit_message: The commit message
        
    Returns:
        bool: True if the preparation was successful, False otherwise
    """
    logger.info(f"Preparing to push to GitHub repository: {repo_name}")
    
    try:
        # Build the command
        command = [sys.executable, "github_push.py", "--repo", repo_name]
        
        if repo_url:
            command.extend(["--url", repo_url])
        
        command.extend(["--branch", branch])
        
        if commit_message:
            command.extend(["--message", commit_message])
        
        # Run the GitHub push script
        result = subprocess.run(command, check=True)
        
        # Check if the preparation was successful
        if result.returncode == 0:
            logger.info("GitHub push preparation successful")
            return True
        else:
            logger.error("GitHub push preparation failed")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Error preparing GitHub push: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error preparing GitHub push: {e}")
        return False


def main():
    """Main function to run the validation process."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the validation process for the Quantum Unity Integration system")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running the tests")
    parser.add_argument("--skip-demo", action="store_true", help="Skip running the visualization demo")
    parser.add_argument("--push", action="store_true", help="Prepare for pushing to GitHub")
    parser.add_argument("--repo", help="The name of the GitHub repository (e.g., 'username/repo')")
    parser.add_argument("--url", help="The URL of the GitHub repository (if None, will be constructed from repo)")
    parser.add_argument("--branch", default="main", help="The branch to push to")
    parser.add_argument("--message", help="The commit message")
    args = parser.parse_args()
    
    # Run the tests
    tests_passed = True
    if not args.skip_tests:
        tests_passed = run_tests()
    
    # Run the visualization demo
    demo_successful = True
    if not args.skip_demo and tests_passed:
        demo_successful = run_visualization_demo()
    
    # Prepare for pushing to GitHub
    push_prepared = True
    if args.push and tests_passed and demo_successful:
        if not args.repo:
            logger.error("Repository name is required for GitHub push")
            push_prepared = False
        else:
            push_prepared = prepare_github_push(
                args.repo,
                args.url,
                args.branch,
                args.message
            )
    
    # Check if the validation process was successful
    if tests_passed and demo_successful and push_prepared:
        logger.info("Validation process completed successfully")
        return True
    else:
        logger.error("Validation process failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 