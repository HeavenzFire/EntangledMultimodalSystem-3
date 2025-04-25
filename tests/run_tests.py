import pytest
import sys
import os
from datetime import datetime
from src.utils.logger import logger

def run_tests():
    """Run all tests and generate a comprehensive report."""
    try:
        # Create reports directory if it doesn't exist
        if not os.path.exists('reports'):
            os.makedirs('reports')
        
        # Generate timestamp for report filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f'reports/test_report_{timestamp}.html'
        
        # Run tests with pytest
        pytest_args = [
            'tests/',
            '--html=' + report_file,
            '--self-contained-html',
            '--verbose',
            '--cov=src',
            '--cov-report=html:reports/coverage',
            '--cov-report=term-missing'
        ]
        
        logger.info("Starting test execution...")
        exit_code = pytest.main(pytest_args)
        
        if exit_code == 0:
            logger.info("All tests passed successfully!")
            logger.info(f"Test report generated at: {report_file}")
            logger.info("Coverage report available in reports/coverage/")
        else:
            logger.error(f"Tests failed with exit code: {exit_code}")
            logger.info(f"Check the report at {report_file} for details")
        
        return exit_code
        
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests()) 