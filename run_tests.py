#!/usr/bin/env python3
"""
Test runner for GraphYML.
Runs all unit tests and reports results.
"""
import unittest
import sys
import os
from pathlib import Path


def run_tests():
    """Run all unit tests."""
    # Add the project root to the Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(project_root, "tests")
    suite = loader.discover(start_dir, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())

