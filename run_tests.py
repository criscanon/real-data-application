import unittest
import coverage

def run_tests():
    """
    Run all unit tests in the 'tests' directory using the unittest framework with coverage.

    Returns:
        Coverage object: The coverage object after running the tests.
    """
    # Start code coverage
    cov = coverage.Coverage()
    cov.start()

    # Load all tests from the 'tests' directory
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests')

    # Run the tests
    result = unittest.TextTestRunner().run(test_suite)

    # Stop code coverage
    cov.stop()

    # Output the result of the tests
    if result.wasSuccessful():
        print("All tests passed!")
    else:
        print(f"{len(result.errors)} errors and {len(result.failures)} failures occurred.")

    # Generate and print coverage report
    cov.report()

    return cov

if __name__ == '__main__':
    # Run the tests and store the coverage object
    coverage_object = run_tests()
