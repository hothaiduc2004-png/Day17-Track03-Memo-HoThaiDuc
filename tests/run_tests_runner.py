import unittest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

loader = unittest.TestLoader()
start_dir = str(Path(__file__).resolve().parents[1])
suite = loader.discover(start_dir, pattern='test_baseline_and_memory.py')

runner = unittest.TextTestRunner(verbosity=2)
res = runner.run(suite)
if res.wasSuccessful():
    print('ALL_TESTS_PASSED')
    sys.exit(0)
else:
    print('SOME_TESTS_FAILED')
    sys.exit(2)
