import unittest
from pathlib import Path

__PROBLEM_INPUT__ = [[]]
__PROBLEM_OUTPUT__ = [[]]

__TASK_FUNC__ = lambda: NotImplementedError()

__OUTPUT_FILE__ = ""


class TestCases(unittest.TestCase):
    def test_transform(self):
        pred = __TASK_FUNC__(__PROBLEM_INPUT__)
        if __OUTPUT_FILE__:
            Path(__OUTPUT_FILE__).write_text(repr(pred))

        self.assertEqual(pred, __PROBLEM_OUTPUT__)
