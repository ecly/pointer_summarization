"""
Copyright (c) 2019 Emil Lynegaard
Distributed under the MIT software license, see the
accompanying LICENSE.md or https://opensource.org/licenses/MIT

Some basic tests for the edit distance computation in the `calc_stats` module.

Example: `python -m tools.calc_stats_tests`
"""
import unittest
from .calc_stats import edit_distance


class TestEditDistance(unittest.TestCase):
    """Unit test the edit_distance module"""

    def test_edit_distance_dels(self):
        """Ensure deletes are free"""

        source1 = "it was a good day".split()
        target1 = "good day".split()
        distance1 = edit_distance(source1, target1)
        self.assertEqual(0, distance1)

        source2 = "the weather is sunny today".split()
        target2 = "the weather".split()
        distance2 = edit_distance(source2, target2)
        self.assertEqual(0, distance2)

        source3 = "the sidewalk is wet and slippery".split()
        target3 = "sidewalk is wet".split()
        distance3 = edit_distance(source3, target3)
        self.assertEqual(0, distance3)

    def test_edit_distance_subs(self):
        """Ensure that subs cost 1"""

        source1 = "it was a day that was good".split()
        target1 = "it was a day that was bad".split()
        distance1 = edit_distance(source1, target1)
        self.assertEqual(1, distance1)

        source2 = "pigs are pink".split()
        target2 = "rainbows are pink".split()
        distance2 = edit_distance(source2, target2)
        self.assertEqual(1, distance2)

        source2 = "it was a jolly christmas morning today".split()
        target2 = "it was a fuzzy christmas evening yesterday".split()
        distance2 = edit_distance(source2, target2)
        self.assertEqual(3, distance2)

    def test_edit_distance_best_alignment(self):
        """Ensure that best alignment is chosen"""

        source1 = "it was a day that was really good today".split()
        target1 = "today was a day that was good".split()
        distance1 = edit_distance(source1, target1)
        self.assertEqual(1, distance1)


if __name__ == "__main__":
    unittest.main()
