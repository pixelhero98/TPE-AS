import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tpeas.search_space import CategoricalParam, FloatParam, IntParam, SearchSpace


class SearchSpaceTests(unittest.TestCase):
    def test_sampling_and_encoding_roundtrip(self):
        space = SearchSpace(
            [
                FloatParam("lr", 1e-4, 1e-1, log_scale=True),
                IntParam("depth", 1, 5),
                CategoricalParam("mode", ["a", "b"]),
            ]
        )
        rng = np.random.default_rng(7)
        sample = space.sample(rng)

        self.assertGreaterEqual(sample["lr"], 1e-4)
        self.assertLessEqual(sample["lr"], 1e-1)
        self.assertGreaterEqual(sample["depth"], 1)
        self.assertLessEqual(sample["depth"], 5)
        self.assertIn(sample["mode"], {"a", "b"})

        decoded = space.decode(space.encode(sample))
        self.assertEqual(decoded["depth"], sample["depth"])
        self.assertEqual(decoded["mode"], sample["mode"])
        self.assertAlmostEqual(decoded["lr"], sample["lr"])

    def test_duplicate_names_are_rejected(self):
        with self.assertRaises(ValueError):
            SearchSpace([FloatParam("x", 0.0, 1.0), IntParam("x", 1, 2)])


if __name__ == "__main__":
    unittest.main()

