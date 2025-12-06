import unittest
import importlib

class TestSmoke(unittest.TestCase):
    def test_imports(self):
        """Ensure main modules import without errors."""
        self.assertIsNotNone(importlib.import_module("src.preprocessing"))
        self.assertIsNotNone(importlib.import_module("src.model"))

    def test_dummy(self):
        """A basic sanity test."""
        self.assertEqual(2 + 3, 5)

if __name__ == "__main__":
    unittest.main()
