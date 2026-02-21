import sys
import unittest
from unittest.mock import patch

from lmms_eval import __main__ as main_module


class TestCliParseArgs(unittest.TestCase):
    def test_max_tokens_arg_is_parsed(self):
        argv = [
            "lmms-eval",
            "--model",
            "openai",
            "--tasks",
            "mme",
            "--max_tokens",
            "12345",
        ]
        with patch.object(sys, "argv", argv):
            _, args = main_module.parse_eval_args()
        self.assertEqual(args.max_tokens, 12345)

    def test_max_tokens_defaults_to_none(self):
        argv = [
            "lmms-eval",
            "--model",
            "openai",
            "--tasks",
            "mme",
        ]
        with patch.object(sys, "argv", argv):
            _, args = main_module.parse_eval_args()
        self.assertIsNone(args.max_tokens)


if __name__ == "__main__":
    unittest.main()
