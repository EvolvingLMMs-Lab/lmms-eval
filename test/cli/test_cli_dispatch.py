"""Tests for the unified CLI dispatch router and subcommand parsers.

Covers:
- _is_legacy_invocation() detection logic
- _is_eval_wizard() detection logic
- main() routing: banner, help, subcommand dispatch, legacy backward compat
- Subcommand parsers: models (--aliases gating), tasks (action choices),
  version (output structure)
- models_cmd._col() helper
"""

import argparse
import sys
import unittest
from io import StringIO
from unittest.mock import patch

from lmms_eval.cli.dispatch import _is_eval_wizard, _is_legacy_invocation
from lmms_eval.cli.models_cmd import _col

# ===========================================================================
# _is_legacy_invocation
# ===========================================================================


class TestIsLegacyInvocation(unittest.TestCase):
    """Pure function — no side effects, parametrize edge cases."""

    def test_empty_argv_is_not_legacy(self):
        self.assertFalse(_is_legacy_invocation([]))

    def test_flag_starting_with_dash_is_legacy(self):
        self.assertTrue(_is_legacy_invocation(["--model", "openai", "--tasks", "mme"]))

    def test_single_flag_is_legacy(self):
        self.assertTrue(_is_legacy_invocation(["--help"]))

    def test_short_flag_is_legacy(self):
        self.assertTrue(_is_legacy_invocation(["-h"]))

    def test_subcommand_is_not_legacy(self):
        self.assertFalse(_is_legacy_invocation(["tasks"]))

    def test_unknown_bare_word_is_not_legacy(self):
        self.assertFalse(_is_legacy_invocation(["foobar"]))

    def test_eval_subcommand_is_not_legacy(self):
        self.assertFalse(_is_legacy_invocation(["eval", "--model", "openai"]))


# ===========================================================================
# _is_eval_wizard
# ===========================================================================


class TestIsEvalWizard(unittest.TestCase):
    """Pure function — detect wizard mode."""

    def test_empty_argv_is_not_wizard(self):
        self.assertFalse(_is_eval_wizard([]))

    def test_eval_alone_is_wizard(self):
        self.assertTrue(_is_eval_wizard(["eval"]))

    def test_eval_with_flags_is_not_wizard(self):
        self.assertFalse(_is_eval_wizard(["eval", "--model", "openai"]))

    def test_non_eval_subcommand_is_not_wizard(self):
        self.assertFalse(_is_eval_wizard(["tasks"]))


# ===========================================================================
# _col helper
# ===========================================================================


class TestColHelper(unittest.TestCase):
    def test_short_text_is_padded(self):
        result = _col("hi", 10)
        self.assertEqual(len(result), 10)
        self.assertTrue(result.startswith("hi"))

    def test_long_text_is_truncated(self):
        result = _col("abcdefghij", 5)
        self.assertEqual(result, "abcde")
        self.assertEqual(len(result), 5)

    def test_exact_width(self):
        result = _col("abc", 3)
        self.assertEqual(result, "abc")


# ===========================================================================
# main() routing — banner
# ===========================================================================


class TestMainBanner(unittest.TestCase):
    def test_no_args_prints_banner_and_exits(self):
        from lmms_eval.cli.dispatch import main

        with patch.object(sys, "argv", ["lmms-eval"]):
            with self.assertRaises(SystemExit) as ctx:
                with patch("sys.stdout", new_callable=StringIO) as mock_out:
                    main()
            self.assertEqual(ctx.exception.code, 0)

    def test_help_flag_prints_usage_and_exits(self):
        from lmms_eval.cli.dispatch import main

        with patch.object(sys, "argv", ["lmms-eval", "--help"]):
            with self.assertRaises(SystemExit) as ctx:
                with patch("sys.stdout", new_callable=StringIO):
                    main()
            self.assertEqual(ctx.exception.code, 0)


# ===========================================================================
# Subcommand parsers — models
# ===========================================================================


class TestModelsSubcommand(unittest.TestCase):
    def _run_models(self, aliases=False):
        """Run models subcommand and capture stdout."""
        from lmms_eval.cli.models_cmd import run_models

        ns = argparse.Namespace(aliases=aliases)
        buf = StringIO()
        with patch("sys.stdout", buf):
            run_models(ns)
        return buf.getvalue()

    def test_models_without_aliases_has_no_aliases_column(self):
        output = self._run_models(aliases=False)
        lines = output.strip().splitlines()
        header = lines[2]
        self.assertNotIn("Aliases", header)

    def test_models_with_aliases_has_aliases_column(self):
        output = self._run_models(aliases=True)
        lines = output.strip().splitlines()
        header = lines[2]
        self.assertIn("Aliases", header)

    def test_models_output_contains_known_model(self):
        output = self._run_models(aliases=False)
        self.assertIn("qwen2_5_vl", output)
        self.assertIn("openai", output)

    def test_models_counts_are_consistent(self):
        output = self._run_models(aliases=False)
        for line in output.splitlines():
            if "total" in line and "Registered" in line:
                import re

                match = re.search(r"\((\d+) total\)", line)
                self.assertIsNotNone(match)
                total = int(match.group(1))
                self.assertGreater(total, 0)
                break
        else:
            self.fail("No 'Registered Models (N total)' line found")


# ===========================================================================
# Subcommand parsers — version
# ===========================================================================


class TestVersionSubcommand(unittest.TestCase):
    def test_version_output_has_version_string(self):
        from lmms_eval.cli.version_cmd import run_version

        buf = StringIO()
        with patch("sys.stdout", buf):
            run_version(argparse.Namespace())
        output = buf.getvalue()
        self.assertIn("lmms-eval", output)
        self.assertIn("Python", output)

    def test_version_output_has_torch(self):
        from lmms_eval.cli.version_cmd import run_version

        buf = StringIO()
        with patch("sys.stdout", buf):
            run_version(argparse.Namespace())
        output = buf.getvalue()
        self.assertIn("torch", output)


# ===========================================================================
# Subcommand parsers — tasks
# ===========================================================================


class TestTasksParser(unittest.TestCase):
    def test_tasks_parser_accepts_valid_actions(self):
        from lmms_eval.cli.tasks_cmd import add_tasks_parser

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="subcommand")
        add_tasks_parser(sub)

        for action in ("list", "groups", "subtasks", "tags"):
            args = parser.parse_args(["tasks", action])
            self.assertEqual(args.action, action)

    def test_tasks_parser_defaults_to_list(self):
        from lmms_eval.cli.tasks_cmd import add_tasks_parser

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="subcommand")
        add_tasks_parser(sub)

        args = parser.parse_args(["tasks"])
        self.assertEqual(args.action, "list")

    def test_tasks_parser_rejects_invalid_action(self):
        from lmms_eval.cli.tasks_cmd import add_tasks_parser

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="subcommand")
        add_tasks_parser(sub)

        with self.assertRaises(SystemExit):
            parser.parse_args(["tasks", "invalid_action"])


# ===========================================================================
# Legacy backward compat routing
# ===========================================================================


class TestLegacyBackwardCompat(unittest.TestCase):
    def test_legacy_flags_detected_correctly(self):
        """--model X --tasks Y should be detected as legacy invocation."""
        self.assertTrue(_is_legacy_invocation(["--model", "openai", "--tasks", "mme"]))

    def test_eval_prefix_with_flags_not_detected_as_legacy(self):
        """'eval --model X' starts with 'eval', not '-', so not legacy."""
        self.assertFalse(_is_legacy_invocation(["eval", "--model", "openai"]))

    def test_tasks_list_is_legacy(self):
        """--tasks list is legacy (starts with '-')."""
        self.assertTrue(_is_legacy_invocation(["--tasks", "list"]))


if __name__ == "__main__":
    unittest.main()
