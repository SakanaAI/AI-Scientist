import unittest
from unittest import mock

from ai_scientist.utils import run_chktex, validate_idea_name


class ValidateIdeaNameTest(unittest.TestCase):
    def test_accepts_names_produced_by_the_prompt(self):
        for name in ("diffusion_model", "retrieval-augmented-generation", "idea123"):
            with self.subTest(name=name):
                self.assertEqual(validate_idea_name(name), name)

    def test_rejects_names_that_can_escape_the_results_directory(self):
        for name in ("../escape", "nested/idea", r"nested\idea", "/tmp/idea"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_idea_name(name)

    def test_rejects_shell_metacharacters_and_whitespace(self):
        for name in (
            "idea;touch-pwned",
            "$(touch-pwned)",
            "`touch-pwned`",
            "idea name",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_idea_name(name)

    def test_rejects_empty_and_non_string_names(self):
        for name in ("", None, 1):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_idea_name(name)


class RunChktexTest(unittest.TestCase):
    @mock.patch("ai_scientist.utils.subprocess.run")
    def test_passes_the_filename_as_an_argv_entry_without_a_shell(self, run):
        run.return_value.stdout = "warning"
        writeup_file = "results/idea;touch-pwned/latex/template.tex"

        output = run_chktex(writeup_file)

        self.assertEqual(output, "warning")
        run.assert_called_once_with(
            ["chktex", writeup_file, "-q", "-n2", "-n24", "-n13", "-n1"],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
