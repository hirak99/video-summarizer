import unittest

from . import templater

# pyright: reportPrivateUsage=false


class TestTemplater(unittest.TestCase):

    def test_successful_fill(self):
        template = ["Hello, {name}!", "Your age is {age}.", "Welcome to {city}."]
        prompt_args = {"name": "Alice", "age": "30", "city": "New York"}
        expected_lines = ["Hello, Alice!", "Your age is 30.", "Welcome to New York."]
        self.assertEqual(templater.fill(template, prompt_args), expected_lines)

    def test_no_variables_in_template(self):
        template = ["This is a plain line.", "Another line without variables."]
        prompt_args = {}
        expected_lines = ["This is a plain line.", "Another line without variables."]
        self.assertEqual(templater.fill(template, prompt_args), expected_lines)

    def test_missing_arg_in_prompt_args(self):
        template = ["Hello, {name}!", "Your age is {age}."]
        prompt_args = {"name": "Alice"}
        with self.assertRaisesRegex(
            templater.LeftoverArgs, "Args still remain after replacement"
        ):
            templater.fill(template, prompt_args)

    def test_unused_keys(self):
        template = ["Hello, {name}!"]
        prompt_args = {"name": "Alice", "city": "New York"}
        with self.assertRaisesRegex(templater.UnusedArgs, "Unused keys: {'city'}"):
            templater.fill(template, prompt_args)

    def test_multiple_occurrences_of_same_key(self):
        template = ["Hello, {name}! Your name is {name}."]
        prompt_args = {"name": "Bob"}
        expected_lines = ["Hello, Bob! Your name is Bob."]
        self.assertEqual(templater.fill(template, prompt_args), expected_lines)

    def test_split_double_brace(self):
        splitted = templater._split_double_brace("This is a {{test}}")
        self.assertEqual(
            splitted,
            [
                "This is a ",
                templater._DoubleBrace.OPEN,
                "test",
                templater._DoubleBrace.CLOSE,
                "",
            ],
        )

    def test_split_and_join_double_brace(self):
        cases = [
            ("This is a {{test}}", "This is a {test}"),
            ("Only open {{", "Only open {"),
            ("No double braces", "No double braces"),
            ("Only close }}", "Only close }"),
            ("Single open {", "Single open {"),
            ("Single close }", "Single close }"),
        ]
        for template, expected in cases:
            splitted = templater._split_double_brace(template)
            self.assertEqual(templater._join_double_brace(splitted), expected)

    def test_double_brace(self):
        template = ["Hello, {name}!", "Your age is {age}.", "BIG {{WELCOME}} TO YOU"]
        prompt_args = {"name": "Alice", "age": "30"}
        expected_lines = ["Hello, Alice!", "Your age is 30.", "BIG {WELCOME} TO YOU"]
        self.assertEqual(templater.fill(template, prompt_args), expected_lines)


if __name__ == "__main__":
    unittest.main()
