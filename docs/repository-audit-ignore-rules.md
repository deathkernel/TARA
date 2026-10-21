# Repository audit ignore behavior

TARA's dependency-free static audit reads root and nested `.gitignore` files so
generated or intentionally ignored Python files do not create false syntax
errors. It supports common patterns, including comments, negation, directory
rules, slash-containing patterns, and `*` / `?` wildcards.

This is a practical subset of Git's full ignore/pathspec rules; the audit does
not invoke Git or execute project code. Built-in exclusions (`.git`, `.venv`,
`venv`, and `__pycache__`) remain excluded regardless of ignore-file contents.
