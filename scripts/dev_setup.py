import os


def main():
    """
    Runs venv setup, packages installations, tools and precommit setup
    """

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    if os.system("uv python find 3.12") != 0:
        os.system("uv python install 3.12")

    os.system("uv sync")
    os.system("uv tool install ruff")
    os.system("uv tool install mypy")
    os.system("uv tool install pre-commit")
    os.system("uv tool run pre-commit install")
