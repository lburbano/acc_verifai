# %%
from pathlib import Path
from tomllib import load
from urllib.parse import unquote

from bs4 import BeautifulSoup
from invoke.context import Context
from invoke.tasks import task


# %%
def print_color(text: str, color: str = "yellow") -> None:
    """Print text in a color."""
    # ANSI color codes
    clear_color = "\033[0m"
    if color == "red":
        color_code = "\033[31m"
    elif color == "yellow":
        color_code = "\033[33m"
    else:
        color_code = ""
    print(f"{color_code}{text}{clear_color}")


def add_missing_pytest_files() -> None:
    """Check for the existence of a matching `_test.py` file for each module."""
    print_color("Checking for missing test files")
    root_dir: Path = Path(__file__).parent.parent
    test_dir: Path = root_dir / "tests"
    with (root_dir / "pyproject.toml").open("rb") as f:
        proj_name = load(f)["tool"]["poetry"]["name"].replace("-", "_")
    src_dir: Path = root_dir / proj_name
    for path in src_dir.rglob("*.py"):
        if path.stem == "__init__":
            continue
        test_path = test_dir / path.relative_to(src_dir)
        test_path = test_path.with_name(test_path.stem + "_test.py")

        if not test_path.exists():
            print(f"Missing test file: {test_path}")
            test_path.write_text(f"from {proj_name} import {path.stem}\n\n")


@task
def poetry(ctx: Context) -> None:
    """Check the poetry lock file."""
    print_color("Checking poetry lock file")
    ctx.run("poetry check --lock")


@task
def test(ctx: Context) -> None:
    """Check the tests with pytest."""
    print_color("Running pytest")
    add_missing_pytest_files()
    ctx.run('poetry run coverage run -m pytest -m "not skipci" --junitxml=test-results.xml')


@task
def coverage_report(ctx: Context) -> None:
    """Generate the coverage report."""
    print_color("Generating coverage report")
    ctx.run("poetry run coverage html -i")
    with Path("htmlcov/index.html").open() as f:
        soup = BeautifulSoup(f, "html.parser")
        table = soup.find("main", {"id": "index"}).find("table", {"class": "index"})
    table_html = unquote("## Coverage Summary\n\n" + str(table))
    with Path("htmlcov/index.md").open("w") as f:
        f.write(table_html)
    print("Coverage report generated at htmlcov/index.md")


@task
def lint(ctx: Context) -> None:
    """Check the code with ruff."""
    print_color("Linting with ruff.")
    ctx.run("poetry run ruff check")


@task
def ruff_format(ctx: Context) -> None:
    """Format the code with ruff."""
    print_color("Checking code formatting with ruff.")
    ctx.run("poetry run ruff format --check")


@task(
    pre=[
        poetry,
        ruff_format,
        lint,
        test,
        coverage_report,
    ],
    default=True,
)
def all_tests(_: Context) -> None:
    """Run all checks."""
