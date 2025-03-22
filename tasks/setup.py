# %%
import re
from pathlib import Path
from subprocess import run

from invoke.context import Context
from invoke.tasks import task


root_dir: Path = Path(__file__).parent.parent


# %%
def rename_pyproject_toml() -> str:
    """Asks the user for package details and overwrites the pyproject.toml file."""
    package_name = get_git_remote_package_name()
    # User inputs
    package_name = input(f"Package name [{package_name}]: ") or package_name
    package_name = package_name.strip()
    description = input("Description ['']: ") or ""
    description = description.strip()
    version = input("Version [0.1.0]: ") or "0.1.0"
    version = version.strip()
    author = input("Author [Vince Faller]: ") or "Vince Faller"
    author = author.strip()

    with (root_dir / "pyproject.toml").open("r") as f:
        txt = f.read()

    pattern = r'\[tool\.poetry\]\nname = "[^"]+"\nversion = "[^"]+"\ndescription = "[^"]*"\nauthors = \["[^"]+"\]\n'
    replace_str = re.search(pattern, txt).group(0)
    replace_with = f'[tool.poetry]\nname = "{package_name}"\nversion = "{version}"\ndescription = "{description}"\nauthors = ["{author}"]\n'
    print(f"Replacing\n{replace_str}\nwith\n{replace_with}")
    txt = txt.replace(replace_str, replace_with)
    with (root_dir / "pyproject.toml").open("w") as f:
        f.write(txt)
    return package_name


def get_git_remote_package_name() -> str:
    """Get the package name from the git remote."""
    r = run(["git", "config", "--local", "remote.origin.url"], check=True, capture_output=True)  # noqa: S603, S607
    remote_url = r.stdout.decode()
    package_name = remote_url.split("/")[-1].split(".")[0]
    return package_name


@task
def rename(ctx: Context) -> None:
    """Rename the package in all the spots."""
    with (root_dir / "pyproject.toml").open("r") as f:
        tomltxt = f.read()

    if "python-template" not in tomltxt:
        print("Package name already set in pyproject.toml")
        return

    package_name = rename_pyproject_toml()
    package_name = package_name.replace("-", "_")
    # rename the package in the __init__.py file
    with root_dir / "tests/main_test.py" as f:
        txt = f.read_text()
    txt = txt.replace("python_template", package_name)
    with root_dir / "tests/main_test.py" as f:
        f.write_text(txt)

    # rename python_template to the new package name
    cmd = f"git mv -f python_template {package_name}"
    ctx.run(cmd)


@task
def precommit(ctx: Context) -> None:
    """Install pre-commit hooks."""
    ctx.run("poetry run pre-commit install")


@task
def update_from_template(ctx: Context) -> None:
    """Update the project from the template."""
    # check if remote is already added
    try:
        ctx.run("git remote get-url template", hide=True)
    except Exception:
        print("Adding remote template")
        ctx.run("git remote add template https://github.com/oliverc1623/luau.git")

    ctx.run("git fetch template", hide=True)
    ctx.run("git merge template/main --allow-unrelated-histories")


@task(
    pre=[
        precommit,
        rename,
    ],
    default=True,
)
def all_tasks(_: Context) -> None:
    """Run all setup tasks."""
