import os
import re
import sys
from tempfile import gettempdir
from uuid import uuid4
import tomli


def test_version():
    with open("pyproject.toml", "rb") as f:
        pyproject_toml = tomli.load(f)
    with open("Cargo.toml", "rb") as f:
        cargo_toml = tomli.load(f)
    assert cargo_toml["package"]["version"] == pyproject_toml["project"]["version"]


def test_readme():
    files_under_test = ["../README.md", "README-pypi.md"]
    code_block_start = "```python"
    code_block_end = "```"
    code_block_regex = re.compile(
        f"{code_block_start}(.*?){code_block_end}", flags=re.DOTALL
    )
    for filename in files_under_test:
        with open(filename, "r") as f:
            content = f.read()
        codeblocks = code_block_regex.findall(content)
        codeblocks = "\n".join(codeblocks)
        tmpfile = f"{gettempdir()}/{uuid4().hex}.py"
        exit_code = None
        try:
            with open(tmpfile, "w") as f:
                f.write(codeblocks)
            exit_code = os.system(f"{sys.executable} {tmpfile}")
        finally:
            os.remove(tmpfile)
        assert exit_code == 0


if __name__ == "__main__":
    test_readme()
