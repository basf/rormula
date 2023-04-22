import tomli

def test_version():
    with open("pyproject.toml", "rb") as f:
        pyproject_toml = tomli.load(f)
    with open("Cargo.toml", "rb") as f:
        cargo_toml = tomli.load(f)
    assert cargo_toml["package"]["version"] == pyproject_toml["project"]["version"]