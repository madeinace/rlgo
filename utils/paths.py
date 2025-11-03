from pathlib import Path


def get_project_root(current_path=None, marker_files=None):
    """
    Find the project root by looking for marker files/directories.
    """
    if marker_files is None:
        marker_files = [
            ".git",
            "pyproject.toml",
        ]

    if current_path is None:
        current_path = Path(__file__).resolve()
    else:
        current_path = Path(current_path).resolve()

    for parent in [current_path] + list(current_path.parents):
        if any((parent / marker).exists() for marker in marker_files):
            return parent

    # Fallback to current directory if no marker found
    return Path.cwd()


PROJECT_ROOT = get_project_root()
RUNS_DIR = PROJECT_ROOT / "runs"

if __name__ == "__main__":
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"RUNS_DIR: {RUNS_DIR}")
