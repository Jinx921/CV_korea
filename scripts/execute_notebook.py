import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    project_root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name="ai-cv",
        resources={"metadata": {"path": str(project_root)}},
    )
    client.execute()
    nbformat.write(notebook, notebook_path)
    print(notebook_path)


if __name__ == "__main__":
    main()

