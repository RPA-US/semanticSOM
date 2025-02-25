# SemanticSOM

## Table of Contents

- [Installation](#installation)
- [File Structure and Contents](#file-structure-and-contents)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

## GPU Prequisites

In order to use Nvidia GPU and CUDA to run your inferences you are required to be runnning the code in a windows or linux machine and have cuda 12.4 installed on your system

## Installation

1. **Install uv:**
The project uses the [uv](https://github.com/astral-sh/uv) CLI tool for environment management and package installations.
```sh
uv python install 3.10
```

2. **Run the Development Setup Script:**
Execute the dev-setup script to configure your environment and install required tools.
```sh
uv run dev-setup
```

3. **File Structure and Contents:**
```
.
├── .coveragerc              # Coverage configuration (omit config.py files)
├── .gitignore
├── .mypy_cache/             # mypy cache files
├── .pre-commit-config.yaml   # Pre-commit hooks configuration
├── pyproject.toml           # Project metadata and dependencies
├── README.md                # This file
├── requirements.txt         # Auto-generated requirements
├── scripts/                 # Utility scripts (e.g., [dev_setup.py](scripts/dev_setup.py))
├── src/                     # Source code
│   └── semantics/
│   └── utils/
│       └── [set_of_marks.py](src/utils/set_of_marks.py)  # Main logic for processing image marks
├── tests/                   # Unit tests
└── ...                      # Additional files and directories
```

## Acknowledgments

    Todo when paper is ready

## Citation

    Todo when paper is ready. For now:

@misc{SemanticSOM2025,
  author = {Rodríguez Ruiz, Antonio},
  title = {SemanticSOM},
  year = {2025},
  howpublished = {\url{https://github.com/RPA-US/SemanticSOM}},
}
