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
├── .pre-commit-config.yaml  # Pre-commit hooks configuration
├── pyproject.toml           # Project metadata and dependencies
├── README.md                # This file
├── requirements.txt         # Auto-generated requirements
├── scripts/                 # Utility scripts (e.g., [dev_setup.py](scripts/dev_setup.py))
├── src/                     # Source code
│   └── [cfg.py](src/cfg.py) # Configuration file for the project
│   └── evaluation/
│       └── [setup.py](src/evaluation/setup.py)  # Preprocess given dataset and launches labeling program
│   └── mllm_judge/ # Implementation of mllm judge based on original author's code
│       └── [acknowledgements.md](src/mllm_judge/acknowledgements.md)  # Acknowledgements to original code and authors
│       └── [api_benchmarks.py](src/mllm_judge/api_benchmarks.py)  # API benchmarks
│       └── [get_vlm_res.py](src/mllm_judge/get_vlm_res.py)  # Implementation of calling to different vlm models
│       └── [prompt.py](src/mllm_judge/prompt.py)  # Available prompts when running the judge. It corresponds to judge modes described in the paper
│   └── models/
│       └── [models.py](src/models/models.py)  # LLM and VLM model abstractions with support for local and OpenAI inferences
│   └── semantics/
│       └── [activity_semantics.py](src/semantics/activity_semantics.py)  # Infer Activity Lables from enriched log with event semantics
│       └── [event_semantics.py](src/semantics/event_semantics.py)  # Infer Event descriptions from enriched log with target objects semantics
│       └── [object_semantics.py](src/semantics/target_object.py)  # Infer Target Object descriptions from Image, SOM and coords
│       └── [prompts.py](src/semantics/prompts.py)  # Prompts for the different techniques and use cases of vlms and llms in the framework
│   └── utils/
│       └── [hierarchy_constructor.py](src/utils/hierarchy_constructor.py)  # Converts lableme annotations to a Screen Object Model
│       └── [images.py](src/utils/images.py)  # Utils for image processing
│       └── [prompt_processing.py](src/utils/prompt_processing.py)  # Process prompts for the different techniques and use cases of vlms and llms in the framework
│       └── [set_of_marks.py](src/utils/set_of_marks.py)  # Set of Marks prompting technique implementation
├── tests/                   # Unit tests
└── ...                      # Additional files and directories
```

## Acknowledgments

    Todo when paper is ready

## Citation

    Todo when paper is ready. For now:

    @misc{SemanticSOM2025,
      author = {A. Roddriguez-Ruiz, A. Martínez-Rojas, J.G. Enríquez, A. Jímenez-Ramírez, and S. Agostinelli},
      title = {SemanticSOM},
      year = {2025},
      howpublished = {\url{https://github.com/RPA-US/SemanticSOM}},
    }
