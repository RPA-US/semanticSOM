# This script is now a simple wrapper around the more general analyze_model_performance.py script

import os
import sys
import subprocess


def main():
    """Run the error count analysis using the analyze_model_performance.py script"""
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the analyze_model_performance.py script
    analysis_script = os.path.join(current_dir, "analyze_model_performance.py")

    # Run the analysis script with the error_count metric
    cmd = [sys.executable, analysis_script, "--metric", "error_count"]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
