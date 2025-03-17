import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.evaluation.compare_csv_targets import load_csv_files, compare_event_targets
import argparse
import os


def compare_and_visualize(
    file1: str,
    file2: str,
    model1_name: str,
    model2_name: str,
    output_dir: str | None = None,
):
    """
    Compare two model outputs and generate visualizations

    Args:
        file1: Path to first model's CSV output
        file2: Path to second model's CSV output
        model1_name: Name of first model for labeling
        model2_name: Name of second model for labeling
        output_dir: Directory to save outputs
    """
    df1, df2 = load_csv_files(file1, file2)
    results = compare_event_targets(df1, df2)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if output_dir:
        comparison_path = os.path.join(
            output_dir, f"comparison_{model1_name}_vs_{model2_name}.csv"
        )
        results.to_csv(comparison_path, index=False)

    # 1. Histogram of similarity scores
    plt.figure(figsize=(10, 6))
    sns.histplot(results["Similarity"], bins=20, kde=True)
    plt.title(f"Similarity Distribution: {model1_name} vs {model2_name}")
    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")
    if output_dir:
        plt.savefig(
            os.path.join(
                output_dir, f"similarity_histogram_{model1_name}_vs_{model2_name}.png"
            )
        )
    else:
        plt.show()

    # 2. Box plot of similarity by screenshot
    plt.figure(figsize=(14, 8))
    sns.boxplot(x="Screenshot", y="Similarity", data=results)
    plt.title(f"Similarity by Screenshot: {model1_name} vs {model2_name}")
    plt.xticks(rotation=90)
    plt.tight_layout()
    if output_dir:
        plt.savefig(
            os.path.join(
                output_dir,
                f"similarity_by_screenshot_{model1_name}_vs_{model2_name}.png",
            )
        )
    else:
        plt.show()

    summary = pd.DataFrame(
        {
            "Metric": [
                "Mean Similarity",
                "Median Similarity",
                "Min Similarity",
                "Max Similarity",
                "Similarity > 0.9",
                "Similarity > 0.7",
                "Similarity < 0.5",
            ],
            "Value": [
                results["Similarity"].mean(),
                results["Similarity"].median(),
                results["Similarity"].min(),
                results["Similarity"].max(),
                (results["Similarity"] > 0.9).mean(),
                (results["Similarity"] > 0.7).mean(),
                (results["Similarity"] < 0.5).mean(),
            ],
        }
    )

    if output_dir:
        summary.to_csv(
            os.path.join(output_dir, f"summary_{model1_name}_vs_{model2_name}.csv"),
            index=False,
        )

    print(f"Comparison Summary: {model1_name} vs {model2_name}")
    print(f"Total entries compared: {len(results)}")
    print(f"Mean similarity: {results['Similarity'].mean():.4f}")
    print(f"Median similarity: {results['Similarity'].median():.4f}")
    print(
        f"Percentage with high similarity (>0.9): {(results['Similarity'] > 0.9).mean() * 100:.2f}%"
    )
    print(
        f"Percentage with moderate similarity (>0.7): {(results['Similarity'] > 0.7).mean() * 100:.2f}%"
    )
    print(
        f"Percentage with low similarity (<0.5): {(results['Similarity'] < 0.5).mean() * 100:.2f}%"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare and visualize outputs from two different models"
    )
    parser.add_argument(
        "--file1", required=True, help="Path to the first model's CSV output"
    )
    parser.add_argument(
        "--file2", required=True, help="Path to the second model's CSV output"
    )
    parser.add_argument("--model1", required=True, help="Name of the first model")
    parser.add_argument("--model2", required=True, help="Name of the second model")
    parser.add_argument("--output-dir", help="Directory to save output files")

    args = parser.parse_args()

    compare_and_visualize(
        args.file1, args.file2, args.model1, args.model2, args.output_dir
    )


if __name__ == "__main__":
    main()
