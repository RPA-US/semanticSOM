import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob


def configure_plot_style():
    """Configure matplotlib to use LaTeX fonts and styling"""
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "axes.labelsize": 16,
            "font.size": 16,
            "legend.fontsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )


def load_and_process_file(file_path):
    """
    Load and process CSV file to calculate average similarity scores by number of events

    Args:
        file_path: Path to the CSV file

    Returns:
        Dictionary with average scores by number of events
    """
    try:
        # Load CSV file
        df = pd.read_csv(file_path)

        # Check if required columns exist
        if "ScreenID" not in df.columns or "SBERT Similarity" not in df.columns:
            print(f"Required columns not found in {file_path}")
            return {}

        # Group by ScreenID and count number of events per screen
        screen_counts = df.groupby("ScreenID").size().reset_index(name="ActivityCount")

        # Calculate average similarity score for each ScreenID
        screen_scores = (
            df.groupby("ScreenID")["SBERT Similarity"]
            .mean()
            .reset_index(name="AvgSimilarity")
        )

        # Merge counts and scores
        screen_data = pd.merge(screen_counts, screen_scores, on="ScreenID")

        # Group by number of events and calculate average similarity score
        # This gives us the average similarity score for screens with N events
        event_scores = (
            screen_data.groupby("ActivityCount")["AvgSimilarity"]
            .agg(["mean", "count", "std"])
            .reset_index()
        )

        # Convert to dictionary for easier handling
        result = {}
        for _, row in event_scores.iterrows():
            event_count = int(row["ActivityCount"])
            result[event_count] = {
                "mean": row["mean"],
                "count": row["count"],
                "std": row["std"] if not np.isnan(row["std"]) else 0,
            }
        result["total"] = {
            "mean": screen_data["AvgSimilarity"].mean(),
            "count": len(screen_data),
            "std": screen_data["AvgSimilarity"].std()
            if not np.isnan(screen_data["AvgSimilarity"].std())
            else 0,
        }

        return result

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {}


def load_and_process_multimodel(base_path, models, cot_value="False"):
    """
    Load and process CSV files from multiple models to analyze events per activity

    Args:
        base_path: Base path to look for model data
        models: List of model names to process
        cot_value: Chain of thought setting ("True" or "False")

    Returns:
        Dictionary with data organized by model and events per activity
    """
    # Dictionary to store results by model
    results = {}

    for model, name in models.items():
        # Define pattern to search for files for this model
        pattern = os.path.join(base_path, f"eval_cot_*_{model}_scores.csv")

        # Find all matching files
        files = glob.glob(pattern)

        if not files:
            continue

        # Combine data from all files for this model
        model_data = {}

        for file_path in files:
            try:
                # Load CSV file
                df = pd.read_csv(file_path)

                # Process data to get events per activity
                # Group by ScreenID to count events per activity
                screen_events = (
                    df.groupby("ScreenID").size().reset_index(name="EventCount")
                )

                # Process labeling performance
                if "GroundTruth" in df.columns and "ActivityLabel" in df.columns:
                    # Get unique ScreenIDs and their event counts and similarities
                    for _, row in screen_events.iterrows():
                        screen_id = row["ScreenID"]
                        event_count = int(row["EventCount"])

                        screen_df = df[df["ScreenID"] == screen_id]

                        # Get the ground truth and activity label for this screen
                        # For simplicity, use the first occurrence if there are multiple
                        ground_truth = screen_df["GroundTruth"].iloc[0]
                        activity_label = screen_df["ActivityLabel"].iloc[0]

                        # Calculate similarity score - if not present, calculate one
                        if "SBERT Similarity" in screen_df.columns:
                            similarity = screen_df["SBERT Similarity"].mean()
                        else:
                            # Placeholder - in a real implementation you'd calculate a similarity measure
                            similarity = 0

                        # Store data by event count
                        if event_count not in model_data:
                            model_data[event_count] = {"scores": [], "count": 0}

                        model_data[event_count]["scores"].append(similarity)
                        model_data[event_count]["count"] += 1
                else:
                    print(f"Required columns not found in {file_path}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # Calculate averages for each event count
        for event_count, data in model_data.items():
            if data["scores"]:
                data["mean"] = np.mean(data["scores"])
                data["std"] = np.std(data["scores"]) if len(data["scores"]) > 1 else 0
            else:
                data["mean"] = 0
                data["std"] = 0

        # Store processed data for this model
        results[name] = model_data

    return results


def plot_events_per_activity(data, output_path, cot_value="False"):
    """
    Create a line plot showing the effect of number of events per activity on labeling performance

    Args:
        data: Dictionary with data organized by model and events per activity
        output_path: Path to save the output figure
        cot_value: Chain of thought setting used (for filename)
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define markers and linestyles for each model
    markers = ["o", "+", "x", "^", "s", "D", "*", "v"]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

    # Colors for models (grayscale gradient)
    gray_levels = [
        "#000000",
        "#444444",
        "#888888",
        "#bbbbbb",
        "#333333",
        "#666666",
        "#999999",
        "#dddddd",
    ]

    # Plot lines for each model
    for i, (model, model_data) in enumerate(data.items()):
        # Skip models with no data
        if not model_data:
            continue

        # Extract data points
        event_counts = sorted(model_data.keys())
        means = [model_data[ec]["mean"] for ec in event_counts if ec != "total"]

        # Skip if no valid data points
        if not means or all(m == 0 for m in means):
            continue

        # Select style elements for this model
        marker_idx = i % len(markers)
        style_idx = i % len(linestyles)
        color_idx = i % len(gray_levels)

        # Plot the line
        ax.plot(
            event_counts,
            means,
            marker=markers[marker_idx],
            linestyle=linestyles[style_idx],
            linewidth=2,
            color=gray_levels[color_idx],
            label=r"\texttt{" + model + "}",
        )

    # Set labels and styling
    ax.set_ylabel(r"\textbf{Average Similarity Score}", fontsize=16)

    # Set y limits
    ax.set_ylim([0, 1.1])

    # Add grid lines
    ax.grid(linestyle="--", alpha=0.7)

    # Force x-axis to use integers
    ax.xaxis.get_major_locator().set_params(integer=True)

    # Add legend
    ax.legend(loc="best", prop={"size": 12})

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    filename = f"events_per_activity_cot_{cot_value}.svg"
    plt.savefig(os.path.join(output_path, filename), format="svg", bbox_inches="tight")
    plt.show()

    # Print average scores
    print(
        f"Average similarity scores by model and events per activity (CoT={cot_value}):"
    )
    for model, model_data in data.items():
        print(f"\n{model}:")
        for event_count in sorted(model_data.keys()):
            if event_count == "total":
                continue
            event_data = model_data[event_count]
            print(
                f"  {event_count} events: {event_data['mean']:.4f} (n={event_data['count']})"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze labeling performance by number of events"
    )
    parser.add_argument(
        "--file", type=str, help="Path to a single CSV file with labeling data"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default=r"c:\Code\BPMExtension\output\phase_3",
        help="Path to directory containing model output files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"c:\Code\BPMExtension\output",
        help="Path to save the output figures",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=["bar", "regression", "both"],
        default="both",
        help="Type of plot to generate",
    )
    parser.add_argument(
        "--multimodel",
        action="store_true",
        help="Analyze multiple models instead of a single file",
    )

    args = parser.parse_args()

    # Configure plot styling
    configure_plot_style()

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    if args.multimodel:
        # Models to analyze
        models = {
            "Athene-V2-Chat": "Athene V2 Chat",
            "Llama-3.1-Nemotron-70B-Instruct": "Llama 3.1 Nemotron",
            "Qwen2.5-72B-Instruct": "Qwen2.5",
            "Athene-70B": "Athene",
        }

        # Load and process data for multiple models
        data = load_and_process_multimodel(args.base_path, models)

        # Plot events per activity analysis
        plot_events_per_activity(data, args.output)
    else:
        # For single-file analysis
        if not args.file:
            parser.error("--file is required when not using --multimodel")

        # Load and process data
        data = load_and_process_file(args.file)
        print(data)


if __name__ == "__main__":
    main()
