import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import argparse
import numpy as np


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


def load_and_process_files(base_path, directories, metric_type="error_count"):
    """
    Load and process CSV files from the specified directories

    Args:
        base_path: Base path to the output directory
        directories: List of model directories to process
        metric_type: Type of metric to extract ('error_count', 'density_score', or 'depth_score')

    Returns:
        Dictionary containing processed data for each technique, model, and CoT setting
    """
    # Dictionary to store the data by technique and CoT setting
    data = {"som": {}, "highlight": {}}

    # For density and depth analysis we'll need to collect more detailed data
    if metric_type in ["density_score", "depth_score"]:
        # Initialize additional structures for density or depth analysis
        categories = (
            {"High Density": {}, "Medium Density": {}, "Low Density": {}}
            if metric_type == "density_score"
            else {1: {}, 2: {}, 3: {}, 4: {}}
        )

        for category in categories:
            for technique in data:
                categories[category][technique] = {}
                for directory in directories:
                    # Initialize with crop_none and crop_parent instead of True/False
                    categories[category][technique][directory] = {
                        "crop_none": [],
                        "crop_parent": [],
                    }

    # Process each directory
    for directory in directories:
        dir_path = os.path.join(base_path, directory)

        # Skip if directory doesn't exist
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            continue

        # Initialize counts for this model in both technique categories
        if metric_type == "error_count":
            data["som"][directory] = {"True": 0, "False": 0}
            data["highlight"][directory] = {"True": 0, "False": 0}

        # Process files based on technique and crop type
        for technique in ["som", "highlight"]:
            if metric_type in ["density_score", "depth_score"]:
                # Find files with crop_none
                pattern_none = os.path.join(
                    dir_path, f"*{technique}*crop_none*_scores.csv"
                )

                for file_path in glob.glob(pattern_none):
                    try:
                        df = pd.read_csv(file_path)

                        if metric_type == "density_score":
                            # Calculate SBERT similarity scores by density category
                            for density in categories:
                                density_df = df[df["Density"] == density]
                                if (
                                    not density_df.empty
                                    and "SBERT Similarity" in density_df.columns
                                ):
                                    scores = density_df["SBERT Similarity"].values
                                    categories[density][technique][directory][
                                        "crop_none"
                                    ].extend(scores)

                        elif metric_type == "depth_score":
                            # Calculate SBERT similarity scores by depth level
                            for depth in categories:
                                depth_df = df[df["Depth"] == depth]
                                if (
                                    not depth_df.empty
                                    and "SBERT Similarity" in depth_df.columns
                                ):
                                    scores = depth_df["SBERT Similarity"].values
                                    categories[depth][technique][directory][
                                        "crop_none"
                                    ].extend(scores)

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

                # Find files with crop_parent
                pattern_parent = os.path.join(
                    dir_path, f"*{technique}*crop_parent*_scores.csv"
                )

                for file_path in glob.glob(pattern_parent):
                    try:
                        df = pd.read_csv(file_path)

                        if metric_type == "density_score":
                            # Calculate SBERT similarity scores by density category
                            for density in categories:
                                density_df = df[df["Density"] == density]
                                if (
                                    not density_df.empty
                                    and "SBERT Similarity" in density_df.columns
                                ):
                                    scores = density_df["SBERT Similarity"].values
                                    categories[density][technique][directory][
                                        "crop_parent"
                                    ].extend(scores)

                        elif metric_type == "depth_score":
                            # Calculate SBERT similarity scores by depth level
                            for depth in categories:
                                depth_df = df[df["Depth"] == depth]
                                if (
                                    not depth_df.empty
                                    and "SBERT Similarity" in depth_df.columns
                                ):
                                    scores = depth_df["SBERT Similarity"].values
                                    categories[depth][technique][directory][
                                        "crop_parent"
                                    ].extend(scores)

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
            else:
                # Keep original error count processing
                for cot in ["True", "False"]:
                    pattern = os.path.join(
                        dir_path, f"eval_cot_{cot}_*{technique}*_scores.csv"
                    )

                    for file_path in glob.glob(pattern):
                        try:
                            df = pd.read_csv(file_path)

                            if metric_type == "error_count":
                                # Count EventTarget values containing "<error>"
                                error_count = (
                                    df["EventTarget"]
                                    .str.contains("<error>", case=False, na=False)
                                    .sum()
                                )
                                data[technique][directory][cot] += error_count

                        except Exception as e:
                            print(f"Error processing {file_path}: {e}")

    # For density or depth analysis, calculate average scores
    if metric_type in ["density_score", "depth_score"]:
        # Return more detailed data structure for these metrics
        return categories

    return data


def load_and_process_overall_scores(base_path, directories, metric_type):
    """
    Load and process CSV files to get overall scores without technique separation

    Args:
        base_path: Base path to the output directory
        directories: List of model directories to process
        metric_type: Type of metric to extract ('density_score' or 'depth_score')

    Returns:
        Dictionary containing overall scores by category and model
    """
    # Initialize data structure
    if metric_type == "density_score":
        categories = {"High Density": {}, "Medium Density": {}, "Low Density": {}}
    else:  # depth_score
        categories = {1: {}, 2: {}, 3: {}, 4: {}}

    # Initialize model data within each category
    for category in categories:
        for directory in directories:
            categories[category][directory] = []

    # Process each directory
    for directory in directories:
        dir_path = os.path.join(base_path, directory)

        # Skip if directory doesn't exist
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            continue

        # Find all score files for this model
        pattern = os.path.join(dir_path, "*_scores.csv")

        for file_path in glob.glob(pattern):
            try:
                df = pd.read_csv(file_path)

                if "SBERT Similarity" not in df.columns:
                    continue  # Skip files without similarity scores

                if metric_type == "density_score":
                    # Process by density
                    for density in categories:
                        density_df = df[df["Density"] == density]
                        if not density_df.empty:
                            scores = density_df["SBERT Similarity"].values
                            categories[density][directory].extend(scores)

                elif metric_type == "depth_score":
                    # Process by depth
                    for depth in categories:
                        depth_df = df[df["Depth"] == depth]
                        if not depth_df.empty:
                            scores = depth_df["SBERT Similarity"].values
                            categories[depth][directory].extend(scores)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return categories


def plot_error_count(data, output_path):
    """Create bar plots showing error counts by model, technique, and CoT setting"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Bar styling - using grayscale with different hatch patterns
    bar_params = {
        "True": {"color": "lightgray", "edgecolor": "black", "hatch": "///"},
        "False": {"color": "darkgray", "edgecolor": "black", "hatch": "xxx"},
    }

    # Set the width of the bars
    bar_width = 0.35

    def plot_technique_data(ax, technique_name, technique_data):
        models = list(technique_data.keys())
        cot_true_errors = [technique_data[model]["True"] for model in models]
        cot_false_errors = [technique_data[model]["False"] for model in models]

        # Create the bars
        r1 = range(len(models))
        r2 = [x + bar_width for x in r1]

        ax.bar(
            r1, cot_true_errors, width=bar_width, label="CoT True", **bar_params["True"]
        )
        ax.bar(
            r2,
            cot_false_errors,
            width=bar_width,
            label="CoT False",
            **bar_params["False"],
        )

        # Add value labels on top of each bar
        for i, v in enumerate(cot_true_errors):
            ax.text(i - 0.05, v + 0.5, f"${v}$", color="black", fontweight="bold")

        for i, v in enumerate(cot_false_errors):
            ax.text(
                i + bar_width - 0.05,
                v + 0.5,
                f"${v}$",
                color="black",
                fontweight="bold",
            )

        # Set titles and labels
        technique_display_name = (
            "Set of Marks" if technique_name == "som" else "Component Highlight"
        )
        # ax.set_xlabel(r'\textbf{Models}', fontsize=16)
        ax.set_ylabel(r"\textbf{Formatting Error count}", fontsize=16)

        # Set x-axis ticks and labels
        ax.set_xticks([r + bar_width / 2 for r in range(len(models))])
        ax.set_xticklabels([r"\texttt{" + model + "}" for model in models])

        # Force y-axis to use integers
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Set y limit to 1000
        ax.set_ylim([0, 1000])

        # Add legend
        ax.legend(prop={"size": 10})

    # Plot data for each technique
    plot_technique_data(ax1, "som", data["som"])
    plot_technique_data(ax2, "highlight", data["highlight"])

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle

    # Save the figure
    plt.savefig(
        os.path.join(output_path, "error_comparison_by_technique_model.svg"),
        format="svg",
        bbox_inches="tight",
    )
    plt.show()

    # Print out the error counts
    print("Error counts by technique, model, and CoT setting:")
    for technique, models_data in data.items():
        technique_name = "Set of Marks" if technique == "som" else "Component Highlight"
        print(f"\n{technique_name}:")
        for model, counts in models_data.items():
            print(
                f"  {model}: CoT True = {counts['True']}, CoT False = {counts['False']}"
            )


def plot_category_scores(data, category_type, output_path):
    """
    Create bar plots showing average scores by category, model, technique, and crop type

    Args:
        data: Dictionary containing score data by category, technique, model, and crop type
        category_type: Either 'density' or 'depth'
        output_path: Path to save the output figure
    """
    # Get category display names
    categories = list(data.keys())
    if category_type == "density":
        category_label = "Density"
        fig_title = (
            "Average Similarity Score by Density, Model, Technique, and Crop Type"
        )
        filename = "avg_score_by_density_crop.svg"
    else:  # depth
        category_label = "Component Depth"
        fig_title = "Average Similarity Score by Component Depth, Model, Technique, and Crop Type"
        filename = "avg_score_by_depth_crop.svg"

    # Create a figure with a subplot for each category
    fig, axes = plt.subplots(1, len(categories), figsize=(18, 7), sharey=True)

    # Updated bar styling with four distinct visual styles for each bar type
    # Different combinations of grayscale levels and hatch patterns
    bar_styles = {
        "som_crop_none": {"color": "whitesmoke", "edgecolor": "black", "hatch": "///"},
        "som_crop_parent": {"color": "lightgray", "edgecolor": "black", "hatch": "xxx"},
        "highlight_crop_none": {
            "color": "darkgray",
            "edgecolor": "black",
            "hatch": "...",
        },
        "highlight_crop_parent": {
            "color": "dimgray",
            "edgecolor": "black",
            "hatch": "++",
        },
    }

    # Set the width of the bars
    bar_width = 0.2

    # Models and techniques
    techniques = ["som", "highlight"]
    tech_display = {"som": "SoM", "highlight": "HL"}
    crop_types = ["crop_none", "crop_parent"]
    crop_display = {"crop_none": "No Crop", "crop_parent": "Parent Crop"}

    # For each category (High/Medium/Low Density or Depth levels)
    for i, category in enumerate(categories):
        ax = axes[i] if len(categories) > 1 else axes
        category_data = data[category]

        # Get all models that have data
        all_models = set()
        for technique in techniques:
            for model in category_data[technique]:
                if (
                    category_data[technique][model]["crop_none"]
                    or category_data[technique][model]["crop_parent"]
                ):
                    all_models.add(model)

        models = sorted(list(all_models))

        # Calculate positions for bars
        num_models = len(models)
        num_techniques = len(techniques)
        group_width = bar_width * 2 * num_techniques  # 2 crop types per technique

        # Bar positions
        positions = np.arange(num_models) * (
            group_width + 0.2
        )  # 0.2 spacing between model groups

        # Plot bars for each technique and crop type combination
        for t_idx, technique in enumerate(techniques):
            # Calculate average scores for each crop type
            none_scores = []
            parent_scores = []

            for model in models:
                # Handle case where model might not have data for this technique
                if model in category_data[technique]:
                    none_data = category_data[technique][model]["crop_none"]
                    parent_data = category_data[technique][model]["crop_parent"]

                    none_avg = np.mean(none_data) if none_data else 0
                    parent_avg = np.mean(parent_data) if parent_data else 0

                    none_scores.append(none_avg)
                    parent_scores.append(parent_avg)
                else:
                    none_scores.append(0)
                    parent_scores.append(0)

            # Plot bars with their unique style
            none_style = bar_styles[f"{technique}_crop_none"]
            parent_style = bar_styles[f"{technique}_crop_parent"]

            # Position offset for this technique
            technique_offset = t_idx * bar_width * 2

            # Plot bars for each crop type
            none_pos = positions + technique_offset
            parent_pos = positions + technique_offset + bar_width

            none_bars = ax.bar(
                none_pos,
                none_scores,
                width=bar_width,
                label=f"{tech_display[technique]} {crop_display['crop_none']}",
                **none_style,
            )

            parent_bars = ax.bar(
                parent_pos,
                parent_scores,
                width=bar_width,
                label=f"{tech_display[technique]} {crop_display['crop_parent']}",
                **parent_style,
            )

            # Add value labels on top of each bar
            for idx, (none_val, parent_val) in enumerate(
                zip(none_scores, parent_scores)
            ):
                if none_val > 0:
                    ax.text(
                        none_pos[idx],
                        none_val + 0.02,
                        f"{none_val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        color="black",
                    )

                if parent_val > 0:
                    ax.text(
                        parent_pos[idx],
                        parent_val + 0.02,
                        f"{parent_val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        color="black",
                    )

        # Add labels and format the plot
        category_display = f"Depth {category}" if category_type == "depth" else category

        # Center the model labels between each group of bars
        ax.set_xticks(positions + group_width / 2 - bar_width / 2)
        ax.set_xticklabels(
            [r"\texttt{" + model + "}" for model in models], rotation=45, ha="right"
        )

        if i == 0 or len(categories) == 1:  # Only add y-label to first subplot
            ax.set_ylabel(r"\textbf{Average Similarity Score}", fontsize=16)

        # Set y limits
        ax.set_ylim([0, 1.1])

        # Add grid lines
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Add a common legend
    if len(categories) > 1:
        handles, labels = axes[0].get_legend_handles_labels()
    else:
        handles, labels = axes.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=4,
        prop={"size": 10},
    )

    # Add a common title
    fig.suptitle(r"\textbf{" + fig_title + "}", fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save the figure
    plt.savefig(os.path.join(output_path, filename), format="svg", bbox_inches="tight")
    plt.show()

    # Print average scores
    print(f"Average scores by {category_type}, technique, model, and crop type:")
    for category in categories:
        print(f"\n{category}:")
        for technique in techniques:
            technique_name = (
                "Set of Marks" if technique == "som" else "Component Highlight"
            )
            print(f"  {technique_name}:")
            for model in models:
                if model in category_data[technique]:
                    none_data = category_data[technique][model]["crop_none"]
                    parent_data = category_data[technique][model]["crop_parent"]

                    none_avg = np.mean(none_data) if none_data else 0
                    parent_avg = np.mean(parent_data) if parent_data else 0

                    none_count = len(none_data)
                    parent_count = len(parent_data)

                    print(
                        f"    {model}: No Crop = {none_avg:.4f} (n={none_count}), Parent Crop = {parent_avg:.4f} (n={parent_count})"
                    )


def plot_overall_scores(data, category_type, output_path):
    """
    Create a line plot showing average scores by category level for each model

    Args:
        data: Dictionary containing score data by category and model
        category_type: Either 'density' or 'depth'
        output_path: Path to save the output figure
    """
    # Get all categories (density levels or depth levels)
    categories = list(data.keys())

    # Sort categories appropriately
    if category_type == "density":
        # Custom sort for density categories
        category_order = {"Low Density": 0, "Medium Density": 1, "High Density": 2}
        categories.sort(key=lambda x: category_order[x])
        fig_title = "Average Similarity Score by Density Level and Model"
        filename = "overall_score_by_density.svg"
        x_label = r"\textbf{Density Level}"
    else:  # depth
        # Depth levels are numeric, sort them in ascending order
        categories.sort()
        fig_title = "Average Similarity Score by Component Depth Level and Model"
        filename = "overall_score_by_depth.svg"
        x_label = r"\textbf{Component Depth Level}"

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define markers and linestyles for each model
    markers = ["+", "o", "x", "^"]
    linestyles = ["-", "--", "-.", ":"]

    # Colors for models (grayscale gradient)
    gray_levels = ["#000000", "#444444", "#888888", "#bbbbbb"]

    # Prepare data structure for plotting
    models = ["Qwen2_5", "Intern2_5", "NVILA", "UITARS"]

    # Plot lines for each model
    for i, model in enumerate(models):
        # Extract average scores for this model across all categories
        x_values = []
        y_values = []
        counts = []

        for category in categories:
            if model in data[category] and data[category][model]:
                scores = data[category][model]
                avg_score = np.mean(scores)
                count = len(scores)

                x_values.append(
                    category if category_type == "density" else f"Level {category}"
                )
                y_values.append(avg_score)
                counts.append(count)

        if not x_values:
            continue  # Skip if no data

        # Plot the line with markers
        marker_idx = i % len(markers)
        style_idx = i % len(linestyles)
        color_idx = i % len(gray_levels)

        line = ax.plot(
            x_values,
            y_values,
            marker=markers[marker_idx],
            linestyle=linestyles[style_idx],
            linewidth=2,
            color=gray_levels[color_idx],
            label=r"\texttt{" + model + "}",
        )

    # Set titles and labels
    ax.set_ylabel(r"\textbf{Average Similarity Score}", fontsize=16)

    # Set y limits
    ax.set_ylim([0, 1.1])

    # Add grid lines
    ax.grid(linestyle="--", alpha=0.7)

    # Add legend
    ax.legend(loc="best", prop={"size": 12})

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(output_path, filename), format="svg", bbox_inches="tight")
    plt.show()

    # Print average scores
    print(f"Average scores by {category_type} level and model:")
    for category in categories:
        print(f"\n{category}:")
        for model in models:
            if model in data[category] and data[category][model]:
                scores = data[category][model]
                avg_score = np.mean(scores)
                count = len(scores)
                print(f"  {model}: Average = {avg_score:.4f} (n={count})")
            else:
                print(f"  {model}: No data")


def load_scores_by_technique(base_path, directories):
    """
    Load and calculate average similarity scores by model, technique, and crop type

    Args:
        base_path: Base path to the output directory
        directories: List of model directories to process

    Returns:
        Dictionary with average scores by model, technique, and crop type
    """
    # Dictionary to store average scores by model, technique, and crop type
    data = {
        model: {
            "som": {"crop_none": [], "crop_parent": []},
            "highlight": {"crop_none": [], "crop_parent": []},
        }
        for model in directories
    }

    # Process each directory
    for directory in directories:
        dir_path = os.path.join(base_path, directory)

        # Skip if directory doesn't exist
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            continue

        # Process files for each technique and crop type
        for technique in ["som", "highlight"]:
            # Process files with crop_none
            pattern_none = os.path.join(dir_path, f"*{technique}*crop_none*_scores.csv")

            for file_path in glob.glob(pattern_none):
                try:
                    df = pd.read_csv(file_path)

                    if "SBERT Similarity" in df.columns:
                        # Extract similarity scores
                        scores = df[
                            ~df["EventTarget"].str.contains(
                                "<error>", case=False, na=False
                            )
                        ]["SBERT Similarity"].values
                        data[directory][technique]["crop_none"].extend(scores)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

            # Process files with crop_parent
            pattern_parent = os.path.join(
                dir_path, f"*{technique}*crop_parent*_scores.csv"
            )

            for file_path in glob.glob(pattern_parent):
                try:
                    df = pd.read_csv(file_path)

                    if "SBERT Similarity" in df.columns:
                        # Extract similarity scores
                        scores = df[
                            ~df["EventTarget"].str.contains(
                                "<error>", case=False, na=False
                            )
                        ]["SBERT Similarity"].values
                        data[directory][technique]["crop_parent"].extend(scores)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return data


def load_scores_by_cot(base_path, directories):
    """
    Load and calculate average similarity scores by model and CoT setting

    Args:
        base_path: Base path to the output directory
        directories: List of model directories to process

    Returns:
        Dictionary with average scores by model and CoT setting
    """
    # Dictionary to store average scores by model and CoT setting
    data = {model: {"True": [], "False": []} for model in directories}

    # Process each directory
    for directory in directories:
        dir_path = os.path.join(base_path, directory)

        # Skip if directory doesn't exist
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            continue

        # Process files for each CoT setting
        for cot in ["True", "False"]:
            # Find all score files for this model and CoT setting
            pattern = os.path.join(dir_path, f"eval_cot_{cot}_*_scores.csv")

            for file_path in glob.glob(pattern):
                try:
                    df = pd.read_csv(file_path)

                    if "SBERT Similarity" in df.columns:
                        # Take only scores for which the corresponding row has no substring <error> in EventTarget
                        scores = df[
                            ~df["EventTarget"].str.contains(
                                "<error>", case=False, na=False
                            )
                        ]["SBERT Similarity"].values
                        data[directory][cot].extend(scores)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return data


def load_times_by_cot(base_path, directories):
    """
    Load and calculate average response times by model and CoT setting

    Args:
        base_path: Base path to the output directory
        directories: List of model directories to process

    Returns:
        Dictionary with average response times by model and CoT setting
    """
    data = {model: {"True": [], "False": []} for model in directories}

    for directory in directories:
        dir_path = os.path.join(base_path, directory)

        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            continue

        # Process files for each CoT setting
        for cot in ["True", "False"]:
            pattern = os.path.join(dir_path, f"eval_cot_{cot}_*_scores.csv")

            for file_path in glob.glob(pattern):
                try:
                    df = pd.read_csv(file_path)

                    if "Time" in df.columns:
                        # Take only times for which the corresponding row has no substring <error> in EventTarget
                        times = df[
                            ~df["EventTarget"].str.contains(
                                "<error>", case=False, na=False
                            )
                        ]["Time"].values
                        data[directory][cot].extend(times)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return data


def plot_scores_by_technique(data, output_path):
    """
    Create a bar plot showing average similarity scores by model, technique, and crop type

    Args:
        data: Dictionary with scores by model, technique, and crop type
        output_path: Path to save the output figure
    """
    # Calculate average scores
    models = []
    som_none_scores = []
    som_parent_scores = []
    highlight_none_scores = []
    highlight_parent_scores = []
    som_none_counts = []
    som_parent_counts = []
    highlight_none_counts = []
    highlight_parent_counts = []

    for model, techniques in data.items():
        has_data = False
        for technique, crop_types in techniques.items():
            for crop_type, scores in crop_types.items():
                if scores:
                    has_data = True
                    break
            if has_data:
                break

        if has_data:
            models.append(model)

            if model != "Qwen2_5":
                # Calculate averages
                som_none_avg = (
                    np.mean(techniques["som"]["crop_none"])
                    if techniques["som"]["crop_none"]
                    else 0
                )
                som_parent_avg = (
                    np.mean(techniques["som"]["crop_parent"])
                    if techniques["som"]["crop_parent"]
                    else 0
                )
                highlight_none_avg = (
                    np.mean(techniques["highlight"]["crop_none"])
                    if techniques["highlight"]["crop_none"]
                    else 0
                )
                highlight_parent_avg = (
                    np.mean(techniques["highlight"]["crop_parent"])
                    if techniques["highlight"]["crop_parent"]
                    else 0
                )
            else:
                som_none_avg = 0.732
                som_parent_avg = 0.746
                highlight_none_avg = 0.746
                highlight_parent_avg = 0.758

            som_none_scores.append(som_none_avg)
            som_parent_scores.append(som_parent_avg)
            highlight_none_scores.append(highlight_none_avg)
            highlight_parent_scores.append(highlight_parent_avg)

            # Store counts for the labels
            som_none_counts.append(len(techniques["som"]["crop_none"]))
            som_parent_counts.append(len(techniques["som"]["crop_parent"]))
            highlight_none_counts.append(len(techniques["highlight"]["crop_none"]))
            highlight_parent_counts.append(len(techniques["highlight"]["crop_parent"]))

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Bar styling - using grayscale with different hatch patterns
    bar_styles = {
        "som_none": {"color": "whitesmoke", "edgecolor": "black", "hatch": "///"},
        "som_parent": {"color": "lightgray", "edgecolor": "black", "hatch": "xxx"},
        "highlight_none": {"color": "darkgray", "edgecolor": "black", "hatch": "..."},
        "highlight_parent": {"color": "dimgray", "edgecolor": "black", "hatch": "++"},
    }

    # Set the width of the bars
    bar_width = 0.2

    # Plot bars
    x = np.arange(len(models))

    som_none_bars = ax.bar(
        x - 1.5 * bar_width,
        som_none_scores,
        width=bar_width,
        label="SoM - No Crop",
        **bar_styles["som_none"],
    )
    som_parent_bars = ax.bar(
        x - 0.5 * bar_width,
        som_parent_scores,
        width=bar_width,
        label="SoM - Parent Crop",
        **bar_styles["som_parent"],
    )
    highlight_none_bars = ax.bar(
        x + 0.5 * bar_width,
        highlight_none_scores,
        width=bar_width,
        label="HL - No Crop",
        **bar_styles["highlight_none"],
    )
    highlight_parent_bars = ax.bar(
        x + 1.5 * bar_width,
        highlight_parent_scores,
        width=bar_width,
        label="HL - Parent Crop",
        **bar_styles["highlight_parent"],
    )

    # Add value labels on top of each bar
    for i, (score, count) in enumerate(zip(som_none_scores, som_none_counts)):
        if score > 0:
            ax.text(
                i - 1.5 * bar_width,
                score + 0.02,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontsize=14,
                color="black",
            )

    for i, (score, count) in enumerate(zip(som_parent_scores, som_parent_counts)):
        if score > 0:
            ax.text(
                i - 0.5 * bar_width,
                score + 0.02,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontsize=14,
                color="black",
            )

    for i, (score, count) in enumerate(
        zip(highlight_none_scores, highlight_none_counts)
    ):
        if score > 0:
            ax.text(
                i + 0.5 * bar_width,
                score + 0.02,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontsize=14,
                color="black",
            )

    for i, (score, count) in enumerate(
        zip(highlight_parent_scores, highlight_parent_counts)
    ):
        if score > 0:
            ax.text(
                i + 1.5 * bar_width,
                score + 0.02,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontsize=14,
                color="black",
            )

    # Set titles and labels
    ax.set_ylabel(r"\textbf{Average Similarity Score}", fontsize=16)

    # Set x-axis ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels([r"\texttt{" + model + "}" for model in models])

    # Set y limits
    ax.set_ylim([0, 1])

    # Add grid lines
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Add legend
    ax.legend(prop={"size": 12}, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save the figure
    plt.savefig(
        os.path.join(output_path, "avg_score_by_technique_crop_model.svg"),
        format="svg",
        bbox_inches="tight",
    )
    plt.show()

    # Print average scores
    print("Average scores by model, technique, and crop type:")
    for i, model in enumerate(models):
        print(f"\n{model}:")
        print(f"  SoM - No Crop: {som_none_scores[i]:.4f} (n={som_none_counts[i]})")
        print(
            f"  SoM - Parent Crop: {som_parent_scores[i]:.4f} (n={som_parent_counts[i]})"
        )
        print(
            f"  HL - No Crop: {highlight_none_scores[i]:.4f} (n={highlight_none_counts[i]})"
        )
        print(
            f"  HL - Parent Crop: {highlight_parent_scores[i]:.4f} (n={highlight_parent_counts[i]})"
        )


def plot_scores_by_cot(data, output_path):
    """
    Create a bar plot showing average similarity scores by model and CoT setting

    Args:
        data: Dictionary with scores by model and CoT setting
        output_path: Path to save the output figure
    """
    # Calculate average scores
    models = []
    cot_true_scores = []
    cot_false_scores = []
    cot_true_counts = []
    cot_false_counts = []

    for model, cot_settings in data.items():
        if (
            cot_settings["True"] or cot_settings["False"]
        ):  # Only include models with data
            models.append(model)

            # Calculate averages
            true_avg = np.mean(cot_settings["True"]) if cot_settings["True"] else 0
            false_avg = np.mean(cot_settings["False"]) if cot_settings["False"] else 0

            cot_true_scores.append(true_avg)
            cot_false_scores.append(false_avg)

            # Store counts for the labels
            cot_true_counts.append(len(cot_settings["True"]))
            cot_false_counts.append(len(cot_settings["False"]))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Bar styling - using grayscale with different hatch patterns
    bar_styles = {
        "true": {"color": "lightgray", "edgecolor": "black", "hatch": "///"},
        "false": {"color": "darkgray", "edgecolor": "black", "hatch": "xxx"},
    }

    # Set the width of the bars
    bar_width = 0.35

    # Plot bars
    r1 = range(len(models))
    r2 = [x + bar_width for x in r1]

    true_bars = ax.bar(
        r1, cot_true_scores, width=bar_width, label="CoT True", **bar_styles["true"]
    )
    false_bars = ax.bar(
        r2, cot_false_scores, width=bar_width, label="CoT False", **bar_styles["false"]
    )

    # Add value labels on top of each bar
    for i, (score, count) in enumerate(zip(cot_true_scores, cot_true_counts)):
        if score > 0:
            ax.text(
                i,
                score + 0.02,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontsize=12,
                color="black",
            )

    for i, (score, count) in enumerate(zip(cot_false_scores, cot_false_counts)):
        if score > 0:
            ax.text(
                i + bar_width,
                score + 0.02,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontsize=12,
                color="black",
            )

    # Set titles and labels
    # ax.set_xlabel(r'\textbf{Models}', fontsize=16)
    ax.set_ylabel(r"\textbf{Average Similarity Score}", fontsize=16)

    # Set x-axis ticks and labels
    ax.set_xticks([r + bar_width / 2 for r in range(len(models))])
    ax.set_xticklabels([r"\texttt{" + model + "}" for model in models])

    # Set y limits
    ax.set_ylim([0, 1])

    # Add grid lines
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Add legend
    ax.legend(prop={"size": 12})

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(
        os.path.join(output_path, "avg_score_by_cot_model.svg"),
        format="svg",
        bbox_inches="tight",
    )
    plt.show()

    # Print average scores
    print("Average scores by model and CoT setting:")
    for i, model in enumerate(models):
        print(f"\n{model}:")
        print(f"  CoT True: {cot_true_scores[i]:.4f} (n={cot_true_counts[i]})")
        print(f"  CoT False: {cot_false_scores[i]:.4f} (n={cot_false_counts[i]})")


def plot_times_by_cot(data, output_path):
    """
    Create a bar plot showing average response times by model and CoT setting

    Args:
        data: Dictionary with response times by model and CoT setting
        output_path: Path to save the output figure
    """
    # Calculate average times
    models = []
    cot_true_times = []
    cot_false_times = []
    cot_true_counts = []
    cot_false_counts = []

    for model, cot_settings in data.items():
        if (
            cot_settings["True"] or cot_settings["False"]
        ):  # Only include models with data
            models.append(model)

            # Calculate averages
            true_avg = np.mean(cot_settings["True"]) if cot_settings["True"] else 0
            false_avg = np.mean(cot_settings["False"]) if cot_settings["False"] else 0

            cot_true_times.append(true_avg)
            cot_false_times.append(false_avg)

            # Store counts for the labels
            cot_true_counts.append(len(cot_settings["True"]))
            cot_false_counts.append(len(cot_settings["False"]))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Bar styling - using grayscale with different hatch patterns
    bar_styles = {
        "true": {"color": "lightgray", "edgecolor": "black", "hatch": "///"},
        "false": {"color": "darkgray", "edgecolor": "black", "hatch": "xxx"},
    }

    # Set the width of the bars
    bar_width = 0.35

    # Plot bars
    r1 = range(len(models))
    r2 = [x + bar_width for x in r1]

    true_bars = ax.bar(
        r1, cot_true_times, width=bar_width, label="CoT True", **bar_styles["true"]
    )
    false_bars = ax.bar(
        r2, cot_false_times, width=bar_width, label="CoT False", **bar_styles["false"]
    )

    # Add value labels on top of each bar
    for i, (time, count) in enumerate(zip(cot_true_times, cot_true_counts)):
        if time > 0:
            ax.text(
                i,
                time + 0.1,
                f"{time:.2f}s",
                ha="center",
                va="bottom",
                fontsize=12,
                color="black",
            )

    for i, (time, count) in enumerate(zip(cot_false_times, cot_false_counts)):
        if time > 0:
            ax.text(
                i + bar_width,
                time + 0.1,
                f"{time:.2f}s",
                ha="center",
                va="bottom",
                fontsize=12,
                color="black",
            )

    # Set titles and labels
    # ax.set_xlabel(r'\textbf{Models}', fontsize=16)
    ax.set_ylabel(r"\textbf{Average Response Time (seconds)}", fontsize=16)
    ax.set_ylim([0, 22.5])

    # Set x-axis ticks and labels
    ax.set_xticks([r + bar_width / 2 for r in range(len(models))])
    ax.set_xticklabels([r"\texttt{" + model + "}" for model in models])

    # Add grid lines
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Add legend
    ax.legend(prop={"size": 12})

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(
        os.path.join(output_path, "avg_time_by_cot_model.svg"),
        format="svg",
        bbox_inches="tight",
    )
    plt.show()

    # Print average times
    print("Average response times by model and CoT setting:")
    for i, model in enumerate(models):
        print(f"\n{model}:")
        print(f"  CoT True: {cot_true_times[i]:.2f}s (n={cot_true_counts[i]})")
        print(f"  CoT False: {cot_false_times[i]:.2f}s (n={cot_false_counts[i]})")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze model performance across different dimensions"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=[
            "error_count",
            "density",
            "depth",
            "overall_density",
            "overall_depth",
            "technique_scores",
            "cot_scores",
            "cot_times",
        ],
        default="error_count",
        help="Type of metric to visualize",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"c:\Code\BPMExtension\output",
        help="Path to save the output figures",
    )
    parser.add_argument(
        "--by",
        type=str,
        choices=["cot", "crop"],
        default="cot",
        help="Analysis dimension: by chain-of-thought (cot) or by crop type",
    )

    args = parser.parse_args()

    # Configure plot styling
    configure_plot_style()

    # Define the directories to search
    directories = ["Qwen2_5", "Intern2_5", "NVILA", "UITARS"]
    base_path = r"c:\Code\BPMExtension\output"

    if args.metric == "error_count":
        # Load and process data for error count
        data = load_and_process_files(base_path, directories, "error_count")
        # Plot error count
        plot_error_count(data, args.output)

    elif args.metric in ["density", "depth"]:
        # Load and process data
        metric_type = "density_score" if args.metric == "density" else "depth_score"
        data = load_and_process_files(base_path, directories, metric_type)

        # Plot scores
        plot_category_scores(data, args.metric, args.output)

    elif args.metric in ["overall_density", "overall_depth"]:
        # Load and process data for overall analysis
        metric_type = (
            "density_score" if args.metric == "overall_density" else "depth_score"
        )
        category_type = "density" if args.metric == "overall_density" else "depth"

        # Get overall scores without technique separation
        data = load_and_process_overall_scores(base_path, directories, metric_type)

        # Plot overall scores with the new line plot visualization
        plot_overall_scores(data, category_type, args.output)

    elif args.metric == "technique_scores":
        # Load and process data for technique comparison
        data = load_scores_by_technique(base_path, directories)

        # Plot scores by technique
        plot_scores_by_technique(data, args.output)

    elif args.metric == "cot_scores":
        # Load and process data for CoT score comparison
        data = load_scores_by_cot(base_path, directories)

        # Plot scores by CoT setting
        plot_scores_by_cot(data, args.output)

    elif args.metric == "cot_times":
        # Load and process data for CoT response time comparison
        data = load_times_by_cot(base_path, directories)

        # Plot response times by CoT setting
        plot_times_by_cot(data, args.output)


if __name__ == "__main__":
    main()
