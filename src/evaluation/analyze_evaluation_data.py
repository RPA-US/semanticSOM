import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def configure_plot_style():
    """Configure matplotlib to use LaTeX fonts and styling"""
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "axes.labelsize": 12,
            "font.size": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def load_evaluation_data(file_path):
    """Load evaluation data from the CSV file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Evaluation data file not found: {file_path}")

    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} entries from evaluation data")
    return df


def plot_components_by_type_and_density(df, output_path):
    """
    Create a horizontal bar plot showing component counts by type and density level

    Args:
        df: DataFrame containing evaluation data
        output_path: Path to save the output figure
    """
    # Group by Class and Density to count components
    component_counts = df.groupby(["Class", "Density"]).size().unstack(fill_value=0)

    # Calculate total counts and sort by total (descending)
    component_counts["Total"] = component_counts.sum(axis=1)
    component_counts = component_counts.sort_values("Total")

    # Remove the Total column after sorting
    component_counts = component_counts.drop("Total", axis=1)

    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set up the bar width and positions
    bar_width = 0.25
    density_levels = ["Low Density", "Medium Density", "High Density"]

    # Different styles for each density level (grayscale with different hatches)
    styles = {
        "Low Density": {"color": "lightgray", "hatch": "///"},
        "Medium Density": {"color": "darkgray", "hatch": "xxx"},
        "High Density": {"color": "dimgray", "hatch": "..."},
    }

    # Plot bars for each density level
    y_positions = np.arange(len(component_counts.index))

    for i, density in enumerate(density_levels):
        # Skip if density not present in data
        if density not in component_counts.columns:
            continue

        counts = component_counts[density]
        x_pos = y_positions - bar_width + (i * bar_width)

        bars = ax.barh(
            x_pos,
            counts,
            height=bar_width,
            label=density,
            color=styles[density]["color"],
            hatch=styles[density]["hatch"],
            edgecolor="black",
        )

        # # Add value labels on each bar
        # for j, count in enumerate(counts):
        #     if count > 0:
        #         # Position the text in the middle of the bar
        #         ax.text(
        #             count + 2,
        #             x_pos[j],
        #             str(count),
        #             va='center',
        #             fontsize=9,
        #             fontweight='bold'
        #         )

    # Customize the plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r"\textbf{" + c + "}" for c in component_counts.index])

    ax.set_xlabel(r"\textbf{Number of Components}", fontsize=14)
    ax.set_title(r"\textbf{Component Distribution by Type and Density}", fontsize=16)

    # Add gridlines for better readability
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    # Add a legend
    ax.legend(title=r"\textbf{Density Level}", loc="upper right")

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    output_file = os.path.join(
        output_path, "component_distribution_by_type_density.png"
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved component distribution chart to {output_file}")

    plt.show()

    # Print summary statistics
    print("\nComponent counts by type and density:")
    print(component_counts)

    # Print totals
    print("\nTotal component counts by density:")
    print(component_counts.sum())
    print("\nTotal component counts by type:")
    for component_type, counts in component_counts.iterrows():
        print(f"{component_type}: {counts.sum()}")


def main():
    """Main function to analyze evaluation data"""
    # Set up paths
    base_path = r"c:\Code\BPMExtension"
    input_file = os.path.join(base_path, "input", "eval", "eval_dt.csv")
    output_path = os.path.join(base_path, "output")

    # Make sure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Configure plot styling
    configure_plot_style()

    # Load data
    df = load_evaluation_data(input_file)

    # Display basic info about the data
    print("\nBasic dataset information:")
    print(f"Number of entries: {len(df)}")
    print(f"Component types: {df['Class'].unique()}")
    print(f"Density levels: {df['Density'].unique()}")

    # Generate component distribution visualization
    plot_components_by_type_and_density(df, output_path)


if __name__ == "__main__":
    main()
