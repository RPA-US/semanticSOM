import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize evaluation scores")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="c:/Code/BPMExtension/output/eval_Qwen2_5-70B.csv",
        help="Path to CSV file with evaluation results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="c:/Code/BPMExtension/output/visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["SBERT Similarity"],
        help="Metrics to visualize (column names in CSV)",
    )
    return parser.parse_args()


def ensure_output_dir(output_dir):
    """Ensure the output directory exists"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def load_data(csv_path):
    """Load data from CSV file"""
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None


def hommel_test(data, group_column, value_column):
    """
    Perform Hommel's method for multiple pairwise t-tests with p-value adjustment
    Returns a dictionary with test results and conclusion
    """
    groups = data[group_column].unique()
    results = []
    p_values = []

    # Perform pairwise t-tests
    for i, group1 in enumerate(groups):
        for group2 in groups[i + 1 :]:
            sample1 = data[data[group_column] == group1][value_column]
            sample2 = data[data[group_column] == group2][value_column]

            if (
                len(sample1) > 1 and len(sample2) > 1
            ):  # Need at least 2 samples for t-test
                t_stat, p_val = stats.ttest_ind(sample1, sample2, equal_var=False)
                results.append(
                    {
                        "group1": group1,
                        "group2": group2,
                        "p_value": p_val,
                        "t_statistic": t_stat,
                    }
                )
                p_values.append(p_val)

    # Apply Hommel's correction
    if p_values:
        reject, p_adjusted, _, _ = multipletests(p_values, method="hommel")

        # Update results with adjusted p-values
        for i, res in enumerate(results):
            res["p_adjusted"] = p_adjusted[i]
            res["significant"] = reject[i]

        # Overall conclusion
        any_significant = any(reject)
        conclusion = (
            "Significant differences found"
            if any_significant
            else "No significant differences"
        )

        return {
            "results": results,
            "conclusion": conclusion,
            "any_significant": any_significant,
        }

    return {
        "results": [],
        "conclusion": "Insufficient data for testing",
        "any_significant": False,
    }


def kruskal_wallis_test(data, group_column, value_column):
    """
    Perform Kruskal-Wallis H-test to determine if there are statistically significant differences
    Returns a dictionary with test results and conclusion
    """
    groups = data[group_column].unique()
    samples = [
        data[data[group_column] == group][value_column].values for group in groups
    ]

    # Filter out empty samples
    samples = [sample for sample in samples if len(sample) > 0]

    if len(samples) > 1:  # Need at least 2 groups for the test
        try:
            h_stat, p_val = stats.kruskal(*samples)
            significant = p_val < 0.05
            conclusion = (
                "Significant differences found"
                if significant
                else "No significant differences"
            )
            return {
                "h_statistic": h_stat,
                "p_value": p_val,
                "significant": significant,
                "conclusion": conclusion,
            }
        except Exception as e:
            return {"conclusion": f"Test error: {str(e)}", "significant": False}

    return {"conclusion": "Insufficient groups for testing", "significant": False}


def plot_by_depth(data, metrics, output_dir):
    """Generate violin plots showing relationship between scores and component depth"""
    print("Generating depth-based visualizations...")

    # Convert depth to numeric
    data["Depth"] = pd.to_numeric(data["Depth"])

    for metric in metrics:
        if metric not in data.columns:
            print(f"Warning: Metric {metric} not found in dataset")
            continue

        plt.figure(figsize=(10, 6))

        # Violin plot with individual points
        sns.violinplot(x="Depth", y=metric, data=data, inner=None, alpha=0.6)
        sns.stripplot(
            x="Depth",
            y=metric,
            data=data,
            color="black",
            size=3,
            jitter=True,
            alpha=0.7,
        )

        # Calculate mean scores by depth and display them
        means = data.groupby("Depth")[metric].mean().reset_index()
        for i, row in means.iterrows():
            plt.annotate(
                f"Mean: {row[metric]:.3f}",
                (row["Depth"], row[metric]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        plt.title(f"{metric} by Depth (Violin Plot)")
        plt.xlabel("Depth")
        plt.ylim(0, 1)
        plt.ylabel(metric)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xticks([1, 2, 3, 4, 5])

        # Perform Hommel test specifically on depths 2 and 3
        depths_2_3 = data[data["Depth"].isin([2, 3])]
        if len(depths_2_3) > 0:
            test_result = hommel_test(depths_2_3, "Depth", metric)
            plt.figtext(
                0.95,
                0.95,
                test_result["conclusion"],
                horizontalalignment="right",
                verticalalignment="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.5"),
            )

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"depth_{metric.replace(' ', '_')}.png")
        plt.savefig(output_path)
        print(f"  Saved: {output_path}")
        plt.close()


def plot_by_density(data, metrics, output_dir):
    """Generate scatterplots showing relationship between scores and image density"""
    print("Generating density-based visualizations...")

    # Order density categories
    density_order = ["Low Density", "Medium Density", "High Density"]
    data["Density_Numeric"] = pd.Categorical(
        data["Density"], categories=density_order, ordered=True
    ).codes

    for metric in metrics:
        if metric not in data.columns:
            continue

        plt.figure(figsize=(12, 6))

        # Violin plot with individual points
        sns.violinplot(
            x="Density", y=metric, data=data, order=density_order, inner=None, alpha=0.6
        )
        sns.stripplot(
            x="Density",
            y=metric,
            data=data,
            order=density_order,
            color="black",
            size=3,
            jitter=True,
            alpha=0.7,
        )
        plt.title(f"{metric} by Density (Violin Plot)")
        plt.xticks(rotation=45)
        plt.ylim(0, 1)

        # Calculate mean scores by density and display them
        means = data.groupby("Density")[metric].mean()
        for i, density in enumerate(density_order):
            if density in means.index:
                plt.annotate(
                    f"Mean: {means[density]:.3f}",
                    xy=(i, means[density]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                )

        # Perform Kruskal-Wallis test for density levels
        test_result = kruskal_wallis_test(data, "Density", metric)
        plt.figtext(
            0.95,
            0.95,
            test_result["conclusion"],
            horizontalalignment="right",
            verticalalignment="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.5"),
        )

        plt.tight_layout()
        output_path = os.path.join(
            output_dir, f"density_{metric.replace(' ', '_')}.png"
        )
        plt.savefig(output_path)
        print(f"  Saved: {output_path}")
        plt.close()


def plot_by_component_type(data, metrics, output_dir):
    """Generate violin plots showing relationship between scores and component type"""
    print("Generating component type-based visualizations...")

    for metric in metrics:
        if metric not in data.columns:
            continue

        # Count components of each type for sorting
        type_counts = data["Class"].value_counts()
        common_types = type_counts[type_counts >= 3].index.tolist()
        filtered_data = data[data["Class"].isin(common_types)]

        if len(common_types) > 0:
            plt.figure(figsize=(14, 7))

            # Violin plot by component type
            sns.violinplot(
                x="Class", y=metric, data=filtered_data, inner=None, alpha=0.6
            )
            sns.stripplot(
                x="Class",
                y=metric,
                data=filtered_data,
                color="black",
                size=3,
                jitter=True,
                alpha=0.7,
            )

            # Calculate and display means
            means = filtered_data.groupby("Class")[metric].mean()
            for i, comp_type in enumerate(means.index):
                plt.annotate(
                    f"Mean: {means[comp_type]:.3f}",
                    xy=(i, means[comp_type]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                )

            # Perform Hommel test for component types
            test_result = hommel_test(filtered_data, "Class", metric)
            plt.figtext(
                0.95,
                0.95,
                test_result["conclusion"],
                horizontalalignment="right",
                verticalalignment="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.5"),
            )

            plt.title(f"{metric} by Component Type (Violin Plot)")
            plt.xlabel("Component Type")
            plt.ylim(0, 1)
            plt.ylabel(metric)
            plt.xticks(rotation=45, ha="right")
            plt.grid(True, linestyle="--", alpha=0.3, axis="y")
            plt.tight_layout()

            output_path = os.path.join(
                output_dir, f"component_{metric.replace(' ', '_')}.png"
            )
            plt.savefig(output_path)
            print(f"  Saved: {output_path}")
            plt.close()


def visualize_scores(csv_path, output_dir, metrics):
    """Main function to visualize scores"""
    # Ensure output directory exists
    ensure_output_dir(output_dir)

    # Load data
    data = load_data(csv_path)
    if data is None:
        return

    # Check if metrics exist in the data
    available_metrics = [m for m in metrics if m in data.columns]
    if not available_metrics:
        print(f"None of the specified metrics {metrics} found in the dataset")
        print(f"Available metrics: {', '.join(data.columns)}")
        return

    print(f"Generating visualizations for metrics: {', '.join(available_metrics)}")

    # Set the style for all plots
    sns.set_style("whitegrid")

    # Generate plots
    plot_by_depth(data, available_metrics, output_dir)
    plot_by_density(data, available_metrics, output_dir)
    plot_by_component_type(data, available_metrics, output_dir)

    print(f"All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    visualize_scores(
        csv_path=args.csv_path, output_dir=args.output_dir, metrics=args.metrics
    )
