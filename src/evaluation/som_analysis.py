import os
import glob
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns  # added seaborn import
import numpy as np


def analyze_soms(directory):
    comp_counts = []
    depth_counter = Counter()
    class_counter = Counter()
    comp_info = []  # list of tuples: (file, comp_count)

    files = glob.glob(os.path.join(directory, "*.json"))
    for f in files:
        try:
            with open(f, "r") as jf:
                data = json.load(jf)
        except Exception as e:
            print(f"Error processing {f}: {e}")
            continue

        nodes = data.get("compos")
        count = len(nodes)
        comp_counts.append(count)
        comp_info.append((f, count))

        # Extract node metrics
        extract_node_metrics(nodes, depth_counter, class_counter)

    return comp_counts, depth_counter, class_counter, comp_info


def extract_node_metrics(nodes, depth_counter, class_counter):
    """Extract metrics from nodes and update the counters."""
    for node in nodes:
        depth = node.get("depth")
        if depth is not None:
            depth_counter[depth] += 1
        cls = node.get("class")
        if cls is not None:
            class_counter[cls] += 1


def plot_metrics(comp_counts, depth_counter, class_counter, title_prefix=""):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    sns.histplot(comp_counts, edgecolor="black", kde=True)
    plt.xlabel("Components per Image")
    plt.ylabel("Frequency")
    plt.title(f"{title_prefix}Distribution of Components per Image")

    plt.subplot(1, 3, 2)
    all_depths = [d for d, count in depth_counter.items() for _ in range(count)]
    if all_depths:  # Check if there are any depths to plot
        sns.histplot(
            all_depths,
            bins=range(min(all_depths), max(all_depths) + 2),
            edgecolor="black",
        )
    plt.xlabel("Depth Level")
    plt.ylabel("Count")
    plt.title(f"{title_prefix}Distribution of Depth Levels")

    plt.subplot(1, 3, 3)
    all_classes = [cls for cls, count in class_counter.items() for _ in range(count)]
    if all_classes:  # Check if there are any classes to plot
        ax = sns.histplot(all_classes, stat="count", discrete=True, edgecolor="black")
        # Add counts on top of each bar
        for p in ax.patches:
            height = p.get_height()
            if height > 0:  # Only annotate if there's a bar
                ax.annotate(
                    f"{int(height)}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                    xytext=(0, 5),
                    textcoords="offset points",
                    rotation=90,
                )
    plt.xlabel("Component Class")
    plt.ylabel("Count")
    plt.yscale("log")
    plt.title(f"{title_prefix}Number of Items per Class")
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()


def group_data_into_terciles(comp_info):
    """Group the data into terciles and return the groups along with their metrics."""
    if not comp_info:
        return [], [], []

    sorted_info = sorted(comp_info, key=lambda x: x[1])
    n = len(sorted_info)
    # Divide into terciles
    group1 = sorted_info[: n // 3]
    group2 = sorted_info[n // 3 : 2 * n // 3]
    group3 = sorted_info[2 * n // 3 :]

    for img in group1:
        update_soms_with_density(img[0], "Low Density")
    for img in group2:
        update_soms_with_density(img[0], "Medium Density")
    for img in group3:
        update_soms_with_density(img[0], "High Density")

    return [group1, group2, group3]


def analyze_group_metrics(directory, groups):
    """Analyze metrics for each group."""
    group_metrics = []

    for group in groups:
        files = [item[0] for item in group]
        comp_counts = [item[1] for item in group]
        depth_counter = Counter()
        class_counter = Counter()

        for f in files:
            try:
                with open(f, "r") as jf:
                    data = json.load(fp=jf)
                    nodes = data.get("compos", [])
                    # Use the shared function for metrics extraction
                    extract_node_metrics(nodes, depth_counter, class_counter)
            except Exception as e:
                print(f"Error processing {f}: {e}")
                continue

        group_metrics.append((comp_counts, depth_counter, class_counter))

    return group_metrics


def select_and_show_representative_images(directory, comp_info):
    if not comp_info:
        print("No image data available.")
        return

    groups = group_data_into_terciles(comp_info)
    representatives = []

    # For each group, compute the average and pick the file closest to that average
    for group in groups:
        if not group:
            continue
        counts = [x[1] for x in group]
        avg = np.mean(counts)
        rep = min(group, key=lambda x: abs(x[1] - avg))
        representatives.append(rep[0])

    # Display the three representative images
    group_names = ["Low density", "Medium density", "High density"]
    plt.figure(figsize=(15, 5))
    for idx, rep_file in enumerate(representatives):
        # Try finding an image with same base name and .png or .jpg extension
        base = os.path.splitext(rep_file)[0]
        img_path = None
        for ext in [".png", ".jpg"]:
            candidate = os.path.basename(base).replace("_som", "") + ext
            if os.path.exists(pt := os.path.join(directory, candidate)):
                img_path = pt
                break
        plt.subplot(1, 3, idx + 1)
        if img_path:
            img = plt.imread(img_path)
            plt.imshow(img)
            plt.title(group_names[idx])
        else:
            plt.text(
                0.5,
                0.5,
                f"No image found for\n{os.path.basename(rep_file)}",
                ha="center",
                va="center",
                fontsize=12,
            )
            plt.title(f"Group {idx + 1}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    return groups


def update_soms_with_density(file_path, density_level):
    """
    Update SOM JSON file with their corresponding density level.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

            # Add density information to the JSON
            data["DensityLevel"] = density_level

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error updating {file_path}: {e}")


if __name__ == "__main__":
    # Define input directories
    som_dir = "input/soms"

    # Get overall metrics
    comp_counts, depth_counter, class_counter, comp_info = analyze_soms(som_dir)

    # Plot overall metrics
    plot_metrics(comp_counts, depth_counter, class_counter, "Overall: ")

    # Get groups and show representative images
    groups = select_and_show_representative_images(
        "input/screen2som_dataset", comp_info
    )

    # Analyze and plot metrics for each group
    group_metrics = analyze_group_metrics(som_dir, groups)
    group_names = ["Low Density: ", "Medium Density: ", "High Density: "]

    for i, (group_comp_counts, group_depth_counter, group_class_counter) in enumerate(
        group_metrics
    ):
        plot_metrics(
            group_comp_counts, group_depth_counter, group_class_counter, group_names[i]
        )
