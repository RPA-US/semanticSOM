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
        for node in nodes:
            depth = node.get("depth")
            if depth is not None:
                depth_counter[depth] += 1
            cls = node.get("class")
            if cls is not None:
                class_counter[cls] += 1

    return comp_counts, depth_counter, class_counter, comp_info


def plot_metrics(comp_counts, depth_counter, class_counter):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    sns.histplot(comp_counts, edgecolor="black", kde=True)
    plt.xlabel("Components per Image")
    plt.ylabel("Frequency")
    plt.title("Distribution of Components per Image")

    plt.subplot(1, 3, 2)
    all_depths = [d for d, count in depth_counter.items() for _ in range(count)]
    sns.histplot(
        all_depths, bins=range(min(all_depths), max(all_depths) + 2), edgecolor="black"
    )
    plt.xlabel("Depth Level")
    plt.ylabel("Count")
    plt.title("Distribution of Depth Levels")

    plt.subplot(1, 3, 3)
    all_classes = [cls for cls, count in class_counter.items() for _ in range(count)]
    ax = sns.histplot(all_classes, stat="count", discrete=True, edgecolor="black")
    plt.xlabel("Component Class")
    plt.ylabel("Count")
    plt.yscale("log")
    plt.title("Number of Items per Class")
    plt.xticks(rotation=90)

    # Add counts on top of each bar
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{int(height)}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
            rotation=90,
        )

    plt.tight_layout()
    plt.show()


def select_and_show_representative_images(directory, comp_info):
    if not comp_info:
        print("No image data available.")
        return
    sorted_info = sorted(comp_info, key=lambda x: x[1])
    n = len(sorted_info)
    # Divide into (terciles)
    group1 = sorted_info[: n // 3]
    group2 = sorted_info[n // 3 : 2 * n // 3]
    group3 = sorted_info[2 * n // 3 :]
    groups = [group1, group2, group3]
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


if __name__ == "__main__":
    comp_counts, depth_counter, class_counter, comp_info = analyze_soms("input/soms")
    plot_metrics(comp_counts, depth_counter, class_counter)
    select_and_show_representative_images("input/screen2som_dataset", comp_info)
