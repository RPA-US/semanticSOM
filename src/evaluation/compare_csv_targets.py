import pandas as pd
import argparse
from sentence_transformers import SentenceTransformer, util
from typing import Tuple


def load_csv_files(
    file1_path: str, file2_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load two CSV files containing EventTarget columns

    Args:
        file1_path: Path to the first CSV file
        file2_path: Path to the second CSV file

    Returns:
        Tuple of two pandas DataFrames
    """
    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)

        if "EventTarget" not in df1.columns or "EventTarget" not in df2.columns:
            raise ValueError("Both CSV files must contain an 'EventTarget' column")

        return df1, df2
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        raise


def compute_similarity(text1: str, text2: str, model) -> float:
    """
    Compute semantic similarity between two text strings

    Args:
        text1: First text string
        text2: Second text string
        model: SentenceTransformer model

    Returns:
        Similarity score between 0 and 1
    """
    try:
        embedding1 = model.encode(text1, convert_to_tensor=True)
        embedding2 = model.encode(text2, convert_to_tensor=True)

        similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
        return similarity
    except Exception as e:
        print(f"Error computing similarity: {e}")
        return 0.0


def compare_event_targets(
    df1: pd.DataFrame, df2: pd.DataFrame, output_path: str | None = None
) -> pd.DataFrame:
    """
    Compare EventTarget columns between two DataFrames

    Args:
        df1: First DataFrame with EventTarget column
        df2: Second DataFrame with EventTarget column
        output_path: Optional path to save comparison results

    Returns:
        DataFrame with comparison results
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")

    results = pd.DataFrame(
        columns=["Screenshot", "Coords", "EventTarget1", "EventTarget2", "Similarity"]
    )

    screenshots1 = df1["Screenshot"].unique()
    screenshots2 = df2["Screenshot"].unique()
    common_screenshots = set(screenshots1).intersection(set(screenshots2))

    for screenshot in common_screenshots:
        df1_screenshot = df1[df1["Screenshot"] == screenshot]
        df2_screenshot = df2[df2["Screenshot"] == screenshot]

        for _, row1 in df1_screenshot.iterrows():
            coords = row1["Coords"]

            matching_rows = df2_screenshot[df2_screenshot["Coords"] == coords]

            if not matching_rows.empty:
                row2 = matching_rows.iloc[0]

                target1 = row1["EventTarget"]
                target2 = row2["EventTarget"]

                similarity = compute_similarity(target1, target2, model)

                new_row = {
                    "Screenshot": screenshot,
                    "Coords": coords,
                    "EventTarget1": target1,
                    "EventTarget2": target2,
                    "Similarity": similarity,
                }

                results = pd.concat(
                    [results, pd.DataFrame([new_row])], ignore_index=True
                )

    if output_path:
        results.to_csv(output_path, index=False)
        print(f"Comparison results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare EventTarget columns between two CSV files"
    )
    parser.add_argument("--file1", required=True, help="Path to the first CSV file")
    parser.add_argument("--file2", required=True, help="Path to the second CSV file")
    parser.add_argument("--output", help="Path to save the comparison results")
    args = parser.parse_args()

    df1, df2 = load_csv_files(args.file1, args.file2)
    results = compare_event_targets(df1, df2, args.output)

    print(f"Total entries compared: {len(results)}")
    print(f"Average similarity score: {results['Similarity'].mean():.4f}")
    print(f"Minimum similarity score: {results['Similarity'].min():.4f}")
    print(f"Maximum similarity score: {results['Similarity'].max():.4f}")

    if not results.empty:
        print("\nExamples of high similarity:")
        high_sim = results.nlargest(3, "Similarity")
        for _, row in high_sim.iterrows():
            print(f"- {row['Screenshot']}, {row['Coords']}: {row['Similarity']:.4f}")
            print(f'  Target1: "{row["EventTarget1"]}"')
            print(f'  Target2: "{row["EventTarget2"]}"')

        print("\nExamples of low similarity:")
        low_sim = results.nsmallest(3, "Similarity")
        for _, row in low_sim.iterrows():
            print(f"- {row['Screenshot']}, {row['Coords']}: {row['Similarity']:.4f}")
            print(f'  Target1: "{row["EventTarget1"]}"')
            print(f'  Target2: "{row["EventTarget2"]}"')


if __name__ == "__main__":
    main()
