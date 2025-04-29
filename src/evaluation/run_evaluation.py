import csv
import numpy as np
import json
import re
import argparse
from src.evaluation.evaluation_vector_similarity import vector_similarity
from src.evaluation.evaluation_jaccard_similarity import jaccard_similarity
from src.evaluation.evaluation_levenshtein_similarity import levenshtein_similarity
from src.evaluation.evaluation_sbert_similarity import sbert_similarity
from src.models.models import TextModel
from src.mllm_judge.api_benchmarks import construct_input
from src.mllm_judge.prompt import get_prompt

CSV_FILE_PATH: str = "c:/apa/BPMExtension/output/eval.csv"


def llm_eval(model, ground_truth, event_target):
    prompt_dict = get_prompt()
    prompt = construct_input(
        prompt_dict=prompt_dict,
        judge_mode="ground_truth",
        setting="No Figure",
        instruction=None,
        responses=[ground_truth, event_target],
    )
    res = model(prompt)
    if match := re.search(r"{(.*?)}", res, re.DOTALL):
        res = match.group()
    else:
        raise Exception("LLM did not produce a valid response")
    json_res = json.loads(res)
    judgement = json_res["Judgement"].split("[[")[-1].split("]]")[0]
    return int(judgement)


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation with selected metrics")
    parser.add_argument(
        "--csv-path", type=str, default=CSV_FILE_PATH, help="Path to CSV file"
    )
    parser.add_argument(
        "--vector", action="store_true", help="Run Vector Similarity evaluation"
    )
    parser.add_argument(
        "--jaccard", action="store_true", help="Run Jaccard Similarity evaluation"
    )
    parser.add_argument(
        "--levenshtein",
        action="store_true",
        help="Run Levenshtein Similarity evaluation",
    )
    parser.add_argument(
        "--sbert", action="store_true", help="Run SBERT Similarity evaluation"
    )
    parser.add_argument("--mllm", action="store_true", help="Run MLLM evaluation")
    parser.add_argument("--human", action="store_true", help="Include Human evaluation")
    parser.add_argument("--all", action="store_true", help="Run all evaluation metrics")
    parser.add_argument(
        "--inferred",
        type=str,
        default="EventTarget",
        help="Column name containing inferred/generated text (default: EventTarget)",
    )
    parser.add_argument(
        "--unique",
        action="store_true",
        help="Use unique ScreenID for each row",
    )
    return parser.parse_args()


def run_evaluation(
    csv_path: str = CSV_FILE_PATH,
    use_vector: bool = False,
    use_jaccard: bool = False,
    use_levenshtein: bool = False,
    use_sbert: bool = False,
    use_mllm: bool = False,
    use_human: bool = False,
    inferred_column: str = "EventTarget",
    unique_screenid: bool = False,
) -> None:
    # Define bins for quartile calculation
    bins = [0.25, 0.5, 0.75]
    scores: list[tuple] = []
    unique_scores = []
    quartile_scores: list[tuple] = []
    unique_quartile_scores = []

    metric_functions = {}

    if (
        use_vector
        or use_jaccard
        or use_levenshtein
        or use_sbert
        or use_mllm
        or use_human
    ):
        if use_vector:
            metric_functions["Vector Similarity"] = vector_similarity
        if use_jaccard:
            metric_functions["Jaccard Similarity"] = jaccard_similarity
        if use_levenshtein:
            metric_functions["Levenshtein Similarity"] = levenshtein_similarity
        if use_sbert:
            metric_functions["SBERT Similarity"] = sbert_similarity
    else:
        # Default behavior if no flags are set: ask for one of the options
        print("Please select at least one evaluation metric to run.")
        print("Options:")
        print("  --vector:      Vector Similarity")
        print("  --jaccard:     Jaccard Similarity")
        print("  --levenshtein: Levenshtein Similarity")
        print("  --sbert:       SBERT Similarity")
        print("  --mllm:        MLLM Evaluation")
        print("  --human:       Human Evaluation")
        print("  --all:         Run all evaluation metrics")
        print(
            "  --inferred:     Column name for inferred/generated text (default: EventTarget)"
        )
        print("  --unique:      Unique ScreenID for each row")
        return

    model = None
    if use_mllm:
        model = TextModel("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
        model.manual_load()

    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        screenids = []
        for row in reader:
            if "ScreenID" in row and row["ScreenID"] not in screenids:
                screenids.append(row["ScreenID"])
            elif "ScreenID" in row:
                scores.append(scores[-1])
                quartile_scores.append(quartile_scores[-1])
                continue
            ground_truth: str = row["GroundTruth"].lower()
            event_target: str = row[inferred_column].lower()

            # Compute selected similarity metrics
            metrics = {}
            for name, func in metric_functions.items():
                metrics[name] = func(ground_truth, event_target)

            if use_mllm:
                metrics["MLLM"] = llm_eval(model, ground_truth, event_target) * 0.33

            print(f"Screenshot: {row['Screenshot']}")
            print(f"  Ground Truth: {ground_truth} | {inferred_column}: {event_target}")

            if use_human:
                while True:
                    try:
                        human_input = input("    Please enter human eval score (1-4): ")
                        human_score = int(human_input) - 1
                        if 0 <= human_score <= 3:
                            break
                        else:
                            print("    Invalid input. Enter a number between 1 and 4.")
                    except ValueError:
                        print(
                            "    Invalid input. Please enter an integer between 1 and 4."
                        )
                metrics["Human"] = human_score * 0.33

            quartiles = {
                name: np.digitize(score, bins) for name, score in metrics.items()
            }

            for name, score in metrics.items():
                print(f"    {name}: {score:.4f} (Quartile: {quartiles[name] + 1})")
            print("-----------------------------------------------------")

            scores.append(tuple(metrics.values()))
            unique_scores.append(tuple(metrics.values()))
            quartile_scores.append(tuple(quartiles.values()))
            unique_quartile_scores.append(tuple(quartiles.values()))

    if scores:
        if unique_screenid:
            avg_scores = [sum(vals) / len(vals) for vals in zip(*unique_scores)]
            avg_quartiles = [
                sum(vals) / len(vals) for vals in zip(*unique_quartile_scores)
            ]
            std_devs = [np.std(vals) for vals in zip(*unique_scores)]
        else:
            avg_scores = [sum(vals) / len(vals) for vals in zip(*scores)]
            avg_quartiles = [sum(vals) / len(vals) for vals in zip(*quartile_scores)]
            std_devs = [np.std(vals) for vals in zip(*scores)]

        print("Average Scores:")
        metric_names = list(metrics.keys())
        for i, name in enumerate(metric_names):
            print(
                f"  {name}: {avg_scores[i]:.4f} | Average Quartile: {avg_quartiles[i]:.4f} | Std Dev: {std_devs[i]:.4f}"
            )

        # Save scores to same CSV in new columns
        with open(csv_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            assert isinstance(reader.fieldnames, list)
            fieldnames = reader.fieldnames + metric_names  # noqa
            new_csvfile = csv_path.replace(".csv", "_scores.csv")
            with open(new_csvfile, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row, row_scores in zip(reader, scores):
                    row.update(
                        {name: score for name, score in zip(metric_names, row_scores)}
                    )
                    writer.writerow(row)

        if use_human and use_mllm:
            human_index = metric_names.index("Human")
            mllm_index = metric_names.index("MLLM")
            human_scores = [score[human_index] for score in scores]
            mllm_scores = [score[mllm_index] for score in scores]
            human_mllm_agreement = sum(
                1 for human, mllm in zip(human_scores, mllm_scores) if human == mllm
            )
            human_mllm_agreement_ratio = human_mllm_agreement / len(human_scores)
            print(f"Human-MLLM Agreement Ratio: {human_mllm_agreement_ratio}")

    print("Evaluation complete.")


if __name__ == "__main__":
    args = parse_args()

    if args.all:
        run_evaluation(
            csv_path=args.csv_path,
            use_vector=True,
            use_jaccard=True,
            use_levenshtein=True,
            use_sbert=True,
            use_mllm=True,
            use_human=True,
            inferred_column=args.inferred,
            unique_screenid=args.unique,
        )
    else:
        run_evaluation(
            csv_path=args.csv_path,
            use_vector=args.vector,
            use_jaccard=args.jaccard,
            use_levenshtein=args.levenshtein,
            use_sbert=args.sbert,
            use_mllm=args.mllm,
            use_human=args.human,
            inferred_column=args.inferred,
            unique_screenid=args.unique,
        )
