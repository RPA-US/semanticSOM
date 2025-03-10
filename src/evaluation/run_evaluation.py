import csv
import numpy as np
import json
import re
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


def run_evaluation(csv_path: str = CSV_FILE_PATH) -> None:
    # Define bins for quartile calculation
    bins = [0.25, 0.5, 0.75]
    scores = []
    quartile_scores = []

    model = TextModel("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
    model.manual_load()

    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ground_truth: str = row["GroundTruth"].lower()
            event_target: str = row["EventTarget"].lower()
            # Compute all similarity metrics
            metrics = {
                "Vector Similarity": vector_similarity(ground_truth, event_target),
                "Jaccard Similarity": jaccard_similarity(ground_truth, event_target),
                "Levenshtein Similarity": levenshtein_similarity(
                    ground_truth, event_target
                ),
                "SBERT Similarity": sbert_similarity(ground_truth, event_target),
                "MLLM": llm_eval(model, ground_truth, event_target) * 0.33,
            }

            print(f"Screenshot: {row['Screenshot']}")
            print(f"  Ground Truth: {ground_truth} | Event Target: {event_target}")
            # Prompt the user for a human evaluation score between 1 and 4
            while True:
                try:
                    human_input = input("    Please enter human eval score (1-4): ")
                    human_score = int(human_input) - 1
                    if 0 <= human_score <= 3:
                        break
                    else:
                        print("    Invalid input. Enter a number between 1 and 4.")
                except ValueError:
                    print("    Invalid input. Please enter an integer between 1 and 4.")

            metrics["Human"] = human_score * 0.33
            # Compute quartile for each metric
            quartiles = {
                name: np.digitize(score, bins) for name, score in metrics.items()
            }

            for name, score in metrics.items():
                print(f"    {name}: {score:.4f} (Quartile: {quartiles[name] + 1})")
            print("-----------------------------------------------------")

            scores.append(tuple(metrics.values()))
            quartile_scores.append(tuple(quartiles.values()))
    # Compute and print average scores
    avg_scores = [sum(vals) / len(vals) for vals in zip(*scores)]
    avg_quartiles = [sum(vals) / len(vals) * 0.25 for vals in zip(*quartile_scores)]

    print("Average Scores:")
    print(
        f"  Vector Similarity:    {avg_scores[0]:.4f} | Average Quartile: {avg_quartiles[0]:.4f}"
    )
    print(
        f"  Jaccard Similarity:   {avg_scores[1]:.4f} | Average Quartile: {avg_quartiles[1]:.4f}"
    )
    print(
        f"  Levenshtein Similarity: {avg_scores[2]:.4f} | Average Quartile: {avg_quartiles[2]:.4f}"
    )
    print(
        f"  SBERT Similarity:     {avg_scores[3]:.4f} | Average Quartile: {avg_quartiles[3]:.4f}"
    )
    print(
        f"  MLLM:                 {avg_scores[4]:.4f} | Average Quartile: {avg_quartiles[4]:.4f}"
    )
    print(
        f"  Human:                {avg_scores[5]:.4f} | Average Quartile: {avg_quartiles[5]:.4f}"
    )

    # Human-MLLM Agreement ratio
    human_scores = [score[5] for score in scores]
    mlm_scores = [score[4] for score in scores]
    human_mlm_agreement = sum(
        1 for human, mlm in zip(human_scores, mlm_scores) if human == mlm
    )
    human_mlm_agreement_ratio = human_mlm_agreement / len(human_scores)
    print(f"Human-MLLM Agreement Ratio: {human_mlm_agreement_ratio}")

    print("Evaluation complete.")


if __name__ == "__main__":
    run_evaluation()
