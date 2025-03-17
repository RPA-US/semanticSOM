# Base comes from commit b55c3c2
from argparse import ArgumentParser

from accelerate import (  # noqa
    init_empty_weights,  # noqa
    infer_auto_device_map,  # noqa
    load_checkpoint_and_dispatch,  # noqa
)

# ruff: on
import json
from tqdm import tqdm
import os
from src.mllm_judge.prompt import get_prompt

from src.mllm_judge.get_vlm_res import (  # noqa
    gpt_4v,  # noqa
    gpt_4o,  # noqa
    llava_1_6_34b,  # noqa
    llava_1_6_13b,  # noqa
    llava_1_6_7b,  # noqa
    qwen_vl_plus,  # noqa
    qwen_vl_max,  # noqa
    local_qwen_vl,  # noqa
    local_deepseek_api_14b,  # noqa
)

from src.cfg import CFG

import time


def retry(attempts=3, delay=10):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {i + 1}/{attempts} failed: {e}")
                    if i < attempts - 1:
                        time.sleep(delay)  # Wait a bit before retrying
            return None

        return wrapper

    return decorator


@retry(3)
def get_res(
    model, image_path, prompt, api, temperature, top_p, llm_assist: bool = False
):
    func_name: str = model.replace("-", "_").replace(".", "_")
    func = globals().get(func_name)
    if not callable(func):
        raise ValueError(f"Invalid model name: {model}")
    output = func(image_path, prompt, api, temperature, top_p)  # type: ignore
    try:
        json_output = json.loads(output)
        return output, json_output
    except KeyError:
        if llm_assist:
            pass
        return output, None


def construct_input(prompt_dict, judge_mode, setting, instruction, responses):
    prompt = (
        prompt_dict["start"]
        + "\nEvaluation Steps:\n"
        + prompt_dict["setting"][setting]
        + "\nEvaluation Method:\n"
        + prompt_dict["tasks"][judge_mode]
        + "\nNotice:\n"
        + prompt_dict["notice"]
        + "\nHere is the input:\n"
    )
    if judge_mode == "score":
        prompt += f"""
[The Start of User Instruction]
{instruction}
[The End of User Instruction]
[The Start of Assistant’s Answer]
{responses[0]}
[The End of Assistant’s Answer]"""
    elif judge_mode == "pair":
        prompt += f"""
[The Start of User Instruction]
{instruction}
[The End of User Instruction]
[The Start of Assistant A’s Answer]
{responses[0]}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{responses[1]}
[The End of Assistant B’s Answer]"""
    elif judge_mode == "ground_truth":
        prompt += f"""
[The Start of User Instruction]
{instruction}
[The End of User Instruction]
[The Start of Ground Truth Answer]
{responses[0]}
[The End of Ground Truth Answer]
[The Start of Assistant's Answer]
{responses[1]}
[The End of Assistant's Answer]"""
    elif judge_mode == "batch":
        prompt += f"""
[The Start of User Instruction]
{instruction}
[The End of User Instruction]"""
        assistant_name = "A"
        num_assistant = 0
        for i in range(len(responses)):
            prompt += f"[The Start of Assistant {assistant_name}’s Answer]\n"
            prompt += responses[num_assistant] + "\n"
            prompt += f"[The End of Assistant {assistant_name}’s Answer]\n"
            assistant_name = chr(ord(assistant_name) + 1)
            num_assistant += 1
    return prompt


def benchmark(model, judge_mode, setting, api, image_dir, temperature, top_p):
    items = []
    with open(CFG.eval_config["jsonl_path"], "r") as json_file:
        for line in json_file:
            items.append(json.loads(line))

    output_path = CFG.eval_config["output_path"]
    folder_path = os.path.dirname(output_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    prompt_dict = get_prompt()
    for item in tqdm(items, desc="Processing items"):
        image_path = os.path.join(image_dir, item["image_path"])
        if judge_mode == "score":
            responses = [item["answer"]]
        elif judge_mode == "pair":
            responses = [item["answer1"]["answer"], item["answer2"]["answer"]]
        elif judge_mode == "ground_truth":
            responses = [item["human"]["answer"], item["assistant"]["answer"]]
        elif judge_mode == "batch":
            responses = [i["answer"] for i in item["answers"]]
        prompt = construct_input(
            prompt_dict, judge_mode, setting, item["instruction"], responses=responses
        )
        print(prompt)
        raw_response, json_response = get_res(
            model, image_path, prompt, api, temperature, top_p
        )
        item["mllm_judge"] = raw_response
        # item['json_mllm_judge'] = json_response
        with open(output_path, "a") as jsonl_file:
            jsonl_file.write(json.dumps(item) + "\n")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The path to the JSON file containing the items to evaluate.",
    )
    parser.add_argument(
        "--judge_mode",
        type=str,
        default=None,
        help="The path to the JSON file containing the items to evaluate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="The temperature to use for inference.",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.4, help="The top-p to use for inference."
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=CFG.image_dir,
        help="The root directory for the images.",
    )
    parser.add_argument(
        "--setting", type=str, default="No Figure", help="The setting of the evaluation"
    )
    parser.add_argument("--api", type=str, default=None, help="API for inference.")
    args = parser.parse_args()
    assert args.judge_mode in ["score", "batch", "pair", "ground_truth"], (
        "Invalid judge mode"
    )
    assert args.model in [
        "gemini",
        "gpt-4v",
        "gpt-4o",
        "llava-1.6-34b",
        "llava-1.6-13b",
        "llava-1.6-7b",
        "qwen-vl-plus",
        "qwen-vl-max",
        "qwen-vl-chat",
        "local-qwen-vl",
        "local-deepseek-api-14b",
    ]

    benchmark(
        args.model,
        args.judge_mode,
        args.setting,
        args.api,
        args.image_root,
        args.temperature,
        args.top_p,
    )


if __name__ == "__main__":
    main()
