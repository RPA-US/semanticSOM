import itertools
import traceback
import time
import os
import json
from datetime import datetime
from src.cfg import CFG
from src.models.models import QwenVLModel
from src.semantics.target_object import run_semantization


def run_config_combinations(
    skip_completed=True, output_log="output/config_run_log.json", start_from=None
):
    """
    Cycles through different configuration possibilities and runs semantization
    with each combination.
    """
    cot_options = [True, False]
    technique_options = [["som"], ["highlight"]]
    crop_options = ["parent", "none"]

    # List of models to use
    models = [
        # {"name": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ", "class": Qwen2_5VLModel},
        # {"name": "OpenGVLab/InternVL2_5-38B", "class": InternVLModel}, # Downgraded from 78B to 38B becuase it does not fit in memory, even when quantized
        {"name": "osunlp/UGround-V1-7B", "class": QwenVLModel},
        # {"name": "NVILA-15B", "class": VisionModel, "openai_server": "http://localhost:8000", "api_key": "fake-key"},
        # {"name": "nvidia/NVLM-D-72B", "class": NVLModel}, # We could not get the model to run
        # {"name": "Qwen/QVQ-72B-Preview", "class": QwenVLModel}, # This one is quantized to 8bits. Still, takes too long to generate any answers
        # Models from here on out are not expected to follow the output format
        # {"name": "bytedance-research/UI-TARS-7B-DPO", "class": QwenVLModel},
        # {"name": "xlangai/Aguvis-7B-720P", "class": QwenVLModel},
        # {"name": "OS-Copilot/OS-Atlas-Pro-7B", "class": QwenVLModel},
    ]

    # Keep track of original config to later restore it
    original_prompt_config = CFG.prompt_config.copy()

    all_combinations = list(
        itertools.product(cot_options, technique_options, crop_options)
    )
    total_configs = len(all_combinations) * len(models)

    results = []
    if os.path.exists(output_log):
        try:
            with open(output_log, "r") as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing results from {output_log}")
        except Exception:
            print(f"Could not load results from {output_log}, starting fresh")

    completed_configs = set(
        r["configuration"] for r in results if r.get("success", False)
    )

    print(f"Running semantization with {total_configs} different configurations...")
    print(f"{len(completed_configs)} configurations already completed")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    current_run = 0
    for model_info in models:
        model = model_info["class"](
            model_name=model_info["name"],
            openai_server=model_info.get("openai_server", None),
            api_key=model_info.get("api_key", None),
        )
        if not model.openai_server:
            model.manual_load()

        for cot, technique, crop in all_combinations:
            current_run += 1

            # Skip if we're starting from a specific configuration
            if start_from and current_run < start_from:
                continue

            CFG.prompt_config["cot"] = cot
            CFG.prompt_config["technique"] = technique
            CFG.prompt_config["crop"] = crop

            technique_str = "_".join(technique)
            model_name_short = model_info["name"].split("/")[-1].replace("-", "_")
            batch_name = f"cot_{cot}_technique_{technique_str}_crop_{crop}_model_{model_name_short}"

            if skip_completed and batch_name in completed_configs:
                print(
                    f"Configuration {current_run}/{total_configs}: {batch_name} already completed, skipping"
                )
                continue

            print(f"Running configuration {current_run}/{total_configs}: {batch_name}")

            try:
                # Run semantization
                start_time = time.time()
                run_semantization(model=model, batch_name=batch_name)
                end_time = time.time()

                result = {
                    "configuration": batch_name,
                    "cot": cot,
                    "technique": technique,
                    "crop": crop,
                    "model": model_info["name"],
                    "success": True,
                    "time_taken": end_time - start_time,
                    "timestamp": timestamp,
                }
                results.append(result)

                with open(output_log, "w") as f:
                    json.dump(results, f, indent=2)

                print(
                    f"Configuration {batch_name} completed in {end_time - start_time:.2f} seconds"
                )

            except Exception as e:
                print(f"Error with configuration {batch_name}: {e}")
                print(traceback.format_exc())

                result = {
                    "configuration": batch_name,
                    "cot": cot,
                    "technique": technique,
                    "crop": crop,
                    "model": model_info["name"],
                    "success": False,
                    "error": str(e),
                    "timestamp": timestamp,
                }
                results.append(result)

                # Save results after each run, even failures
                with open(output_log, "w") as f:
                    json.dump(results, f, indent=2)
        if not model.openai_server:
            model.manual_unload()

    # Reset config to original values
    CFG.prompt_config = original_prompt_config

    print("\n===== SUMMARY =====")
    print(f"Total configurations: {total_configs}")
    successful = [r for r in results if r.get("success", False)]
    failures = [r for r in results if not r.get("success", False)]
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failures)}")

    if failures:
        print("\nFailed configurations:")
        for failure in failures:
            print(
                f"- {failure['configuration']}: {failure.get('error', 'Unknown error')}"
            )

    print("\nAll configurations completed!")
    print(f"Results saved to {output_log}")


if __name__ == "__main__":
    run_config_combinations()
