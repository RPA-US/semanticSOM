import polars as pl
import json
import re

from tqdm import tqdm
from timeit import default_timer as timer

from src.models.models import TextModel

from src.semantics.prompts import (
    NAIVE_ACTIVITY_LABELING,
    NO_COT_NAIVE_ACTIVITY_LABELING,
)

from src.cfg import CFG

from PIL import Image


def main(model, batch_name, cot, **kwargs) -> None:
    """
    Main function to run the semantics enrichment.
    """
    model.manual_load()

    event_log: pl.DataFrame = pl.read_csv(
        source="input/phase_3/eval.csv", separator=";"
    )
    event_log = event_log.with_columns(ActivityLabel=pl.lit(""), Time=pl.lit(0.0))

    res_log = pl.DataFrame(schema=event_log.schema)
    assert isinstance(event_log, pl.DataFrame), "event_log must be a Polars DataFrame"

    activities = event_log.group_by("ScreenID")

    for _, group_df in tqdm(activities, desc="Processing activities"):
        events_list = []
        for row in group_df.rows(named=True):
            event = {
                "EventType": row["EventType"],
                "EventDescription": row["EventDescription"],
                "Text": row["Text"],
            }

            events_list.append(event)
        input_events = json.dumps(events_list, sort_keys=True, indent=4)
        prompt = f"\nGiven the following events, label the activity:\n {input_events}"

        if "image" in model.capabilities:
            image = Image.open(f"{CFG.image_dir}/{group_df['Screenshot'][0]}")

        sys_prompt = NAIVE_ACTIVITY_LABELING if cot else NO_COT_NAIVE_ACTIVITY_LABELING

        start = timer()
        model_output: str = ""
        if "image" in model.capabilities:
            model_output = model(
                prompt=prompt, sys_prompt=sys_prompt, image=image, **kwargs
            )
        else:
            model_output = model(prompt=prompt, sys_prompt=sys_prompt, **kwargs)
        end = timer()
        time_taken = end - start
        print(f"{model_output}")

        respose = "<error> No response from model"
        if (
            match_group := re.search(
                pattern=r"<\|activity_label\|>(.*)<\|end_activity_label\|>",
                string=model_output.lower(),  # A common hallucination is to set some letters to uppercase
            )
        ):
            respose = match_group.group(1)

        # Finally, we set in all the rows of the activity the same label and time
        group_df = group_df.with_columns(
            ActivityLabel=pl.lit(respose), Time=pl.lit(time_taken)
        )

        res_log = res_log.vstack(group_df)

    model.manual_unload()

    res_log.sort(by="ScreenID").write_csv(f"output/phase_3/eval_{batch_name}.csv")


if __name__ == "__main__":
    # print("Using Athene-V2-Chat")
    # model = TextModel("Nexusflow/Athene-V2-Chat")
    # main(model, "Athene-V2-Chat", True)

    # print("Using Athene-70B")
    # model = TextModel("Nexusflow/Athene-70B")
    # main(model, "Athene-70B", True, padding=False)

    # print("Using Qwen-72B-Instruct")
    # model = TextModel("Qwen/Qwen2.5-72B-Instruct")
    # main(model, "Qwen2.5-72B-Instruct", True)

    print("Using Llama-3.1-Nemotron-70B-Instruct-HF")
    model = TextModel("nvidia/Llama-3.1-Nemotron-70B-Instruct-HF")
    main(model, "Llama-3.1-Nemotron-70B-Instruct-HF", True, padding=False)
