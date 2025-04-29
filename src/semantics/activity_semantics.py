import polars as pl
import json
import re

from timeit import default_timer as timer

from src.models.models import TextModel

from src.semantics.prompts import NAIVE_ACTIVITY_LABELING


def main() -> None:
    """
    Main function to run the semantics enrichment.
    """
    model = TextModel("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
    event_log: pl.DataFrame = pl.read_csv(
        source="input/phase_3/eval.csv", separator=";"
    )
    event_log = event_log.with_columns(ActivityLabel=pl.lit(""), Time=pl.lit(0.0))

    res_log = pl.DataFrame(schema=event_log.schema)
    assert isinstance(event_log, pl.DataFrame), "event_log must be a Polars DataFrame"

    activities = event_log.group_by("ScreenID")

    for _, group_df in activities:
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

        start = timer()
        model_output: str = model(prompt=prompt, sys_prompt=NAIVE_ACTIVITY_LABELING)
        end = timer()
        time_taken = end - start
        assert isinstance(model_output, str), "model_output must be a string"

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

    res_log.write_csv("output/phase_3/eval_semantics.csv")


if __name__ == "__main__":
    main()
