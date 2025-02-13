import polars as pl

from src.models.models import Qwen2Model

from src.semantics.prompts import FROM_RAW, FROM_STATEMACHINE


def semantics_via_statemachine() -> None:
    model = Qwen2Model("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
    event_log: pl.DataFrame = pl.read_csv(
        source="input/phase_3/email_semantized.csv", separator=","
    )
    with pl.Config(
        tbl_formatting="MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        tbl_cols=-1,
        tbl_rows=-1,
    ):
        str_repr: str = event_log.__str__()
    enriched_log = model(str_repr, sys_prompt=FROM_STATEMACHINE)
    print(enriched_log)


def full_llm() -> None:
    model = Qwen2Model("Qwen/Qwen2.5-7B-Instruct")
    event_log: pl.DataFrame = pl.read_csv(
        source="input/phase_3/brandon_toy_example.csv", separator=","
    )
    with pl.Config(
        tbl_formatting="MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        tbl_cols=-1,
        tbl_rows=-1,
    ):
        str_repr: str = event_log.__str__()
    enriched_log = model(str_repr, sys_prompt=FROM_RAW)
    print(enriched_log)


def main() -> None:
    semantics_via_statemachine()


if __name__ == "__main__":
    main()
