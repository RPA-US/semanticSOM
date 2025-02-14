import polars as pl

from src.models.models import Qwen2Model

from src.semantics.prompts import FROM_RAW, FROM_STATEMACHINE


def semantics_via_statemachine() -> None:
    """
    Enriches the event log using a state machine and a language model.
    """
    model = Qwen2Model("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
    assert isinstance(model, Qwen2Model), "model must be an instance of Qwen2Model"
    event_log: pl.DataFrame = pl.read_csv(
        source="input/phase_3/email_semantized.csv", separator=","
    )
    assert isinstance(event_log, pl.DataFrame), "event_log must be a Polars DataFrame"
    with pl.Config(
        tbl_formatting="MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        tbl_cols=-1,
        tbl_rows=-1,
    ):
        str_repr: str = event_log.__str__()
    enriched_log = model(str_repr, sys_prompt=FROM_STATEMACHINE)
    assert isinstance(enriched_log, str), "enriched_log must be a string"
    print(enriched_log)


def full_llm() -> None:
    """
    Enriches the event log using a full language model.
    """
    model = Qwen2Model("Qwen/Qwen2.5-7B-Instruct")
    assert isinstance(model, Qwen2Model), "model must be an instance of Qwen2Model"
    event_log: pl.DataFrame = pl.read_csv(
        source="input/phase_3/brandon_toy_example.csv", separator=","
    )
    assert isinstance(event_log, pl.DataFrame), "event_log must be a Polars DataFrame"
    with pl.Config(
        tbl_formatting="MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        tbl_cols=-1,
        tbl_rows=-1,
    ):
        str_repr: str = event_log.__str__()
    enriched_log = model(str_repr, sys_prompt=FROM_RAW)
    assert isinstance(enriched_log, str), "enriched_log must be a string"
    print(enriched_log)


def main() -> None:
    """
    Main function to run the semantics enrichment.
    """
    assert callable(semantics_via_statemachine), "semantics_via_statemachine must be callable"
    semantics_via_statemachine()


if __name__ == "__main__":
    main()
