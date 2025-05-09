#   Here we compute the semantics for the actions given a log and object descriptions of the objects the user has interacted with

#   We design this process as a state machine that goes over each individual event, and depending on the action type, we update the state of the objects in the scene

import re
import os

from enum import Enum
from typing import Any

import polars as pl

from statemachine import StateMachine, State
from statemachine.transition_list import TransitionList

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


class Action(Enum):
    """
    Enum representing different types of actions.
    """

    RIGHT_CLICK = 1
    LEFT_CLICK = 2
    TYPE = 3
    PRESS = 4
    SHORTCUT = 5
    TAB = 6
    SUBMIT = 7


class EventState(StateMachine):
    """
    State machine for handling event states and transitions.
    """

    # Define the stuff we will need to store in the state
    interaction_target: str | None = None

    # Define the states
    free = State(
        name="free", initial=True
    )  # Meaning the user has not selected any item to interact with. Here we expect either a click or a keyboard event (shortcut || tab)
    interaction = State(
        name="interaction"
    )  # Meaning the user has selected an item to interact with. Here we expect a click event to start it and then a keyboard event to type on it or another click
    context_menu = State(
        name="context_menu"
    )  # Meaning the user has selected an item with a right click and is interacting with the context menu. Here we expect a click event
    typing = State(name="typing")  # Meaning the user has selected something to type on
    submit = State(name="submit")  # Meaning the user has submitted something

    # Define the transitions
    click: TransitionList = (
        free.to(interaction)
        | context_menu.to(free)
        | typing.to(interaction)
        | interaction.to(interaction)
    )
    shortcut: TransitionList = free.to(free) | interaction.to(free) | typing.to(free)
    tab: TransitionList = interaction.to(interaction) | typing.to(typing)
    enter: TransitionList = typing.to(submit)
    right_click: TransitionList = (
        free.to(context_menu)
        | interaction.to(context_menu)
        | context_menu.to(context_menu)
        | typing.to(context_menu)
    )
    keypress: TransitionList = interaction.to(typing)
    write: TransitionList = typing.to(typing) | interaction.to(typing)

    def on_enter_free(self) -> None:
        """
        Handle entering the 'free' state.
        """
        self.interaction_target = None

    def on_enter_interaction(self, log_event: dict) -> None:
        """
        Handle entering the 'interaction' state.

        Args:
            log_event (dict): The log event triggering the state change.
        """
        self.interaction_target = log_event["EventTarget"]

    def on_exit_typing(
        self, reset_target: bool = True
    ) -> None:  # The target is kept until we stop interacting with an object but not when exiting interaction as we might write on it
        self.interaction_target = None if reset_target else self.interaction_target

    def on_enter_context_menu(self, log_event: dict) -> None:
        """
        Handle entering the 'context_menu' state.

        Args:
            log_event (dict): The log event triggering the state change.
        """
        self.interaction_target = log_event["EventTarget"]

    def on_exit_context_menu(self) -> None:
        """
        Handle exiting the 'context_menu' state.
        """
        self.interaction_target = None

    def on_enter_submit(self) -> None:
        """
        Handle entering the 'submit' state.
        """
        self.interaction_target = None
        self.send("free")


class Semantizer:
    """
    Class for semantizing event logs.
    """

    current_event: dict[str, Any]
    current_action: Action | None

    def __init__(self) -> None:
        self.current_event = {}
        self.current_action = None

    def calc_event(self, event: dict) -> None:
        """
        Calculate the current event and update the state machine.

        Args:
            event (dict): The event to process.
        """
        assert isinstance(event, dict), "event must be a dictionary"
        assert "EventType" in event, "event must contain 'EventType' key"
        self.current_event = event
        match event["EventType"]:
            case "right_click":
                self.current_action = Action.RIGHT_CLICK
                global_state.send("right_click", log_event=event)
            case "left_click" | "click" | "double_click":
                self.current_action = Action.LEFT_CLICK
                global_state.send("click", log_event=event)
            case "keyboard":
                self.current_action = self.infer_keyboard_event(event["Text"].lower())
            case _:
                raise ValueError("Invalid action type")
        assert self.current_action is not None, "current_action must not be None"

    def next(self, event: dict) -> None:
        assert isinstance(event, dict), "event must be a dictionary"
        if self.current_event:
            self.previous_event = self.current_event
        self.calc_event(event)
        assert self.current_event is not None, "current_event must not be None"

    def infer_keyboard_event(self, text: str) -> Action:
        assert isinstance(text, str), "text must be a string"
        shortcut_regex = r"^(ctrl|alt|super|meta)"
        if re.match(shortcut_regex, text):
            global_state.send("shortcut")
            return Action.SHORTCUT
        elif len(text) == 1:
            global_state.send("keypress")
            return Action.PRESS
        match text:
            case "tab":
                global_state.send("tab", reset_target=False)
                return Action.TAB
            case "enter":
                global_state.send("enter")
                return Action.SUBMIT
            case _:
                global_state.send("write")
                return Action.TYPE
        assert isinstance(self.current_action, Action), "current_action must be an instance of Action"
        return self.current_action

    def construct_event_description(self) -> str:
        assert self.current_action is not None, "current_action must not be None"
        match self.current_action:
            case Action.RIGHT_CLICK:
                return f"Right clicked on {self.current_event['EventTarget']}"
            case Action.LEFT_CLICK:
                return f"Left clicked on {self.current_event['EventTarget']}"
            case Action.PRESS:
                return f"Pressed the {self.current_event['Text']} key"
            case Action.SHORTCUT:
                return f"Pressed the {self.current_event['Text']} shortcut"
            case Action.TAB:
                if global_state.interaction_target:
                    return (
                        f"Tabbed to next element from {global_state.interaction_target}"
                    )
                else:
                    return "Pressed the tab key"
            case Action.SUBMIT:
                return "Submitted information"
            case Action.TYPE:
                if global_state.interaction_target:
                    return f"Typed {self.current_event['Text']} on {global_state.interaction_target}"
                else:
                    return f"Typed {self.current_event['Text']}"
            case _:
                return "Invalid action type"

    def semantize_log(self, event_log: pl.DataFrame) -> pl.DataFrame:
        """
        Semantizes the event log.

        Args:
            event_log (pl.DataFrame): The event log to process.

        Returns:
            pl.DataFrame: The semantized event log.
        """
        assert isinstance(event_log, pl.DataFrame), "event_log must be a Polars DataFrame"
        event_desc_col: list[str] = []
        for event in event_log.iter_rows(named=True):  # event: dict[str, Any]
            self.next(event)
            event_desc_col.append(self.construct_event_description())

        res: pl.DataFrame = event_log.with_columns(
            pl.Series("EventDescription", event_desc_col).alias("EventDescription")
        )
        assert isinstance(res, pl.DataFrame), "res must be a Polars DataFrame"
        return res


if __name__ == "__main__":
    global_state = EventState()
    semantizer = Semantizer()
    event_log: pl.DataFrame = pl.read_csv(
        source="input/phase_3/email.csv", separator=","
    )
    semantized_log: pl.DataFrame = semantizer.semantize_log(event_log=event_log)
    print(semantized_log)
