# This file will contain all code to load the original screen2som dataset, process it, grab unique, interactable components (Element Level), prompt the user for labels, and save the data in csv format following:
# | Screenshot | EventType  | Coords | GroundTruth |
# |------------|------------|--------|-------------|
# |    ...     | left_click |   ...  |     ...     |

import os
import json
import shutil
from src.cfg import CFG
import polars as pl
from timeit import default_timer as timer
import tkinter as tk
from collections import Counter
from PIL import Image, ImageDraw
from PIL import ImageTk as itk
from typing import Any
# from src.utils.images import ImageCache

from src.utils.hierarchy_constructor import convert_to_som

_LIMIT_PER_DENSITY = 15

_INTERACTABLE_ELEMENT_CLASSES = [
    "BtnSq",
    "BtnPill",
    "BtnCirc",
    # "Icon",
    # "WebIcon",
    # "Switch",
    # "CheckboxChecked",
    # "CheckboxUnchecked",
    # "RadiobtnSelected",
    # "RadiobtnUnselected",
    "TextInput",
    # "Dropdown",
    # "Link",
    # "Image",
    # "BrowserUrlInput",
]

_CLASS_MAPPING = {
    "BtnSq": "Button",
    "BtnPill": "Button",
    "BtnCirc": "Button",
    "Icon": "Icon",
    "WebIcon": "Icon",
    "Switch": "Switch",
    "CheckboxChecked": "Checkbox",
    "CheckboxUnchecked": "Checkbox",
    "RadiobtnSelected": "Radio Button",
    "RadiobtnUnselected": "Radio Button",
    "TextInput": "Text Input",
    "Dropdown": "Dropdown",
    "Link": "Link",
}


def _preprocess_dataset() -> tuple[list[str], list[str]]:
    """
    Preprocess the dataset by converting JSON files to SOM format and copying image files to the image directory.

    Returns:
        tuple[list[str], list[str]]: A tuple containing a list of SOM file names and a list of image file names.
    """
    soms: list[str] = []
    imgs: list[str] = []
    for file in os.listdir(CFG.s2s_dataset_dir):
        if file.endswith(".json"):
            som_file = f"{CFG.som_dir}/{file.replace('.json', '_som.json')}"
            if not os.path.exists(som_file):
                convert_to_som(
                    f"{CFG.s2s_dataset_dir}/{file}",
                    som_file,
                )
            soms.append(file.replace(".json", "_som.json"))
        elif file.endswith(".png"):
            img_file = f"{CFG.image_dir}/{file}"
            if not os.path.exists(img_file):
                shutil.copy(f"{CFG.s2s_dataset_dir}/{file}", img_file)
            imgs.append(img_file)
    return soms, imgs


def _get_interactable_elements(som: dict[str, Any]) -> list[dict]:
    """
    Extract interactable elements from the SOM data.

    Args:
        som (dict): The SOM data.

    Returns:
        list[dict]: A list of interactable elements.
    """
    assert isinstance(som, dict), "som must be a dictionary"
    assert "compos" in som and isinstance(som["compos"], list), (
        "som must contain 'compos' key with a list of components"
    )
    assert all(
        isinstance(compo, dict) and "class" in compo for compo in som["compos"]
    ), "each component in 'compos' must be a dictionary containing 'class' key"

    return [
        compo
        for compo in som["compos"]
        if compo["class"] in _INTERACTABLE_ELEMENT_CLASSES
    ]


def _bbox_from_object(obj: dict) -> tuple[int, int, int, int]:
    """
    Extract the upper leftmost, lower rightmost coordinates of the object from its list of points.

    Args:
        obj (dict): The object to extract the bounding box from.

    Returns:
        tuple[int, int, int, int]: The bounding box coordinates (x_min, y_min, x_max, y_max).
    """
    assert isinstance(obj, dict), "obj must be a dictionary"
    assert "points" in obj, "obj must contain 'points' key"

    x_coords = [point[0] for point in obj["points"]]
    y_coords = [point[1] for point in obj["points"]]
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)


class LabelingInterface:
    dataframe = pl.DataFrame(
        schema={
            CFG.colnames["Screenshot"]: pl.Utf8,
            CFG.colnames["EventType"]: pl.Utf8,
            CFG.colnames["Coords"]: pl.Utf8,
            CFG.colnames["GroundTruth"]: pl.Utf8,
            "Depth": pl.Int64,
            "Class": pl.Utf8,
            "Density": pl.Utf8,
        }
    )

    def __init__(self) -> None:
        pass

    def get_component_label(self, img_path: str, compo: dict, img_density: str) -> bool:
        """
        Build the labeling interface. This labeling interface is split into two sections: the left section displays the screenshot, and the right section displays the interactable elements.
        The user will be prompted to label each interactable element with a ground truth value.

        Args:
            img_path (str): The path to the screenshot image.
            compo (dict): The interactable component to be labeled.
        """
        assert isinstance(img_path, str) and os.path.isfile(img_path), (
            "img_path must be a valid file path string"
        )
        assert isinstance(compo, dict) and "points" in compo, (
            "compo must be a dictionary containing 'points' key"
        )

        labeled_component = False

        def __submit():
            self._create_df_entry(img_name, compo, entry.get(), img_density)
            nonlocal labeled_component
            labeled_component = True
            root.destroy()

        img = Image.open(img_path)
        img_name = os.path.basename(img_path)
        compo_bbox = _bbox_from_object(compo)
        compo_crop = img.crop(compo_bbox)

        root = tk.Tk()
        root.title("Screen2SOM Labeling Interface")
        root.geometry("1920x1080")
        root.update()  # Required to get the window width

        # Screenshot
        left_master = tk.Frame(root)
        left_master.pack(side="left")

        draw = ImageDraw.Draw(img)
        draw.rectangle(compo_bbox, outline="red", width=3)

        aspect_ratio = img.width / img.height
        new_dims = (
            round(root.winfo_width() * 2 / 3),
            round(root.winfo_width() * 2 / 3 / aspect_ratio),
        )
        img = itk.PhotoImage(img.resize(new_dims))
        img_label = tk.Label(left_master, image=img)
        img_label.pack(side="left")

        # Interactable element and prompt to label
        right_master = tk.Frame(root)
        right_master.pack(side="right")

        aspect_ratio = compo_crop.width / compo_crop.height
        new_dims = (
            round(root.winfo_width() * 1 / 3),
            round(root.winfo_width() * 1 / 3 / aspect_ratio),
        )
        compo_img = itk.PhotoImage(compo_crop.resize(new_dims))
        compo_label = tk.Label(right_master, image=compo_img)
        compo_label.pack(side="top")

        prompt_label = tk.Label(
            right_master,
            text="Please label the interactable element:\n Use 'C-c' to discard.",
        )
        prompt_label.pack(side="top")

        entry = tk.Entry(right_master)
        entry.bind("<Return>", lambda _: __submit())
        entry.bind("<Control-c>", lambda _: root.destroy())

        entry.focus_force()
        entry.pack(side="top")

        # Horizontal layout for the buttons
        btn_layout = tk.Frame(right_master)
        btn_layout.pack(side="bottom")
        submit_button = tk.Button(
            btn_layout,
            text="Submit",
            command=lambda: __submit(),
        )
        submit_button.pack(side="left")

        # Option to discard
        discard_button = tk.Button(
            btn_layout, text="Discard", command=lambda: root.destroy()
        )
        discard_button.pack(side="right")

        root.mainloop()

        # return wether the user labeled the component or not
        return labeled_component

    def _create_df_entry(
        self,
        img_name: str,
        compo: dict,
        label: str,
        img_density: str,
    ) -> None:
        """
        Create a DataFrame entry for the labeled interactable element.

        Args:
            img_name (str): The name of the screenshot image.
            compo (dict): The interactable component.
            label (str): The ground truth label for the interactable component.

        Returns:
            pl.DataFrame: The updated DataFrame with the new entry.
        """
        assert isinstance(img_name, str) and img_name.endswith(
            (".png", ".jpg", ".jpeg")
        ), "img_name must be a string ending with a valid image extension"
        assert isinstance(compo, dict) and "centroid" in compo, (
            "compo must be a dictionary containing 'centroid' key"
        )
        assert isinstance(label, str), "label must be a string"

        entry = {
            CFG.colnames["Screenshot"]: img_name,
            CFG.colnames["EventType"]: "left_click",
            CFG.colnames[
                "Coords"
            ]: f"{round(compo['centroid'][0])},{round(compo['centroid'][1])}",
            CFG.colnames["GroundTruth"]: label,
            "Depth": compo["depth"],
            "Class": _CLASS_MAPPING[compo["class"]],
            "Density": img_density,
        }
        self.dataframe = pl.concat(
            [self.dataframe, pl.DataFrame(entry)], how="vertical"
        )


if __name__ == "__main__":
    """
    Main function to preprocess the dataset and prompt the user for labels.
    """
    # image_cache = ImageCache(":memory:")
    labeling_interface = LabelingInterface()

    timer_start = timer()
    soms, imgs = _preprocess_dataset()
    timer_end = timer()
    print(f"Dataset preprocessed in {timer_end - timer_start} seconds.")

    compo_per_density_counter: dict[str, Counter] = {
        density: Counter()
        for density in ["Low Density", "Medium Density", "High Density"]
    }
    for img_path, som in zip(imgs, soms):
        som_dict: dict[str, Any] = json.load(open(f"{CFG.som_dir}/{som}"))
        interactable_elements = _get_interactable_elements(som_dict)
        img_density = som_dict["DensityLevel"]

        for compo in interactable_elements:
            if (
                compo_per_density_counter[img_density][_CLASS_MAPPING[compo["class"]]]
                >= _LIMIT_PER_DENSITY
            ):
                continue

            # if image_cache.in_cache(
            #     compo_crop := Image.open(img_path).crop(_bbox_from_object(compo))
            # ):
            #     continue
            labeled = labeling_interface.get_component_label(
                img_path, compo, img_density
            )
            # image_cache.update_cache(compo_crop)

            if labeled:
                compo_per_density_counter[img_density][
                    _CLASS_MAPPING[compo["class"]]
                ] += 1

    labeling_interface.dataframe.write_csv("input/eval/eval.csv")
    print("Labels saved successfully.")
