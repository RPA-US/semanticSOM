import unittest
from PIL import Image
from src.utils.prompt_processing import (
    Coords,
    hash_image,
    identify_target_object,
    bbox_from_object,
    process_image_for_prompt,
    highlight_compo,
)
from unittest.mock import patch
from src.cfg import CFG
from src.semantics.prompts import (
    COT_ACTION_TARGET_BASE,
    COT_ACTION_TARGET_COORDS,
    COT_ACTION_TARGET_ELEM,
)


class MockCFG(CFG):
    """
    Configuration class for the project.
    """

    sqlite_db_location: str = ":memory:"

    highlight_config: dict[str, list[str] | str] = {
        "technique": ["highlight"],  # possible values: "som", "highlight"
        "crop": "none",  # possible values: "parent", "target", "none"
    }
    highlight_config_parent: dict[str, list[str] | str] = {
        "technique": ["highlight"],  # possible values: "som", "highlight"
        "crop": "parent",  # possible values: "parent", "target", "none"
    }
    highlight_config_target: dict[str, list[str] | str] = {
        "technique": ["highlight"],  # possible values: "som", "highlight"
        "crop": "target",  # possible values: "parent", "target", "none"
    }

    som_config: dict[str, list[str] | str] = {
        "technique": ["som"],  # possible values: "som", "highlight"
        "crop": "none",  # possible values: "parent", "target", "none"
    }
    som_config_parent: dict[str, list[str] | str] = {
        "technique": ["som"],  # possible values: "som", "highlight"
        "crop": "parent",  # possible values: "parent", "target", "none"
    }
    som_config_target: dict[str, list[str] | str] = {
        "technique": ["som"],  # possible values: "som", "highlight"
        "crop": "target",  # possible values: "parent", "target", "none"
    }

    both_config: dict[str, list[str] | str] = {
        "technique": ["som", "highlight"],  # possible values: "som", "highlight"
        "crop": "none",  # possible values: "parent", "target", "none"
    }
    both_config_parent: dict[str, list[str] | str] = {
        "technique": ["som", "highlight"],  # possible values: "som", "highlight"
        "crop": "parent",  # possible values: "parent", "target", "none"
    }
    both_config_target: dict[str, list[str] | str] = {
        "technique": ["som", "highlight"],  # possible values: "som", "highlight"
        "crop": "target",  # possible values: "parent", "target", "none"
    }

    coords_config: dict[str, list[str] | str] = {
        "technique": [],  # possible values: "som", "highlight"
        "crop": "none",  # possible values: "parent", "target", "none"
    }
    coords_config_parent: dict[str, list[str] | str] = {
        "technique": [],  # possible values: "som", "highlight"
        "crop": "parent",  # possible values: "parent", "target", "none"
    }
    coords_config_target: dict[str, list[str] | str] = {
        "technique": [],  # possible values: "som", "highlight"
        "crop": "target",  # possible values: "parent", "target", "none"
    }


class TestPromptProcessing(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (200, 200), color="white")
        self.som = {
            "compos": [
                {
                    "id": 0,
                    "points": [(0, 0), (100, 0), (100, 100), (0, 100)],
                    "xpath": [0],
                    "class": "Root",
                    "type": "root",
                    "centroid": (50, 50),
                },
                {
                    "id": 1,
                    "points": [(10, 10), (20, 10), (20, 20), (10, 20)],
                    "xpath": [0, 1],
                    "class": "Button",
                    "type": "leaf",
                    "centroid": (15, 15),
                },
                {
                    "id": 2,
                    "xpath": [0, 1, 2],
                    "points": [(30, 30), (40, 30), (40, 40), (30, 40)],
                    "class": "Text",
                    "type": "leaf",
                    "centroid": (35, 35),
                },
            ]
        }
        self.coords = Coords(x=15, y=15)

    def test_hash_image(self):
        # Placeholder test as the function is not implemented
        self.assertIsNone(hash_image(self.image))

    def test_identify_target_object(self):
        target_object = identify_target_object(self.image, self.som, self.coords)
        self.assertIsInstance(target_object, dict)
        self.assertEqual(target_object["id"], 1)

    def test_bbox_from_object(self):
        obj = {"points": [(10, 10), (20, 10), (20, 20), (10, 20)]}
        bbox = bbox_from_object(obj)
        self.assertEqual(bbox, (10, 10, 20, 20))

    @patch("src.cfg.CFG.prompt_config", new=MockCFG.highlight_config)
    def test_process_image_for_prompt_highligth_nocrop(self):
        processed_image, sys_prompt, prompt = process_image_for_prompt(
            self.image, self.som, self.coords
        )
        self.assertIsInstance(processed_image, Image.Image)
        self.assertEqual(processed_image.size, self.image.size)
        self.assertIsInstance(sys_prompt, str)
        self.assertEqual(sys_prompt, COT_ACTION_TARGET_BASE)
        self.assertIsInstance(prompt, str)
        self.assertIn("highlight", prompt)

    @patch("src.cfg.CFG.prompt_config", new=MockCFG.highlight_config_parent)
    def test_process_image_for_prompt_highligth_parent(self):
        processed_image, sys_prompt, prompt = process_image_for_prompt(
            self.image, self.som, self.coords
        )
        self.assertIsInstance(processed_image, Image.Image)
        self.assertEqual(processed_image.size, (100, 100))
        self.assertIsInstance(sys_prompt, str)
        self.assertEqual(sys_prompt, COT_ACTION_TARGET_BASE)
        self.assertIsInstance(prompt, str)
        self.assertIn("highlight", prompt)

    @patch("src.cfg.CFG.prompt_config", new=MockCFG.highlight_config_target)
    def test_process_image_for_prompt_highligth_target(self):
        processed_image, sys_prompt, prompt = process_image_for_prompt(
            self.image, self.som, self.coords
        )
        self.assertIsInstance(processed_image, Image.Image)
        self.assertEqual(processed_image.size, (10, 10))
        self.assertIsInstance(sys_prompt, str)
        self.assertEqual(sys_prompt, COT_ACTION_TARGET_ELEM)
        self.assertIsInstance(prompt, str)
        self.assertEqual(prompt, "")

    @patch("src.cfg.CFG.prompt_config", new=MockCFG.som_config)
    def test_process_image_for_prompt_som_nocrop(self):
        processed_image, sys_prompt, prompt = process_image_for_prompt(
            self.image, self.som, self.coords
        )
        self.assertIsInstance(processed_image, Image.Image)
        self.assertEqual(processed_image.size, self.image.size)
        self.assertIsInstance(sys_prompt, str)
        self.assertEqual(sys_prompt, COT_ACTION_TARGET_BASE)
        self.assertIsInstance(prompt, str)
        self.assertIn("number", prompt)

    @patch("src.cfg.CFG.prompt_config", new=MockCFG.som_config_parent)
    def test_process_image_for_prompt_som_parent(self):
        processed_image, sys_prompt, prompt = process_image_for_prompt(
            self.image, self.som, self.coords
        )
        self.assertIsInstance(processed_image, Image.Image)
        self.assertEqual(processed_image.size, (100, 100))
        self.assertIsInstance(sys_prompt, str)
        self.assertEqual(sys_prompt, COT_ACTION_TARGET_BASE)
        self.assertIsInstance(prompt, str)
        self.assertIn("number", prompt)

    @patch("src.cfg.CFG.prompt_config", new=MockCFG.som_config_target)
    def test_process_image_for_prompt_som_target(self):
        processed_image, sys_prompt, prompt = process_image_for_prompt(
            self.image, self.som, self.coords
        )
        self.assertIsInstance(processed_image, Image.Image)
        self.assertEqual(processed_image.size, (10, 10))
        self.assertIsInstance(sys_prompt, str)
        self.assertEqual(sys_prompt, COT_ACTION_TARGET_ELEM)
        self.assertIsInstance(prompt, str)
        self.assertEqual(prompt, "")

    @patch("src.cfg.CFG.prompt_config", new=MockCFG.both_config)
    def test_process_image_for_prompt_both_nocrop(self):
        processed_image, sys_prompt, prompt = process_image_for_prompt(
            self.image, self.som, self.coords
        )
        self.assertIsInstance(processed_image, Image.Image)
        self.assertEqual(processed_image.size, self.image.size)
        self.assertIsInstance(sys_prompt, str)
        self.assertEqual(sys_prompt, COT_ACTION_TARGET_BASE)
        self.assertIsInstance(prompt, str)
        self.assertIn("highlight", prompt)
        self.assertIn("number", prompt)

    @patch("src.cfg.CFG.prompt_config", new=MockCFG.both_config_parent)
    def test_process_image_for_prompt_both_parent(self):
        processed_image, sys_prompt, prompt = process_image_for_prompt(
            self.image, self.som, self.coords
        )
        self.assertIsInstance(processed_image, Image.Image)
        self.assertEqual(processed_image.size, (100, 100))
        self.assertIsInstance(sys_prompt, str)
        self.assertEqual(sys_prompt, COT_ACTION_TARGET_BASE)
        self.assertIsInstance(prompt, str)
        self.assertIn("highlight", prompt)
        self.assertIn("number", prompt)

    @patch("src.cfg.CFG.prompt_config", new=MockCFG.both_config_target)
    def test_process_image_for_prompt_both_target(self):
        processed_image, sys_prompt, prompt = process_image_for_prompt(
            self.image, self.som, self.coords
        )
        self.assertIsInstance(processed_image, Image.Image)
        self.assertEqual(processed_image.size, (10, 10))
        self.assertIsInstance(sys_prompt, str)
        self.assertEqual(sys_prompt, COT_ACTION_TARGET_ELEM)
        self.assertIsInstance(prompt, str)
        self.assertEqual(prompt, "")

    @patch("src.cfg.CFG.prompt_config", new=MockCFG.coords_config)
    def test_process_image_for_prompt_coords_nocrop(self):
        processed_image, sys_prompt, prompt = process_image_for_prompt(
            self.image, self.som, self.coords
        )
        self.assertIsInstance(processed_image, Image.Image)
        self.assertEqual(processed_image.size, self.image.size)
        self.assertIsInstance(sys_prompt, str)
        self.assertEqual(sys_prompt, COT_ACTION_TARGET_COORDS)
        self.assertIsInstance(prompt, str)
        self.assertIn("coordinates", prompt)

    @patch("src.cfg.CFG.prompt_config", new=MockCFG.coords_config_parent)
    def test_process_image_for_prompt_coords_parent(self):
        processed_image, sys_prompt, prompt = process_image_for_prompt(
            self.image, self.som, self.coords
        )
        self.assertIsInstance(processed_image, Image.Image)
        self.assertEqual(processed_image.size, (100, 100))
        self.assertIsInstance(sys_prompt, str)
        self.assertEqual(sys_prompt, COT_ACTION_TARGET_COORDS)
        self.assertIsInstance(prompt, str)
        self.assertIn("coordinates", prompt)

    @patch("src.cfg.CFG.prompt_config", new=MockCFG.coords_config_target)
    def test_process_image_for_prompt_coords_target(self):
        processed_image, sys_prompt, prompt = process_image_for_prompt(
            self.image, self.som, self.coords
        )
        self.assertIsInstance(processed_image, Image.Image)
        self.assertEqual(processed_image.size, (10, 10))
        self.assertIsInstance(sys_prompt, str)
        self.assertEqual(sys_prompt, COT_ACTION_TARGET_ELEM)
        self.assertIsInstance(prompt, str)
        self.assertEqual(prompt, "")

    def test_highlight_compo(self):
        compo = {"points": [(10, 10), (20, 10), (20, 20), (10, 20)]}
        highlighted_image = highlight_compo(self.image, compo)
        self.assertIsInstance(highlighted_image, Image.Image)


if __name__ == "__main__":
    unittest.main()
