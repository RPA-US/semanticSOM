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


class TestPromptProcessing(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (100, 100), color="white")
        self.som = {
            "compos": [
                {
                    "id": 0,
                    "points": [(0, 0), (0, 100), (100, 0), (100, 100)],
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

    def test_process_image_for_prompt(self):
        processed_image, sys_prompt, prompt = process_image_for_prompt(
            self.image, self.som, self.coords
        )
        self.assertIsInstance(processed_image, Image.Image)
        self.assertIsInstance(sys_prompt, str)
        self.assertIsInstance(prompt, str)

    def test_highlight_compo(self):
        compo = {"points": [(10, 10), (20, 10), (20, 20), (10, 20)]}
        highlighted_image = highlight_compo(self.image, compo)
        self.assertIsInstance(highlighted_image, Image.Image)


if __name__ == "__main__":
    unittest.main()
