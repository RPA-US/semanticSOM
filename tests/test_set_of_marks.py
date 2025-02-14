import unittest
from PIL import Image, ImageFont
import numpy as np
from shapely.geometry import Polygon
from src.utils.set_of_marks import (
    add_num_marks,
    random_color,
    _compute_text_location,
    _move_text_to_free_space,
    _filter_and_sort_compos,
    get_lower_compos,
)


class TestSetOfMarks(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (100, 100), color="white")
        self.compos = [
            {
                "id": 1,
                "points": [(10, 10), (20, 10), (20, 20), (10, 20)],
                "centroid": (15, 15),
                "class": "Button",
                "xpath": [1],
            },
            {  # Button with text inside
                "id": 2,
                "points": [(30, 30), (40, 30), (40, 40), (30, 40)],
                "centroid": (35, 35),
                "class": "Button",
                "xpath": [2],
            },
            {
                "id": 3,
                "points": [(30, 30), (35, 30), (35, 40), (30, 40)],
                "centroid": (35, 35),
                "class": "Text",
                "xpath": [2, 3],
            },
        ]
        self.font = ImageFont.load_default()

    # Tests for add_num_marks
    def test_add_num_marks_positive(self):
        result_image = add_num_marks(self.image, self.compos)
        self.assertIsInstance(result_image, Image.Image)
        self.assertEqual(result_image.size, self.image.size)

    def test_add_num_marks_empty_compos(self):
        result_image = add_num_marks(self.image, [])
        self.assertIsInstance(result_image, Image.Image)
        self.assertEqual(result_image.size, self.image.size)

    def test_add_num_marks_invalid_fontsize(self):
        with self.assertRaises(ValueError):
            add_num_marks(self.image, self.compos, fontsize=-1)

    # Tests for random_color
    def test_random_color_positive(self):
        color = random_color(rgb=True, maximum=255)
        self.assertIsInstance(color, np.ndarray)
        self.assertEqual(color.shape, (3,))
        self.assertTrue((color >= 0).all() and (color <= 255).all())

    def test_random_color_invalid_maximum(self):
        with self.assertRaises(ValueError):
            random_color(rgb=True, maximum=-1)

    # Tests for _compute_text_location
    def test_compute_text_location_no_lower_compos_non_overshadowed(self):
        compo = self.compos[0]
        lower_compos = get_lower_compos(compo, self.compos)
        text_location, text_rectangle = _compute_text_location(
            compo, self.font, lower_compos, allowed_overshadowing=1
        )
        self.assertIsInstance(text_location, tuple)
        self.assertIsInstance(text_rectangle, list)
        self.assertEqual(len(text_rectangle), 2)
        self.assertAlmostEqual(text_location[0], 15, delta=1)
        self.assertAlmostEqual(text_location[1], 15, delta=1)

    def test_compute_text_location_no_lower_compos_overshadowed(self):
        compo = self.compos[0]
        lower_compos = get_lower_compos(compo, self.compos)
        text_location, text_rectangle = _compute_text_location(
            compo, self.font, lower_compos, allowed_overshadowing=0.10
        )
        self.assertIsInstance(text_location, tuple)
        self.assertIsInstance(text_rectangle, list)
        self.assertEqual(len(text_rectangle), 2)
        self.assertAlmostEqual(text_location[0], 19, delta=1)
        self.assertAlmostEqual(text_location[1], 9, delta=1)

    def test_compute_text_location_with_lower_compos_non_overshadowed(self):
        compo = self.compos[1]
        lower_compos = get_lower_compos(compo, self.compos)
        text_location, text_rectangle = _compute_text_location(
            compo, self.font, lower_compos, allowed_overshadowing=1
        )
        self.assertIsInstance(text_location, tuple)
        self.assertIsInstance(text_rectangle, list)
        self.assertEqual(len(text_rectangle), 2)
        self.assertAlmostEqual(text_location[0], 38, delta=1)
        self.assertAlmostEqual(text_location[1], 35, delta=1)

    def test_compute_text_location_with_lower_compos_overshadowed(self):
        compo = self.compos[1]
        lower_compos = get_lower_compos(compo, self.compos)
        text_location, text_rectangle = _compute_text_location(
            compo, self.font, lower_compos, allowed_overshadowing=0.1
        )
        self.assertIsInstance(text_location, tuple)
        self.assertIsInstance(text_rectangle, list)
        self.assertEqual(len(text_rectangle), 2)
        self.assertAlmostEqual(text_location[0], 41, delta=1)
        self.assertAlmostEqual(text_location[1], 30, delta=1)

    def test_compute_text_location_complex_case(self):
        compo = {
            "id": 3,
            "points": [(10, 10), (50, 10), (50, 50), (10, 50)],
            "centroid": (30, 30),
            "class": "Button",
            "xpath": [3],
        }
        lower_compos = [
            {
                "id": 4,
                "points": [(20, 20), (40, 20), (40, 40), (20, 40)],
                "centroid": (30, 30),
                "class": "Button",
                "xpath": [4],
            }
        ]
        text_location, text_rectangle = _compute_text_location(
            compo, self.font, lower_compos, allowed_overshadowing=0.10
        )
        self.assertIsInstance(text_location, tuple)
        self.assertIsInstance(text_rectangle, list)
        self.assertEqual(len(text_rectangle), 2)
        self.assertNotAlmostEqual(text_location[0], 30, delta=1)
        self.assertNotAlmostEqual(text_location[1], 30, delta=1)

    # Tests for _move_text_to_free_space
    def test_move_text_to_free_space_positive(self):
        compo_mask = Polygon(self.compos[0]["points"])
        txt_w, txt_h = 10, 10
        text_location, text_rectangle = _move_text_to_free_space(
            compo_mask, self.font, txt_w, txt_h
        )
        self.assertIsInstance(text_location, tuple)
        self.assertIsInstance(text_rectangle, list)
        self.assertEqual(len(text_rectangle), 2)
        self.assertAlmostEqual(text_location[0], 15, delta=1)
        self.assertAlmostEqual(text_location[1], 15, delta=1)

    def test_move_text_to_free_space_invalid_polygon(self):
        with self.assertRaises(AssertionError):
            _move_text_to_free_space("invalid_polygon", self.font, 10, 10)

    # Tests for get_lower_compos
    def test_get_lower_compos_positive_empty(self):
        compo = self.compos[0]
        lower_compos = get_lower_compos(compo, self.compos)
        self.assertIsInstance(lower_compos, list)
        self.assertEqual(len(lower_compos), 0)

    def test_get_lower_compos_positive_with_compos(self):
        compo = self.compos[1]
        lower_compos = get_lower_compos(compo, self.compos)
        self.assertIsInstance(lower_compos, list)
        self.assertEqual(len(lower_compos), 1)
        self.assertEqual(lower_compos[0]["id"], 3)

    def test_get_lower_compos_no_lower_compos(self):
        compo = {
            "id": 3,
            "points": [(50, 50), (60, 50), (60, 60), (50, 60)],
            "centroid": (55, 55),
            "class": "Button",
            "xpath": [3],
        }
        lower_compos = get_lower_compos(compo, self.compos)
        self.assertEqual(lower_compos, [])

    # Tests for _filter_and_sort_compos
    def test_filter_and_sort_compos_empty_list(self) -> None:
        result = _filter_and_sort_compos([])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_filter_and_sort_compos(self) -> None:
        self.compos.append(
            {
                "id": 4,
                "points": [(100, 50), (60, 50), (100, 100), (50, 100)],
                "centroid": (55, 55),
                "class": "Image",
                "xpath": [4],
            },
        )
        result = _filter_and_sort_compos(self.compos)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["id"], 4)  # Biggest compo
        self.assertEqual(result[1]["id"], 1)
        self.assertEqual(result[2]["id"], 2)


if __name__ == "__main__":
    unittest.main()
