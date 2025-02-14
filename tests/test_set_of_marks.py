import unittest
from PIL import Image
import numpy as np
from src.utils.set_of_marks import (
    add_num_marks,
    random_color,
)


class MockSelf:
    def draw_text(self, text, position, color):
        pass


class TestSetOfMarks(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (100, 100), color="white")
        self.compos = [
            {
                "id": 1,
                "points": [(10, 10), (20, 10), (20, 20), (10, 20)],
                "centroid": (15, 15),
            },
            {
                "id": 2,
                "points": [(30, 30), (40, 30), (40, 40), (30, 40)],
                "centroid": (35, 35),
            },
        ]
        self.mock_self = MockSelf()
        self.binary_mask = np.zeros((100, 100), dtype=np.uint8)
        self.binary_mask[30:70, 30:70] = 1
        self.text = "1"
        self.color = (255, 0, 0)

    def test_add_num_marks(self):
        result_image = add_num_marks(self.image, self.compos)
        self.assertIsInstance(result_image, Image.Image)
        self.assertEqual(result_image.size, self.image.size)

    def test_random_color(self):
        color = random_color(rgb=True, maximum=255)
        self.assertIsInstance(color, np.ndarray)
        self.assertEqual(color.shape, (3,))
        self.assertTrue((color >= 0).all() and (color <= 255).all())


if __name__ == "__main__":
    unittest.main()
