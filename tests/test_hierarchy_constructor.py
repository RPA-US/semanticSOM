import unittest
from src.utils.hierarchy_constructor import (
    build_tree,
    ensure_toplevel,
    readjust_depth,
    labels_to_soms,
    flatten_som,
)


class TestHierarchyConstructor(unittest.TestCase):
    def setUp(self):
        self.tree = [
            {
                "id": 0,
                "points": [(0, 0), (0, 100), (100, 0), (100, 100)],
                "xpath": [0],
                "class": "Root",
                "type": "root",
                "text": "button",
                "centroid": (50, 50),
                "depth": 0,
                "children": [],
            },
            {
                "id": 1,
                "points": [(10, 10), (20, 10), (20, 20), (10, 20)],
                "xpath": [0, 1],
                "class": "Button",
                "type": "leaf",
                "text": "button",
                "centroid": (15, 15),
                "depth": 1,
                "children": [],
            },
            {
                "id": 2,
                "xpath": [0, 1, 2],
                "points": [(30, 30), (40, 30), (40, 40), (30, 40)],
                "class": "Text",
                "type": "leaf",
                "text": "button",
                "centroid": (35, 35),
                "depth": 0,  # This is to be fixed in a test
                "children": [],
            },
            {  # Application for toplevel
                "id": 3,
                "points": [(10, 10), (20, 10), (20, 20), (10, 20)],
                "xpath": [0, 1, 3],
                "class": "Application",
                "type": "leaf",
                "text": "button",
                "centroid": (15, 15),
                "depth": 3,
                "children": [],
            },
        ]
        self.root_children = [
            {
                "id": 1,
                "points": [(10, 10), (20, 10), (20, 20), (10, 20)],
                "xpath": [0, 1],
                "class": "Button",
                "type": "node",
                "text": "button",
                "centroid": (15, 15),
                "depth": 1,
                "children": [
                    {
                        "id": 2,
                        "xpath": [0, 1, 2],
                        "points": [(30, 30), (40, 30), (40, 40), (30, 40)],
                        "class": "Text",
                        "type": "leaf",
                        "text": "button",
                        "centroid": (35, 35),
                        "depth": 0,  # This is to be fixed in a test
                        "children": [],
                    },
                    {  # Application for toplevel
                        "id": 3,
                        "points": [(10, 10), (20, 10), (20, 20), (10, 20)],
                        "xpath": [0, 1, 3],
                        "class": "Application",
                        "type": "leaf",
                        "text": "button",
                        "centroid": (15, 15),
                        "depth": 3,
                        "children": [],
                    },
                ],
            },
        ]
        self.labels = {
            "shapes": [
                {
                    "label": "Button",
                    "points": [[10, 10], [20, 10], [20, 20], [10, 20]],
                    "shape_type": "polygon",
                },
                {
                    "label": "Text",
                    "points": [[30, 30], [40, 30], [40, 40], [30, 40]],
                    "shape_type": "polygon",
                },
            ],
            "imageHeight": 100,
            "imageWidth": 100,
        }

    def test_build_tree(self):
        tree = build_tree(self.tree)
        self.assertIsInstance(tree, list)
        self.assertGreater(len(tree), 0)

    def test_ensure_toplevel(self):
        tree = {"children": self.root_children, "type": "root"}
        new_children, bring_up = ensure_toplevel(tree)
        self.assertIsInstance(new_children, list)
        self.assertIsInstance(bring_up, list)
        self.assertGreater(len(new_children), 0)
        self.assertEquals(new_children[1]["class"], "Application")
        self.assertEquals(new_children[1]["depth"], 1)
        self.assertEquals(new_children[1]["xpath"], [0, 3])

    def test_readjust_depth(self):
        nodes = readjust_depth(self.tree, 2)
        self.assertIsInstance(nodes, list)
        self.assertEqual(nodes[0]["depth"], 2)

    def test_labels_to_soms(self):
        som = labels_to_soms(self.labels)
        self.assertIsInstance(som, dict)
        self.assertIn("som", som)

    def test_flatten_som(self):
        flattened = flatten_som(self.tree)
        self.assertIsInstance(flattened, list)
        self.assertGreater(len(flattened), 0)


if __name__ == "__main__":
    unittest.main()
