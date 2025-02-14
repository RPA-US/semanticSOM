from shapely.geometry import Polygon
import json
import copy


def build_tree(tree: list, depth=1, text_class="Text"):
    """
    Recursively constructs a tree hierarchy from a list of compos.

    Args:
        tree (list): A list of compos.
        depth (int): The current depth of the tree.
        text_class (str): The class name for text components.

    Returns:
        list: A tree representing the hierarchy of the compos.
    """
    assert isinstance(tree, list), "tree must be a list"
    assert all(isinstance(compo, dict) for compo in tree), "all elements in tree must be dictionaries"
    for compo1 in tree:
        if compo1["depth"] != depth:
            continue
        compo1["children"] = []
        for compo2 in tree:
            if compo2["depth"] != depth or compo1 == compo2:
                continue
            polygon1 = Polygon(compo1["points"])
            polygon2 = Polygon(compo2["points"])
            if not polygon1.is_valid or not polygon2.is_valid:
                continue
            intersection = polygon1.intersection(polygon2).area
            try:
                if intersection / polygon2.area > 0.5:
                    compo2["depth"] = depth + 1
                    compo1["children"].append(compo2)
                    compo1["type"] = "node"
                    compo2["xpath"].append(compo1["id"])
                    if compo2["class"] == text_class:
                        compo1["text"] += compo2["text"] + " | "
            except ZeroDivisionError:
                continue
        if len(compo1["children"]) > 0:
            compo1["children"] = build_tree(compo1["children"], depth=depth + 1)
    return list(filter(lambda s: s["depth"] == depth, tree))


def ensure_toplevel(tree: dict, bring_up=None):
    """
    Ensures that the TopLevel labels are in the top level of the tree.

    Args:
        tree (dict): A tree representing the hierarchy of the compos.
        bring_up (list): A list of components to bring up to the top level.

    Returns:
        tuple: A tuple containing the new children and the bring_up list.
    """
    assert isinstance(tree, dict), "tree must be a dictionary"
    assert "children" in tree, "tree must contain 'children' key"
    if bring_up is None:
        bring_up = []
    children = tree["children"]
    for child in children:
        if child["type"] == "node":
            child["children"], bring_up = ensure_toplevel(child, bring_up=bring_up)
            if len(child["children"]) == 0:
                child["type"] = "leaf"
            if child["class"] in ["Application", "Taskbar", "Dock"]:
                bring_up.append(child)
                # remove 1st elements from list(stack), if any
                if len(child["xpath"]) > 0:
                    child["xpath"].pop(0)

    new_children = list(filter(lambda c: c not in bring_up, children))

    if tree["type"] == "root":
        new_children.extend(bring_up)
        new_children = readjust_depth(new_children, 1)

    assert isinstance(new_children, list), "new_children must be a list"
    assert isinstance(bring_up, list), "bring_up must be a list"
    return new_children, bring_up


def readjust_depth(nodes, depth):
    """
    Readjusts the depth of the nodes in the tree.

    Args:
        nodes (list): A list of nodes.
        depth (int): The new depth.

    Returns:
        list: The nodes with adjusted depth.
    """
    assert isinstance(nodes, list), "nodes must be a list"
    assert all(isinstance(node, dict) for node in nodes), "all elements in nodes must be dictionaries"
    for node in nodes:
        # Remove xpath elements no longer needed
        node["xpath"] = node["xpath"][node["depth"] - depth :]
        # Readjust depth
        node["depth"] = depth
        node["children"] = readjust_depth(node["children"], depth + 1)

    assert isinstance(nodes, list), "nodes must be a list"
    return nodes


def labels_to_soms(labels: dict):
    """
    Converts a list of labeled jsons into a list of SOMs.

    Args:
        labels (dict): A dictionary containing labeled jsons.

    Returns:
        dict: Pairs of image name and SOM.
    """
    assert isinstance(labels, dict), "labels must be a dictionary"
    assert "shapes" in labels, "labels must contain 'shapes' key"
    compos = labels["shapes"]
    labels["compos"] = labels.pop("shapes")
    for id, compo in enumerate(compos):
        compo["depth"] = 1
        compo["type"] = "leaf"
        compo["xpath"] = []
        compo["id"] = id
        compo["class"] = compo.pop("label")
        if compo["shape_type"] == "rectangle" and len(compo["points"]) == 2:
            x, y, x2, y2 = [coor for coords in compo["points"] for coor in coords]
            compo["points"] = [[x, y], [x2, y], [x2, y2], [x, y2]]
            compo["shape_type"] = "polygon"
        for point in compo["points"]:
            point[0] = max(0, point[0])
            point[1] = max(0, point[1])
        compo["centroid"] = list(Polygon(compo["points"]).centroid.coords[0])

    compos.sort(key=lambda x: Polygon(x["points"]).area, reverse=True)

    som = {
        "depth": 0,
        "type": "root",
        "id": 0,
        "points": [
            [0, 0],
            [0, labels["imageHeight"]],
            [labels["imageWidth"], 0],
            [labels["imageWidth"], labels["imageHeight"]],
        ],
        "centroid": list(
            Polygon(
                [
                    [0, 0],
                    [0, labels["imageHeight"]],
                    [labels["imageWidth"], 0],
                    [labels["imageWidth"], labels["imageHeight"]],
                ]
            ).centroid.coords[0]
        ),
        "children": build_tree(copy.deepcopy(compos)),
    }

    som["children"], _ = ensure_toplevel(som)

    # Copy XPath of som compos to compos
    for compo in compos:
        for node in flatten_som(som["children"]):
            if compo["id"] == node["id"]:
                node["xpath"].append(node["id"])
                compo["xpath"] = node["xpath"]
                compo["depth"] = node["depth"]
                compo["type"] = node["type"]
                break

    labels["som"] = som

    assert isinstance(labels, dict), "labels must be a dictionary"
    return labels


def flatten_som(tree):
    """
    Flatten the SOM tree into a list of nodes.

    Args:
        tree (list): A tree representing the hierarchy of the SOM.

    Returns:
        list: A flattened list of nodes.
    """
    assert isinstance(tree, list), "tree must be a list"
    assert all(isinstance(node, dict) for node in tree), "all elements in tree must be dictionaries"
    flattened = []
    for node in tree:
        flattened.append(node)
        if node["type"] == "node":
            flattened.extend(flatten_som(node["children"]))
    assert isinstance(flattened, list), "flattened must be a list"
    return flattened


if __name__ == "__main__":
    lables = json.load(open("input/soms/Captura de pantalla (147).json"))
    som = labels_to_soms(lables)
    json.dump(som, open("input/soms/Captura de pantalla (147)_som.json", "w"))
