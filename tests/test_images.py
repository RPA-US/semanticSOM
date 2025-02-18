import unittest
from PIL import Image, ImageDraw, ImageChops
from src.utils.images import im_2_b64, b64_2_img


BLUE, GREEN, RED = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
WHITE = (255, 255, 255)


def gradient_color(minval, maxval, val, color_palette):
    """Computes intermediate RGB color of a value in the range of minval
    to maxval (inclusive) based on a color_palette representing the range.
    """
    max_index = len(color_palette) - 1
    delta = maxval - minval
    if delta == 0:
        delta = 1
    v = float(val - minval) / delta * max_index
    i1, i2 = int(v), min(int(v) + 1, max_index)
    (r1, g1, b1), (r2, g2, b2) = color_palette[i1], color_palette[i2]
    f = v - i1
    return int(r1 + f * (r2 - r1)), int(g1 + f * (g2 - g1)), int(b1 + f * (b2 - b1))


def vert_gradient(draw, height, width, color_func, color_palette):
    minval, maxval = 1, len(color_palette)
    delta = maxval - minval
    height = float(height)  # Cache.
    for y in range(0, int(height) + 1):
        f = (y - 0) / height
        val = minval + f * delta
        color = color_func(minval, maxval, val, color_palette)
        draw.line([(0, y), (100, y)], fill=color)


class TestImages(unittest.TestCase):
    def setUp(self):
        color_palette = [BLUE, GREEN, RED]
        self.image = image = Image.new("RGB", (100, 100), WHITE)
        draw = ImageDraw.Draw(image)
        vert_gradient(draw, 100, 100, gradient_color, color_palette)

        self.b64_image: str = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCABkAGQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDyTyPajyPatP7P7UfZ/av1X+0fM8H25meR7UeR7Vp/Z/aj7P7Uf2j5h7czPI9qPI9q0/s/tR9n9qP7R8w9uZnke1Hke1af2f2o+z+1H9o+Ye3MzyPajyPatP7P7UfZ/aj+0fMPbmZ5HtR5HtWn9n9qPs/tR/aPmHtzM8j2orT+z+1FH9o+Ye3NTyKPIrT8j2o8j2r4P+0fM8f25meRR5Fafke1Hke1H9o+Ye3MzyKPIrT8j2o8j2o/tHzD25meRR5Fafke1Hke1H9o+Ye3MzyKPIrT8j2o8j2o/tHzD25meRR5Fafke1Hke1H9o+Ye3MzyKK0/I9qKP7R8w9uafke1Hke1anke1Hke1fBf2j5nje3MvyPajyPatTyPajyPaj+0fMPbmX5HtR5HtWp5HtR5HtR/aPmHtzL8j2o8j2rU8j2o8j2o/tHzD25l+R7UeR7VqeR7UeR7Uf2j5h7cy/I9qPI9q1PI9qPI9qP7R8w9uZfke1Fanke1FH9o+Ye3Lm5fQUbl9BVTzKPMr53lZ0f2ay3uX0FG5fQVU8yjzKOVh/ZrLe5fQUbl9BVTzKPMo5WH9mst7l9BRuX0FVPMo8yjlYf2ay3uX0FG5fQVU8yjzKOVh/ZrLe5fQUbl9BVTzKPMo5WH9mst7l9BRVTzKKOVh/ZrKfm0ebVTzKPMrr5D9C/szyLfm0ebVTzKPMo5A/szyLfm0ebVTzKPMo5A/szyLfm0ebVTzKPMo5A/szyLfm0ebVTzKPMo5A/szyLfm0ebVTzKPMo5A/szyLfm0VU8yijkD+zPIp+bR5tU/No82uzkP0L+zPIuebR5tU/No82jkD+zPIuebR5tU/No82jkD+zPIuebR5tU/No82jkD+zPIuebR5tU/No82jkD+zPIuebR5tU/No82jkD+zPIuebRVPzaKOQP7M8iruNG40UV0H1tkG40bjRRQFkG40bjRRQFkG40bjRRQFkG40bjRRQFkG40bjRRQFkG40UUUBZH//2Q=="

    def test_im_2_b64_returns_string(self):
        b64_str = im_2_b64(self.image)
        self.assertIsInstance(b64_str, str)
        self.assertTrue(len(b64_str) > 0)
        self.assertEqual(b64_str, self.b64_image)

    def test_b64_2_img_returns_image(self):
        image_converted = b64_2_img(self.b64_image)
        self.assertIsInstance(image_converted, Image.Image)
        self.assertEqual(image_converted.size, self.image.size)
        self.assertEqual(image_converted.mode, self.image.mode)
        self.assertAlmostEqual(
            sum(
                sum(p)
                for p in ImageChops.difference(image_converted, self.image).getdata()
            ),
            0,
            delta=self.image.size[0]
            * self.image.size[1]
            * 3
            * 1.2,  # Account for compression
        )

    def test_b64_2_img_invalid_data(self):
        with self.assertRaises(Exception):
            b64_2_img("not_a_valid_base64_string")


if __name__ == "__main__":
    unittest.main()
