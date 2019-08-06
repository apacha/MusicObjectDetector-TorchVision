import unittest

import numpy
from PIL import Image


def decode_mask_rle(mask_string, shape):
    """Decodes the mask array from the RLE-encoded form
    to the 2D numpy array.
    """
    if mask_string == 'None':
        return None

    values = []
    for kv in mask_string.split(' '):
        k_string, v_string = kv.split(':')
        k, v = int(k_string), int(v_string)
        vs = [k for _ in range(v)]
        values.extend(vs)

    mask = numpy.array(values).reshape(shape)
    return mask


class NodeTest(unittest.TestCase):
    def test_mask_decoding(self):
        test_mask = "0:15 1:10 0:14 1:16 0:11 1:20 0:7 1:22 0:7 1:22 0:6 1:23 0:5 1:25 0:3 1:26 0:3 1:26 0:3 1:24 0:5 1:24 0:4 1:24 0:5 1:23 0:5 1:22 0:7 1:20 0:9 1:19 0:9 1:17 0:13 1:14 0:16 1:10 0:19 1:5 0:22"
        mask = decode_mask_rle(test_mask, (20, 29))
        inverted_mask = (mask * -1 + 1) * 255
        im = Image.fromarray(inverted_mask)
        im = im.convert(mode="L")
        im.save("mask.png")


if __name__ == '__main__':
    unittest.main()
