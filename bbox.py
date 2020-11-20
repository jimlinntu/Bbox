import numpy as np
import cv2
import copy

class Bbox():
    def __init__(self, np_bbox, apply_convex_hull=False):
        assert isinstance(np_bbox, np.ndarray)
        assert np_bbox.shape == (4, 2)
        assert isinstance(apply_convex_hull, bool)
        self.np_bbox = np_bbox
        self.apply_convex_hull = apply_convex_hull

        if self.apply_convex_hull:
            # NOTE: this exploit's cv2.convexHull clockwise order of its return points
            hull = cv2.convexHull(np_bbox)
            assert hull.shape[1:3] == (1, 2)
            hull = np.squeeze(hull, axis=1)
            self.np_bbox = hull

    def error(self):
        return self.np_bbox.shape[0] != 4

    def get_reordered_bbox(self, distance_type="manhattan"):
        '''
            [0]--------------[1]
             |                |
             |                |
            [3]--------------[2]
        '''
        distance = None
        if distance_type == "manhattan":
            distance = lambda point: point.sum()

        startidx = np.argmin([distance(point) for point in self.np_bbox])
        bbox = copy.deepcopy(self.np_bbox)
        bbox = np.roll(bbox, 4 - startidx, 0)
        return bbox

    def crop_image(self, img):
        '''
            Case 1: Normal case
                           w
                [0]------------------[1]
                 |                    |
                 |   1  2  3  4  5    |  h
                 |                    |
                [3]------------------[2]

            Case 2:
                     w
                [0]-----[1]
                 |   5   |
                 |   4   |
                 |   3   | h
                 |   2   |
                 |   1   |
                [3]-----[2]
        '''
        bbox = self.get_reordered_bbox()
        w, h = int(np.linalg.norm(bbox[0] - bbox[1])), int(np.linalg.norm(bbox[0] - bbox[3]))

        if h < w * 1.5:
            # Case 1
            dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(bbox.astype(np.float32), dst)
            crop_width, crop_height = w, h
        else:
            # Case 2
            dst = np.array([[h, 0], [h, w], [0, w], [0, 0]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(bbox.astype(np.float32), dst)
            crop_width, crop_height = h, w

        cropped = cv2.warpPerspective(img, M, (crop_width, crop_height))
        return cropped

