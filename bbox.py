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

