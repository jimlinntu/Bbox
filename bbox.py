import numpy as np
import cv2
import copy
import math

class Bbox():
    def __init__(self, np_bbox, apply_convex_hull=False):
        '''
            np_bbox: should be in clockwise order (Ex. from cv2.boxPoints)
        '''
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
        return self.np_bbox.shape[0] != 4 or (not self.check_clockwise())

    def check_clockwise(self):
        '''
            Ref:
                https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
                https://www.element84.com/blog/determining-the-winding-of-a-polygon-given-as-a-set-of-ordered-points
        '''
        sum_ = 0
        for i in range(4):
            j = (i + 1) % 4
            x_i, y_i = self.np_bbox[i]
            x_j, y_j = self.np_bbox[j]
            # Compute the signed trapezoid region of the line segment (x_i, y_i) to (x_j, y_i)
            sum_ += (x_j - x_i) * (y_j + y_i)
        return sum_ < 0

    def get_reordered_bbox(self, distance_type="manhattan", ratio=0.3):
        '''
            [0]--------------[1]
             |                |
             |                |
            [3]--------------[2]
        '''
        distance = None
        if distance_type == "manhattan":
            distance = lambda point: point.sum()

        distances = [distance(point) for point in self.np_bbox]
        # if there exist two points with almost the same distance
        # we will consider the axis aligned width and height
        startidx = np.argmin(distances)
        for i, dist in enumerate(distances):
            if i != startidx and math.isclose(distances[startidx], dist):
                idx_with_smaller_y = min([startidx, i], key=lambda index: self.np_bbox[index][1])
                idx_with_larger_y = max([startidx, i], key=lambda index: self.np_bbox[index][1])

                # Compute the lengths (w, h) of this box use idx_with_smaller_y as a base point
                width = np.linalg.norm(
                            self.np_bbox[idx_with_smaller_y] - self.np_bbox[(idx_with_smaller_y + 1) % 4])
                height = np.linalg.norm(
                            self.np_bbox[idx_with_smaller_y] - self.np_bbox[(idx_with_smaller_y - 1 + 4) % 4])

                if width > height * (1 + ratio):
                    startidx = idx_with_smaller_y # favor the smaller y
                else:
                    startidx = idx_with_larger_y
                break

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

if __name__ == "__main__":
    np_bbox = np.array([[98, 99], [3, 5], [5, 3], [100, 97]])
    bbox = Bbox(np_bbox)
    assert np.allclose(bbox.get_reordered_bbox(),
            np.array([[5, 3], [100, 97], [98, 99], [3, 5]]))
    np_bbox = np.array([[5, 100], [3, 97], [98, 2], [100, 5]])
    bbox = Bbox(np_bbox)
    assert np.allclose(bbox.get_reordered_bbox(),
            np.array([[3, 97], [98, 2], [100, 5], [5, 100]]))

    assert bbox.check_clockwise()
    assert not bbox.error()
