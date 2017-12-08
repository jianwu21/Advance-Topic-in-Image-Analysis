from __future__ import division, print_function, unicode_literals

import numpy as np
from mean_shift_utils import euclid_distance

MIN_DISTANCE = 0.1


class mean_shift(object):
    def __init__(self, kernel):
        self.kernel = kernel

    def neighbourhood_points(self, X, x_centroid, distance):
        eligible_X = []
        for x in X:
            distance_between = euclid_distance(x, x_centroid)
            if distance_between <= distance:
                eligible_X.append(x)
        return eligible_X

    def _shift_points(self, point, points, kernel_bandwidth):
        points = np.array(self.neighbourhood_points(points, point, 5))

        point_weights = self.kernel(point-points, kernel_bandwidth)
        tiled_weights = np.tile(point_weights, [len(point), 1])
        denominator = sum(point_weights)

        shifted_point = np.multiply(
            tiled_weights.transpose(), points).sum(axis=0) / denominator

        return np.array(shifted_point)

    def cluster(self, points, kernel_bandwidth):
        points = np.array(points)
        shift_points = np.array(points)
        max_min_dist = 1
        iteration_number = 0

        still_shifting = [True] * points.shape[0]
        while max_min_dist > MIN_DISTANCE:
            max_min_dist = 0
            iteration_number += 1
            for i in range(0, len(shift_points)):
                if not still_shifting[i]:
                    continue
                p_new = shift_points[i]
                p_new_start = p_new
                p_new = self._shift_points(p_new, points, kernel_bandwidth)
                dist = euclid_distance(p_new, p_new_start)
                if dist > max_min_dist:
                    max_min_dist = dist
                if dist < MIN_DISTANCE:
                    still_shifting[i] = False
                shift_points[i] = p_new

        return shift_points
