from __future__ import division, print_function, unicode_literals

import numpy as np
from mean_shift_utils import euclid_distance

MIN_DISTANCE = 0.1


class mean_shift(object):
    def __init__(self, kernel):
        self.kernel = kernel

    def _shift_points(self, point, points, kernel_bandwidth):
        points = np.array(points)

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
        # If we find final convercy value for each pixel point. That should
        # take a lot of time on it. But the final result will be much better.
        # So, please choose the steps by your perference.
        '''
        while(max_min_dist > MIN_DISTANCE):
            max_min_dist = 0
            iteration_number += 1
            for i in range(len(shift_points)):
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
            print(max_min_dist)
            print(max_min_dist > MIN_DISTANCE)
        '''
        # It will take a lot of time if insisting the mimum distance. So use
        # fixed number of iteration.
        for i in range(len(shift_points)):
            p_new = shift_points[i]
            for j in range(20):
                p_new = self._shift_points(p_new, points, kernel_bandwidth)
            shift_points[i] = p_new
            '''
            if i % 100 == 0:
                print('Filtering {}th pixel has been done!'.format(i))
            '''
            
        return shift_points
