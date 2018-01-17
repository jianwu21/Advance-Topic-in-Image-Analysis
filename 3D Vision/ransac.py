from __future__ import absolute_import, division, print_function

from pylab import *


def ransac(model, x, y, nsamples, threshold, maxiter = 1e2, desiredprob = 0.99,
           debug = False):
    iterations = 0
    ndata = x.shape[1]
    best_inlier_idxs = None
    best_ninliers = 0
    goodprob = 0
    bestfit = None
    while goodprob < desiredprob and iterations < maxiter:
        all_idxs = arange(ndata)
        shuffle(all_idxs)
        maybe_idxs = all_idxs[:nsamples]
        test_idxs = all_idxs[nsamples:]
        x_maybeinliers = x[:, maybe_idxs]
        y_maybeinliers = y[:, maybe_idxs]
        x_test_points = x[:, test_idxs]
        y_test_points = y[:, test_idxs]

        maybemodel = model.fit(x_maybeinliers, y_maybeinliers)
        test_err = model.get_error(maybemodel, x_test_points, y_test_points)
        # select indices of rows with accepted points
        also_idxs = test_idxs[test_err < threshold]
        x_alsoinliers = x[:, also_idxs]
        y_alsoinliers = y[:, also_idxs]

        ninliers = also_idxs.shape[0]
        if ninliers > best_ninliers:
            print(x_maybeinliers.shape, x_alsoinliers.shape)
            betterx = concatenate((x_maybeinliers, x_alsoinliers), axis=1)
            bettery = concatenate((y_maybeinliers, y_alsoinliers), axis=1)
            bestfit = model.fit(betterx, bettery)
            best_inlier_idxs = concatenate((maybe_idxs, also_idxs))
            best_ninliers = ninliers

        iterations += 1
        goodprob = 1 - (1 - (float(best_ninliers) / ndata)**nsamples)**iterations
        if debug:
            print('test_err.min()',test_err.min())
            print('test_err.max()',test_err.max())
            print('mean(test_err)',mean(test_err))
            print(
                'iteration %d: inliers = %d, goodprob = %f'
                % (iterations, ninliers, goodprob)
            )

    return bestfit, best_inlier_idxs
