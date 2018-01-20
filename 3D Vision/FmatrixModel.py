import numpy as np


class FundamentalMatrixModel(object):
    def fit(self, p1, p2):
        n = p1.shape[1]
        if p2.shape[1] != n:
            raise ValueError('Number of points don\'t match.')

        # build matrix for equations
        A = np.zeros((n,9))
        for i in range(n):
            A[i] = [
                p1[0,i]*p2[0,i], p1[0,i]*p2[1,i], p1[0,i]*p2[2,i],
                p1[1,i]*p2[0,i], p1[1,i]*p2[1,i], p1[1,i]*p2[2,i],
                p1[2,i]*p2[0,i], p1[2,i]*p2[1,i], p1[2,i]*p2[2,i],
            ]

        # compute linear least square solution
        # constrain F
        # make rank 2 by zeroing out last singular value
        return reduce_rank(kernel(A).reshape(3, 3))

    def get_error(self, F, p1, p2):
        # Sampson distance (first-order approximation to geometric error)
        p2_fit = np.dot(F, p1)
        p1_fit = np.dot(F.T, p2)
        p2_f_p1 = np.sum(np.dot(p1_fit.T, p1), axis = 1)

        return np.sqrt(
            p2_f_p1**2 /\
            (p2_fit[0, :]**2 + p2_fit[1, :]**2 + \
             p1_fit[0, :]**2 + p1_fit[1, :]**2)
        )


# Functions
def kernel(A, both = False):
	U, s, Vh = np.linalg.svd(A)
	if both:
		return Vh.T[:, -1], U[:, -1]
	else:
		return Vh.T[:, -1]


def reduce_rank(A, n = 1):
	U, s, Vh = np.linalg.svd(A)
	return np.dot(
        np.dot(
            U,
            np.diag(np.hstack((s[:-n], np.zeros(n))))), 
        Vh)
