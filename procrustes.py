
import numpy as np
from matplotlib import pyplot as plt

def transform_lms(lms, xform):
    k = lms.shape[0]
    ones = np.ones((k, 1))
    lms__ = np.hstack((lms, ones))
    lms__ = np.matmul(lms__, xform)
    return lms__[:, 0:2]


def my_procrustes(X, Y):

    mu_X = np.mean(X, axis=0)
    mu_Y = np.mean(Y, axis=0)

    X0 = X - mu_X
    Y0 = Y - mu_Y

    t = mu_X-mu_Y

    k = X.shape[0]

    X0_sqrtsum = np.sum( np.square(X0), axis=0)
    Y0_sqrtsum = np.sum( np.square(Y0), axis=0)

    s1 =  np.sqrt(np.sum(X0_sqrtsum)/ k )
    s2 =  np.sqrt(np.sum(Y0_sqrtsum)/ k )

    s = s1/s2

    X0 = X0/s1
    Y0 = Y0/s2

    numerator = 0
    denumerator = 0
    mat = np.zeros((2, 2), dtype=np.float32)

    for i in range(k):

        mat[0, :] = Y0[i, :]
        mat[1, :] = X0[i, :]

        numerator += np.linalg.det(mat)
        denumerator += np.dot(X0[i, :], Y0[i, :])


    theta = np.arctan( numerator/denumerator)

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    lms_ = s*(Y - mu_Y)
    lms_ = np.matmul(lms_, R.T)
    lms_ += mu_X

    xform_t1 = np.eye(3,3)
    xform_sR = np.eye(3,3)
    xform_sR = np.eye(3,3)
    xform_t2 = np.eye(3,3)

    xform_t1[0:2, 2] = -mu_Y
    xform_sR[0:2, 0:2] = s*R
    xform_t2[0:2, 2] = mu_X

    xform_ = np.matmul(xform_sR, xform_t1)
    xform_ = np.matmul(xform_t2, xform_)

    return lms_, xform_
    # return mu_X, mu_Y, s, R, xform_.T, lms_