
import numpy as np
from matplotlib import pyplot as plt
import cv2

def transform_lms(lms, xform):

    k = lms.shape[0]

    ones = np.ones((k,1))

    lms__ =  np.hstack((lms, ones))
    lms__ =   np.matmul(lms__, xform)

    return lms__[:,0:2]


def my_procrustes(X, Y):

    mu_X = np.mean(X, axis=0)
    mu_Y = np.mean(Y, axis=0)

    X0 = X - mu_X
    Y0 = Y - mu_Y

    t  = mu_X-mu_Y

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
    mat = np.zeros((2,2), dtype=np.float32)

    for i in range(k):

        mat[0, :] =  Y0[i, :]
        mat[1, :] =  X0[i, :]

        numerator   += np.linalg.det(mat)
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




'''
folder = '/media/morzh/ext4_volume/data/Faces/to_process/set_002/accepted/'
file_1 ='0f23b1f4f067906216b3b51081dc1d3c.jpg.dan'
file_2 ='2cecf1bafcf47dd3ee477b5499316288.jpg.dan'

imfile_1 ='0f23b1f4f067906216b3b51081dc1d3c.jpg'
imfile_2 ='2cecf1bafcf47dd3ee477b5499316288.jpg'

im1 =cv2.imread(folder+imfile_1)
im2 =cv2.imread(folder+imfile_2)
cv2.imshow('im1', im1)
cv2.imshow('im2', im2)
cv2.waitKey(-1)

lms1 = np.loadtxt(folder+file_1)
lms2 = np.loadtxt(folder+file_2)

plt.imshow(im1)
plt.scatter( lms1[:,0] , lms1[:,1])
plt.show()

plt.imshow(im2)
plt.scatter( lms2[:,0] , lms2[:,1])
plt.show()

# print np.sum(np.linalg.norm(lms2-lms1, axis=1)) / 68

lms__, xform = my_procrustes(lms1, lms2)



lms3 = transform_lms(lms2, xform.T)

im__ = cv2.warpPerspective(im2, xform, (im1.shape[1], im1.shape[0]) )
im2__ = cv2.warpPerspective(im1, xform, (im1.shape[1], im1.shape[0]) )

cv2.imshow('im1', im1)
cv2.imshow('im2', im__)
cv2.waitKey(-1)


plt.scatter( lms3[:, 0], lms3[:, 1], s=30, c='k')
plt.scatter( lms__[:, 0], lms__[:, 1], s=10, c='r')
plt.scatter( lms1[:, 0], lms1[:, 1])
# plt.scatter( lms2[:, 0], lms2[:, 1], s=10)


ax = plt.gca()  # get the axis
ax.set_ylim(ax.get_ylim()[::-1])  # invert the axis
ax.xaxis.tick_top()
plt.show()

print np.sum(np.linalg.norm(lms2-lms1, axis=1)) / 68

'''