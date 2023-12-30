import cv2
import numpy as np

def ORB_create():
    nfeatures = 500
    scaleFactor = 1.2
    nlevels = 8
    edgeThreshold = 31
    firstLevel = 0
    WTA_K = 2
    scoreType = cv2.ORB_HARRIS_SCORE
    patchSize = 31
    fastThreshold = 20
    
    # Create ORB object with default parameters
    orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=scaleFactor, nlevels=nlevels, edgeThreshold=edgeThreshold,
                         firstLevel=firstLevel, WTA_K=WTA_K, scoreType=scoreType, patchSize=patchSize,
                         fastThreshold=fastThreshold)
    
    return orb


class DMatch:
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx
        self.trainIdx = trainIdx
        self.distance = distance


def BFMatcher(norm_type, crossCheck):
    class Matcher:
        def __init__(self, norm_type, crossCheck):
            self.norm_type = norm_type
            self.crossCheck = crossCheck

        def match(self, des1, des2):
            matches = []
            for i in range(des1.shape[0]):
                distances = np.sqrt(np.sum((des1[i] - des2)**2, axis=1))
                if self.crossCheck:
                    j = np.argmin(distances)
                    distances2 = np.sqrt(np.sum((des2[j] - des1)**2, axis=1))
                    if i == np.argmin(distances2):
                        matches.append(DMatch(i, j, distances[j]))
                else:
                    j = np.argmin(distances)
                    matches.append(DMatch(i, j, distances[j]))
            return matches

    return Matcher(norm_type, crossCheck)


def warpPerspective(src, M, dsize):
    # Create output image
    dst = np.zeros((dsize[1], dsize[0], src.shape[2]), dtype=np.uint8)

    # Inverse homography matrix
    Minv = np.linalg.inv(M)

    # Loop through each pixel in the output image
    for y in range(dsize[1]):
        for x in range(dsize[0]):
            # Transform pixel coordinates with homography matrix
            p = np.dot(Minv, np.array([x, y, 1]))
            p = p / p[2]

            # Check if pixel coordinates are within the bounds of the input image
            if p[0] >= 0 and p[0] < src.shape[1] and p[1] >= 0 and p[1] < src.shape[0]:
                # Nearest neighbor interpolation
                dst[y, x, :] = src[int(p[1]), int(p[0]), :]

    return dst

def warpPerspective(img, M, dsize):
    # Create output image
    output_img = np.zeros((dsize[1], dsize[0], img.shape[2]), dtype=np.uint8)

    # Compute inverse transformation matrix
    inv_M = np.linalg.inv(M)

    # Iterate over each pixel in the output image
    for y in range(dsize[1]):
        for x in range(dsize[0]):
            # Transform coordinates using inverse matrix
            src_coord = np.dot(inv_M, [x, y, 1])

            # Normalize coordinates
            src_coord = src_coord / src_coord[2]

            # Check if transformed coordinates are within input image bounds
            if src_coord[0] >= 0 and src_coord[0] < img.shape[1] and \
               src_coord[1] >= 0 and src_coord[1] < img.shape[0]:
                # Interpolate pixel value using bilinear interpolation
                x0 = int(src_coord[0])
                y0 = int(src_coord[1])
                x1 = min(x0+1, img.shape[1]-1)
                y1 = min(y0+1, img.shape[0]-1)
                alpha = src_coord[0] - x0
                beta = src_coord[1] - y0
                output_img[y, x] = (1-alpha)*(1-beta)*img[y0, x0] + \
                                    alpha*(1-beta)*img[y0, x1] + \
                                    (1-alpha)*beta*img[y1, x0] + \
                                    alpha*beta*img[y1, x1]

    return output_img


def addWeighted(src1, alpha, src2, beta, gamma):
    # Check if the dimensions of both images are the same
    assert src1.shape == src2.shape
    
    # Compute the weighted sum of the images
    output = cv2.addWeighted(src1, alpha, src2, beta, gamma)
    
    return output


# Load SAR images
img1 = cv2.imread('referenced.jpeg')
img2 = cv2.imread('sensed.jpeg')

# Convert to Grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect ORB Features
orb = ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# Match Feature Descriptors
bf = BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key = lambda x:x.distance)

# Compute Homography Matrix
src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Warp Sensed Image
warp_img = warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))

# Registered SAR Image
registered_img = addWeighted(warp_img, 0.5, img2, 0.5, 0.0)

import matplotlib.pyplot as plt

def imshow(window_name, image):
   
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Display the image
    plt.imshow(image)
    plt.title(window_name)
    plt.show()

def destroyAllWindows():
    windows = [w for w in cv2.namedWindows()]
    for w in windows:
        cv2.destroyWindow(w)


# Display results
imshow('Registered SAR Image', registered_img)
cv2.waitKey(0)
destroyAllWindows()
