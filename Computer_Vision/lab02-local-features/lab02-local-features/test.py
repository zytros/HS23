import numpy as np
import cv2

from scipy import signal #for the scipy.signal.convolve2d function
from scipy import ndimage #for the scipy.ndimage.maximum_filter

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img_c = img.copy()
    img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    img = img.astype(float) / 255.0

    # 1. Compute image gradients in x and y direction
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    ky = np.array([[-1,-2,-1] ,[0,0,0], [1,2,1]])
    dx=signal.convolve2d(img,kx, "same")
    dy=signal.convolve2d(img,ky, "same")
    #dx = cv2.normalize(dx, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #dy = cv2.normalize(dy, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    
    # 2. Blur the computed gradients
    # TODO: compute the blurred image gradients
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)
    dx2 = ndimage.gaussian_filter(dx**2, sigma)
    dy2 = ndimage.gaussian_filter(dy**2, sigma)
    dxy = ndimage.gaussian_filter(dx*dy, sigma)


    # 3. Compute elements of the local auto-correlation matrix "M"
    # TODO: compute the auto-correlation matrix here
    
    #done above
    

    # 4. Compute Harris response function C
    # TODO: compute the Harris response function C here
    

    detA = dx2 * dy2 - dxy ** 2
    traceA = dx2 + dy2
    
    C = detA - k * (traceA ** 2)
    
    C = cv2.normalize(C, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    

    # 5. Detection with threshold and non-maximum suppression
    # TODO: detection and find the corners here
    # For the non-maximum suppression, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    # You may refer to np.where to find coordinates of points that fulfill some condition; Please, pay attention to the order of the coordinates.
    # You may refer to np.stack to stack the coordinates to the correct output format
    print('sigma = ',sigma)
    print('min',np.min(C))
    print('max',np.max(C))
    print('--------------------')
    loc = np.where(C > thresh)
    
    for pt in zip(*loc[::-1]):
        cv2.circle(img, pt, 1, 0, -1)
    
    cv2.imshow('C',C)
    cv2.waitKey(0)
    non_max_suppression = ndimage.maximum_filter(C, size=(3,3))
    cv2.imshow('non_max_suppression',non_max_suppression)
    cv2.waitKey(0)
    corners = 0
    
    return corners, C

def main():
    _, C = extract_harris(cv2.imread("images/blocks.jpg"))

main()