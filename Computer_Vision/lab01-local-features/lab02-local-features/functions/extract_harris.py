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
    img = img.astype(float) / 255.0

    MAX_FILTER_SIZE = 15
    
    # 1. Compute image gradients in x and y direction
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    ky = np.array([[-1,-2,-1] ,[0,0,0], [1,2,1]])
    dx=signal.convolve2d(img,kx, "same")
    dy=signal.convolve2d(img,ky, "same")
    
    
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
    

    # 5. Detection with threshold and non-maximum suppression
    # TODO: detection and find the corners here
    # For the non-maximum suppression, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    # You may refer to np.where to find coordinates of points that fulfill some condition; Please, pay attention to the order of the coordinates.
    # You may refer to np.stack to stack the coordinates to the correct output format
    
    thresh_mask = C > thresh
    C_filtered = ndimage.maximum_filter(C, size=MAX_FILTER_SIZE)
    diff_mask = abs(C_filtered - C) < 1e-5
    coords = thresh_mask & diff_mask
    y,x = np.where(coords)
    corners = np.stack((x,y), axis=1)
    return corners, C

