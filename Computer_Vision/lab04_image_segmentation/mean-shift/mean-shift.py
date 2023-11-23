import time
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

def distance(x, X):
    #use one func to calc all dist
    a = np.linalg.norm(X, axis=1) * np.ones((X.shape[0], X.shape[0]))
    b = (np.linalg.norm(X, axis=1) * np.ones((X.shape[0], X.shape[0]))).transpose()
    ab = np.matmul(X, X.transpose())
    dist = np.sqrt(np.abs(a**2 - 2*ab + b**2))
    return dist

def gaussian(dist, bandwidth):
    d = np.exp(-0.5 * (dist/bandwidth)**2)
    return d


def update_point(weights, X):
    s = np.sum(weights,axis=1,keepdims=True)
    #print(s)
    return np.matmul(weights, X) / s

<<<<<<< HEAD
def meanshift_step(X, bandwidth=10):
=======
def meanshift_step(X, bandwidth=4):
>>>>>>> f44cdc93de53ffe9ff55a14c91b039eb9a855299
    X_new = X.copy()
    dist = distance(X, X)
    weight = gaussian(dist, bandwidth)
    X_new = update_point(weight, X)
    return X_new

def meanshift(X,):
    for i in range(20):
        t = time.time()
        X = meanshift_step(X)
        print('Iteration {} in {}s'.format(i, time.time() - t))
    return X

scale = 0.5   # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('eth.jpg'), scale, channel_axis=-1)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image


# Run your mean-shift algorithm
t = time.time()
X = meanshift(image_lab)
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, channel_axis=-1)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
