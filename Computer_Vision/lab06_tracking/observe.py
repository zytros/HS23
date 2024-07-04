import numpy as np
from color_histogram import color_histogram
from chi2_cost import chi2_cost

def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist, sigma_observe):
    """
    Observe the particles and return the weights.
    """
    part_weights = np.ones((particles.shape[0], 1))
    
    x1 = particles[:,0] - bbox_width / 2
    x2 = particles[:,0] + bbox_width / 2
    y1 = particles[:,1] - bbox_height / 2
    y2 = particles[:,1] + bbox_height / 2
    
    for i in range(particles.shape[0]):
        dist = chi2_cost(color_histogram(x1[i], y1[i], x2[i], y2[i], frame, hist_bin), hist)
        pdf =  np.exp(-dist**2/( 2*(sigma_observe**2) ))
        part_weights[i] = 1/(np.sqrt(2*np.pi)*sigma_observe) * pdf
        
    part_weights /= np.sum(part_weights)

    return part_weights