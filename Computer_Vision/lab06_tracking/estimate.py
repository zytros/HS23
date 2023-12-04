import numpy as np

def estimate(particles, particles_w):
    """
    Estimate the state of the object.
    :param particles: (N, 2) array of particles
    :param particles_w: (N, ) array of particles' weights
    :return: (2, ) array of estimated state
    """
    return np.sum(particles * particles_w,axis=0)