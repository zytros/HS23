import numpy as np

def resample(particles, particles_w):
    """
    Resample particles according to their weights.
    :param particles: np.array of shape (n_particles, 2)
    :param particles_w: np.array of shape (n_particles, )
    """
    
    indices = np.random.choice(np.arange(particles.shape[0]),size=particles.shape[0],p=particles_w.flatten())
    
    samples_w = particles_w[indices]
    samples_w /= np.sum(samples_w)
    
    return particles[indices], samples_w