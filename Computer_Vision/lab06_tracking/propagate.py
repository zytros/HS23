import numpy as np

def propagate(particles, frame_height, frame_width, params):
    """
    Propagate particles according to the dynamic model.
    :param particles: particles to propagate
    :param frame_height: height of the frame
    :param frame_width: width of the frame
    :param params: parameters of the tracker
    :return: propagated particles
    """
    model = params['model']
    sigma_pos = params['sigma_position']
    sigma_vel = params['sigma_velocity']
    mod0_shape = 2
    mod1_shape = 4
    A = np.identity(mod0_shape)
    p = np.zeros(particles.shape)
    
    if model == 1:
        A = np.identity(mod1_shape)
        A[0,2] = 1
        A[1,3] = 1
        n = np.random.normal(0, sigma_vel, p[:,2:4].shape)
        p[:,:2] += n

    n = np.random.normal(0, sigma_pos, p[:,:2].shape)
    p[:,:2] += n
    particles_new = np.matmul(A, particles.T).T + p
    particles_new[:,0] = np.clip(particles_new[:,0], 0, frame_width-1)
    particles_new[:,1] = np.clip(particles_new[:,1], 0, frame_height-1)
    
    return particles_new