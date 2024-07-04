import numpy as np
import scipy

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please
    # desc1[feature_num, feature]
    
    distances = scipy.spatial.distance.cdist(desc1, desc2, 'sqeuclidean')
    return distances

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here
        # You may refer to np.argmin to find the index of the minimum over any axis
        
        min_idx = np.argmin(distances, axis=1)
        matches = np.stack((np.arange(q1), min_idx), axis=1)
        
    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        # You may refer to np.min to find the minimum over any axis
        
        matches = []
        min_idx1 = np.argmin(distances, axis=1)
        min_idx2 = np.argmin(distances, axis=0)
        
        for i in range(len(min_idx1)):
            j = min_idx1[i]
            if min_idx2[j] == i:
                matches.append((i,j))
        matches = np.array(matches)
        
        
        
    elif method == "ratio":
        # TODO: implement the ratio test matching here
        # You may use np.partition(distances,2,axis=0)[:,1] to find the second smallest value over a row
        
        
        matches = []
        min_idx = np.argmin(distances, axis=1)
        snd_min = np.partition(distances,2,axis=1)[:,1]
        
        for i in range(len(min_idx)):            
            if distances[i][min_idx[i]]/snd_min[i] < ratio_thresh:
                matches.append((i,min_idx[i]))
        matches = np.array(matches)
        
    else:
        raise ValueError("Method not recognized.")
    return matches

