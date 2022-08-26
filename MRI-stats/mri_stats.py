import numpy as np
import scipy


def c4(n):
    '''
    c4(n) = sqrt(2/n-1) (gamma(n/2)/gamma(n-1/2))
    '''
    gamma = scipy.special.gamma
    return np.sqrt(2/(n-1)) * (gamma(n/2)/gamma((n-1)/2))


def get_mean_reference(reference):
    '''
    Compute mean for reference sample
    '''
    reference_data = [image.get_fdata() for image in reference]
    return np.ma.mean(reference_data, axis=0, dtype=np.float64)


def get_std_reference(reference):
    '''
    Unbiased estimator for standard deviation with small sample size.
    '''
    reference_data = np.array([image.get_fdata() for image in reference])
    if reference.size <= 10:
        coef = c4(reference.shape[0])
        return np.ma.std(reference_data, axis=0, ddof=1, dtype=np.float64) / coef
    else:
        return np.ma.std(reference_data, axis=0, dtype=np.float64)


def z_score(x, mean, std):
    '''
    Compute Z-score
    '''
    return (x-mean)/std


def p_values_z_score(x, mean, std, dof):
    '''
    Compute p-values from Z-score
    '''
    z = z_score(x, mean, std)
    return scipy.stats.norm.sf(np.abs(z))*2


def t_score(x, mean, std, nsample):
    '''
    Compute T-score
    '''
    return (x-mean) / (std/np.sqrt(nsample))


def p_values_t_score(x, mean, std, dof):
    '''
    Compute p-values from T-score
    '''
    t = t_score(x, mean, std, dof+1)
    return scipy.stats.t.sf(np.abs(t), df=dof)*2


def get_score(x, N, z_score=True):
    '''
    Get score depending on 
    '''
    if z_score:
        score_name = 'Z-score'
        score = p_values_z_score
    else:
        score_name = 'T-score'
        score = p_values_t_score
    return (score_name, score)
