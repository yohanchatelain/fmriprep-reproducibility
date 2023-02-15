import numpy as np
import scipy
import tqdm

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
    std = np.ma.std(reference_data, axis=0, ddof=1, dtype=np.float64)
    if reference.size <= 10:
        return std / c4(reference.shape[0])
    else:
        return std


def get_sem_reference(reference):
    reference_data = np.array([image.get_fdata() for image in reference])
    sem = scipy.stats.sem(reference_data, axis=0, ddof=1, dtype=np.float64)
    return sem


def z_score(x, mean, std):
    '''
    Compute Z-score
    '''
    z = (x-mean)/std
    if mean.ndim == 1:
        return z
    else:
        np.min(z, axis=0)

    
def gmm_ppf(pdf, cdf, alpha, x):
    
    low = optimize(cdf, alpha/2, x)
    high = optimize(cdf, 1-alpha/2, x)
    return low, high
               
def p_values_z_score(x, mean, std, weights, alpha):
    '''
    Compute p-values from Z-score
    '''
    def cdf(x):
        return weights[0] * scipy.stats.norm.cdf(x, loc=mean[0], scale=std[0]) + \
            weights[1] * scipy.stats.norm.cdf(x, loc=mean[1], scale=std[1])
    def cdf_local(i,x):
        return weights[0] * scipy.stats.norm.cdf(x, loc=mean[0][i], scale=std[0][i]) + \
            weights[1] * scipy.stats.norm.cdf(x, loc=mean[1][i], scale=std[1][i])

    funs = lambda i : lambda z: weights[0] * scipy.stats.norm.cdf(z, loc=mean[0][i], scale=std[0][i]) + \
               weights[1] * scipy.stats.norm.cdf(z, loc=mean[1][i], scale=std[1][i])

    print(funs(0)(0))
    if mean.ndim > 1:
        #z = np.min(z, axis=0)
        w0, w1 = weights
        lt,ht = gmm_ppf(funs, alpha, x)
        return 2 * np.min( (cdf(lt), 1-cdf(ht)), axis=0 )
    else:
        z = np.abs((x-mean)/std)
        return scipy.stats.norm.sf(z)*2


def t_score(x, mean, std=None):
    '''
    Compute T-score
    '''
    sem = scipy.stats.sem(x, axis=0, ddof=1, dtype=np.float64)
    return np.abs(x-mean) / sem


def p_values_t_score(x, mean, std=None):
    '''
    Compute p-values from T-score
    '''
    t = t_score(x, mean)
    return scipy.stats.t.sf(np.abs(t), df=len(x)-1)*2


def get_score(score='z-score'):
    '''
    Get score
    '''
    if score == 'z-score':
        score_name = 'Z-score'
        score = p_values_z_score
    elif score == 't-score':
        score_name = 'T-score'
        score = p_values_t_score
    else:
        msg = f'Unknown score {score}'
        raise Exception(msg)

    return (score_name, score)
