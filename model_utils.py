import numpy as np

def interaction_se(coef, cov):
    ses = []
    for i in range(len(coef)):
        for j in range(len(coef)):
            try:
                ses.append(cov["1"][coef[i]]["1"][coef[j]])
            except KeyError:
                ses.append(cov[coef[i]][coef[j]])
    return np.sqrt(np.sum(ses))


def interaction_coef(coef, params):
    return np.sum(np.array(params.loc[coef]))

