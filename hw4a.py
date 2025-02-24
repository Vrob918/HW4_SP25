#region imports
import math
from random import random as rnd
from scipy.integrate import quad
from scipy.optimize import fsolve
from copy import deepcopy as dc
#endregion

#region functions
def ln_PDF(args):
    '''
    Computes f(D) for the log-normal probability density function.
    :param args: (D, mu, sigma)
    :return: f(D)
    '''
    D, mu, sig = args  # unpack the arguments
    if D == 0.0:
        return 0.0
    p = 1/(D*sig*math.sqrt(2*math.pi))
    _exp = -((math.log(D)-mu)**2)/(2*sig**2)
    return p*math.exp(_exp)

def tln_PDF(args):
    """
    Computes the value of the truncated log-normal probability density function.
    :param args: tuple (D, mu, sig, F_DMin, F_DMax)
    :return: f(D)
    """
    D, mu, sig, F_DMin, F_DMax = args
    return ln_PDF((D, mu, sig)) / (F_DMax - F_DMin)

def F_tlnpdf(args):
    '''
    Integrates the truncated log-normal probability density function from D_Min to D.
    :param args: tuple (mu, sig, D_Min, D_Max, D, F_DMax, F_DMin)
    :return: Integrated probability from D_Min to D
    '''
    mu, sig, D_Min, D_Max, D, F_DMax, F_DMin = args
    if D > D_Max or D < D_Min:
        return 0
    # Use quad to integrate tln_PDF from D_Min to D.
    P, _ = quad(lambda D_val: tln_PDF((D_val, mu, sig, F_DMin, F_DMax)), D_Min, D)
    return P

def makeSample(args, N=100):
    """
    Computes D for each of the N random probabilities in the sample size using the truncated log-normal PDF.
    :param args: a tuple (ln_Mean, ln_sig, D_Min, D_Max, F_DMax, F_DMin)
    :param N: number of items in the sample
    :return: d_s, a list of rock sizes
    """
    ln_Mean, ln_sig, D_Min, D_Max, F_DMax, F_DMin = args
    # Generate uniformly random probability values.
    probs = [rnd() for _ in range(N)]
    # For each probability p, find D such that the integrated probability equals p.
    # fsolve takes a function and an initial guess (here, the midpoint).
    d_s = [
        fsolve(
            lambda D: F_tlnpdf((ln_Mean, ln_sig, D_Min, D_Max, D, F_DMax, F_DMin)) - p,
            (D_Max + D_Min) / 2
        )[0]
        for p in probs
    ]
    return d_s

def sampleStats(D, doPrint=False):
    """
    Computes the mean and variance of the values in D.
    :param D: list of values
    :param doPrint: if True, prints the mean and variance
    :return: (mean, var)
    """
    N = len(D)
    mean = sum(D) / N
    var = sum((d - mean)**2 for d in D) / (N - 1)
    if doPrint:
        print(f"mean = {mean:0.3f}, var = {var:0.3f}")
    return (mean, var)

def getPreSievedParameters(args):
    """
    Prompts user to input the mean and standard deviation for the log-normal PDF.
    :param args: default values (mean_ln, sig_ln)
    :return: (mean_ln, sig_ln)
    """
    mean_ln, sig_ln = args
    st_mean_ln = input(f'Mean of ln(D) for the pre-sieved rocks? (ln({math.exp(mean_ln):0.1f}) = {mean_ln:0.3f}, where D is in inches): ').strip()
    mean_ln = mean_ln if st_mean_ln == '' else float(st_mean_ln)
    st_sig_ln = input(f'Standard deviation of ln(D) for the pre-sieved rocks? ({sig_ln:0.3f}): ').strip()
    sig_ln = sig_ln if st_sig_ln == '' else float(st_sig_ln)
    return (mean_ln, sig_ln)

def getSieveParameters(args):
    """
    Prompts user for the sieve parameters.
    :param args: (D_Min, D_Max)
    :return: (D_Min, D_Max)
    """
    D_Min, D_Max = args
    st_D_Max = input(f'Large aperture size? ({D_Max:0.3f}): ').strip()
    D_Max = D_Max if st_D_Max == '' else float(st_D_Max)
    st_D_Min = input(f'Small aperture size? ({D_Min:0.3f}): ').strip()
    D_Min = D_Min if st_D_Min == '' else float(st_D_Min)
    return (D_Min, D_Max)

def getSampleParameters(args):
    """
    Prompts user for sample parameters.
    :param args: (N_samples, N_SampleSize)
    :return: (N_samples, N_SampleSize)
    """
    N_samples, N_sampleSize = args
    st_N_Samples = input(f'How many samples? ({N_samples}): ').strip()
    N_samples = N_samples if st_N_Samples == '' else float(st_N_Samples)
    st_N_SampleSize = input(f'How many items in each sample? ({N_sampleSize}): ').strip()
    N_sampleSize = N_sampleSize if st_N_SampleSize == '' else float(st_N_SampleSize)
    return (N_samples, N_sampleSize)

def getFDMaxFDMin(args):
    """
    Computes F_DMax and F_DMin using the log-normal distribution.
    :param args: (mean_ln, sig_ln, D_Min, D_Max)
    :return: (F_DMin, F_DMax)
    """
    mean_ln, sig_ln, D_Min, D_Max = args
    F_DMax, _ = quad(lambda D: ln_PDF((D, mean_ln, sig_ln)), 0, D_Max)
    F_DMin, _ = quad(lambda D: ln_PDF((D, mean_ln, sig_ln)), 0, D_Min)
    return (F_DMin, F_DMax)

def makeSamples(args):
    """
    Generates samples and computes the sample means.
    :param args: (mean_ln, sig_ln, D_Min, D_Max, F_DMax, F_DMin, N_sampleSize, N_samples, doPrint)
    :return: Samples, Means
    """
    mean_ln, sig_ln, D_Min, D_Max, F_DMax, F_DMin, N_sampleSize, N_samples, doPrint = args
    Samples = []
    Means = []
    for n in range(int(N_samples)):
        sample = makeSample((mean_ln, sig_ln, D_Min, D_Max, F_DMax, F_DMin), N=int(N_sampleSize))
        Samples.append(sample)
        sample_Stats = sampleStats(sample)
        Means.append(sample_Stats[0])
        if doPrint:
            print(f"Sample {n}: mean = {sample_Stats[0]:0.3f}, var = {sample_Stats[1]:0.3f}")
    return Samples, Means

def main():
    '''
    Simulates a gravel production process where the rock size distribution follows a log-normal distribution
    that is sieved between two screens. It then produces several samples from the truncated distribution,
    computing the mean and variance for each sample as well as for the overall sampling mean.
    '''
    # Setup default values.
    mean_ln = math.log(2)  # in inches
    sig_ln = 1
    D_Max = 1
    D_Min = 3.0/8.0
    N_samples = 11
    N_sampleSize = 100
    goAgain = True

    while goAgain:
        mean_ln, sig_ln = getPreSievedParameters((mean_ln, sig_ln))
        D_Min, D_Max = getSieveParameters((D_Min, D_Max))
        N_samples, N_sampleSize = getSampleParameters((N_samples, N_sampleSize))
        F_DMin, F_DMax = getFDMaxFDMin((mean_ln, sig_ln, D_Min, D_Max))

        Samples, Means = makeSamples((mean_ln, sig_ln, D_Min, D_Max, F_DMax, F_DMin, N_sampleSize, N_samples, True))

        stats_of_Means = sampleStats(Means)
        print(f"Mean of the sampling mean:  {stats_of_Means[0]:0.3f}")
        print(f"Variance of the sampling mean:  {stats_of_Means[1]:0.6f}")
        goAgain = input('Go again? (No): ').strip().lower().__contains__('y')

if __name__ == '__main__':
    main()
