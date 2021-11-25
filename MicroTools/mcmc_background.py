import matplotlib
matplotlib.use('Agg')
import numpy as np
import emcee
import sys
import tqdm
import time
import argparse
import scipy.stats as stats

from . import miniboone_neutrino_improved_fit as mbfit
#import . miniboone_neutrino_improved_fit as mbfit

#parser = argparse.ArgumentParser()

#parser.add_argument('-t',
#		dest='threads',
#		type=int, 
#		default=6,
#		help='Cores'
#		   )
#parser.add_argument('-w',
#		dest='walkers',
#		type=int, 
#		default=50,
#		help='Number of Walkers'
#		   )
#parser.add_argument('-b',
#		dest='burnin',
#		type=int, 
#		default=20000,
#		help='Burnin steps'
#		   )
#parser.add_argument('-s',
#		dest='steps',
#		type=int, 
#		default=50000,
#		help='MCMC steps'
#		   )

#args = parser.parse_args()

number_of_backgrounds = 3
number_of_bins = 11

NuEM = [0.26881717, 0.43322491, 0.56813788, 0.50590962, 0.49705416, 0.49174142,
        0.40728966, 0.33094358, 0.21159324, 0.15568566, 0.]


NuEKp = [0.07271037, 0.1222734,  0.17862495, 0.23069437, 0.23060864, 0.20047603,
         0.22411117, 0.19365187, 0.16835922, 0.13554328, 0.04772517]


NuEK0 = [0.04481658, 0.04755179, 0.04398599, 0.07189589, 0.05540792, 0.08263944,
         0.05338303, 0.05004523, 0.02752273, 0.03580412, 0.]

Pi0 = [2.05191139, 1.14640005, 0.68066036, 0.35258565, 0.30266261, 0.15816352,
       0.16398261, 0.13065389, 0.09875218, 0.04886478, 0.01253574]

Delta = [0.39175775, 0.55264774, 0.52771141, 0.22134511, 0.10582993, 0.04187509,
         0.005233,   0.,         0.,         0.,         0.]

Dirt = [0.24410324, 0.17854509, 0.12780866, 0.10610038, 0.06358978, 0.02739863,
        0.02747221, 0.01703119, 0.,         0.,         0.,]

Other = [0.53574058, 0.39505153, 0.26406301, 0.2078761,  0.17094773, 0.06413129,
         0.04629203, 0.02970044, 0.,         0.,         0.]

Sterile = [4.07399, 3.28421, 2.93342, 2.223875, 1.78651, 1.301688, 1.07318, 0.8050, 0.544665, 0.3872957, 0.0817373]

BinBound = [200.0,  300.0,  375.0,  475.0,  550.0,  675.0,  800.0,  950.0,  1100.0,  1250.0,  1500.0,  3000.0]

WBkg =np.zeros(number_of_bins)
def RelativeNorm():
    
    for i in range(number_of_bins): 
        WBkg[i] = (BinBound[i+1] - BinBound[i]) * (Other[i] + Dirt[i] + Delta[i] + Pi0[i] + NuEK0[i] + NuEKp[i] + NuEM[i])/mbfit.mb_nue_analysis_predicted_background[i]



SumExcs = 0
SumOD = 0
SumDl = 0
SumPi = 0
SumEn = 0
SumDa = 0
def SumBkg():
    global SumExcs
    global SumOD
    global SumDl
    global SumPi
    global SumEn
    global SumDa
    
    SumExcs =  np.sum(mbfit.mb_nominal_excess)

    for i in range(number_of_bins):
       SumOD += ((BinBound[i+1] - BinBound[i])/WBkg[i]) * (Other[i] + Dirt[i])
       SumDl += ((BinBound[i+1] - BinBound[i])/WBkg[i]) * Delta[i]
       SumPi += ((BinBound[i+1] - BinBound[i])/WBkg[i]) * Pi0[i]
       SumEn += ((BinBound[i+1] - BinBound[i])/WBkg[i]) * (NuEK0[i] + NuEKp[i] + NuEM[i])
       SumDa += mbfit.mb_nue_analysis_data[i]

       
def CalculateNewBackgrounds(n_background_vector):
    background_neutrino =np.zeros(number_of_bins)

    eta0 = (SumDa - n_background_vector[0] * SumDl - n_background_vector[1] * SumPi - n_background_vector[2] * SumEn - SumExcs)/SumOD
    if eta0 < 0 :
        return background_neutrino
    
    for i in range(number_of_bins): 
        background_neutrino[i] = ((BinBound[i+1] - BinBound[i])/WBkg[i]) * ((Other[i] + Dirt[i]) * eta0 + Delta[i] * n_background_vector[0] + Pi0[i] * n_background_vector[1] + (NuEK0[i] + NuEKp[i] + NuEM[i])  * n_background_vector[2])

        

    return background_neutrino

    
def GetLikelihood(n_background_vector):
    background_neutrino = CalculateNewBackgrounds(n_background_vector)

    global_chi2 = 0.0
    global_chi2 += mbfit.CalculateChi2BackgroundExpectations(background_neutrino)

    return -global_chi2/2. # converting from chi^2 to gaussian likelihood

def lnprior(n_background_vector):
    if any(n<0 for n in n_background_vector):
        return -np.inf
    return 0.0

def lnprob(theta):
            
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + GetLikelihood(theta)

#if __name__ == "__main__":
def run_mcmc_background(nwalkers=50, nsteps = 50000, threads=4, burnin_steps = 20000, nbins=11):
    
    RelativeNorm()
    SumBkg()
    
    tt = time.time()
    print("Initializing walkers")
    #nwalkers = args.walkers
    #nwalkers = nwalkers
    
    p0_base = [1.0]*number_of_backgrounds
    p0_std = [0.1]*number_of_backgrounds
    assert(len(p0_base) == len(p0_std))
    ndim = len(p0_base)
    p0 = np.random.normal(p0_base, p0_std, size=[nwalkers, ndim])


    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=args.threads)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=threads)

    print("Running burn-in")

    pos, prob, state = sampler.run_mcmc(p0, 1)
    #burnin_steps = args.burnin
    for _ in tqdm.tqdm(sampler.sample(pos, iterations=burnin_steps), total=burnin_steps):
        pass
    sampler.reset()
        
    #nsteps = args.steps

    for _ in tqdm.tqdm(sampler.sample(pos, iterations=nsteps), total=nsteps):
        pass
    print("Time elapsed", time.time()-tt)

    samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

    np.savetxt("./miniboone_background_chain_NoAbs.dat",sampler.flatchain)

    import corner
    fig = corner.corner(samples)
    fig.savefig("./miniboone_background_triangle_new.png")

    
