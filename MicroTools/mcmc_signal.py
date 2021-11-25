import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import corner
import numpy as np
import scipy
import emcee
import sys
import tqdm
import time
from pathlib import Path
from scipy.interpolate import interp1d

local_dir = Path(__file__).parent
from . import miniboone_fit as mbfit

# ##############################
# # IMPORT IVAN/YUBER AND KEVIN
# path = f'{os.path.split(os.getcwd())[0]}/../../'
# nb_dir = os.path.split(os.getcwd())[0]
# if nb_dir not in sys.path:
#     sys.path.append(path)

from . import CERNInclusive as cern
from . import unfolder 

# converter = oh.MBtomuB(analysis='1eX', remove_high_energy=False, unfold=True)
# def get_microboone_chi2s_in_x(muB_templates, Npoints_for_x=10):
#     print(np.shape(muB_templates))
#     signal_strenghts = np.linspace(0, 2, Npoints_for_x)
#     chi2s_signal_strengths = np.empty(0)
#     for x in signal_strenghts:
#         for template in muB_templates:
#             template = x*template
#             chi2s_signal_strengths = np.append(chi2s_signal_strengths, cern.Chi2_Inclusive(template))
#     return signal_strenghts, np.reshape(chi2s_signal_strengths,  (Npoints_for_x, np.shape(muB_templates)[0]) ).T
# micro_nominal_template = np.append(0.0, converter.miniToMicro(mbfit.mb_nominal_excess))
# x_nominal, chi2_nominal = get_microboone_chi2s_in_x([micro_nominal_template], Npoints_for_x=100)
# f = interp1d(x_nominal, chi2_nominal[0])

NEVENTS_NOMINAL = 360

def CalculateNewBackgrounds(n_background_vector):
    background_neutrino =np.zeros(number_of_bins)

    eta0 = (SumDa - n_background_vector[0] * SumDl - n_background_vector[1] * SumPi - n_background_vector[2] * SumEn - SumExcs)/SumOD
    if eta0 < 0 :
        return background_neutrino
    
    for i in range(number_of_bins): 
        background_neutrino[i] = ((BinBound[i+1] - BinBound[i])/WBkg[i]) * ((Other[i] + Dirt[i]) * eta0 + Delta[i] * n_background_vector[0] + Pi0[i] * n_background_vector[1] + (NuEK0[i] + NuEKp[i] + NuEM[i])  * n_background_vector[2])

    return background_neutrino

def GetLikelihood(parameters, background_model=False):

    if background_model:
        background_neutrino = CalculateNewBackgrounds(parameters)
        chi2_mb = mbfit.CalculateChi2BackgroundExpectations(background_neutrino)

    # # Get microboone chi2
    # template = n_signal_vector
    # template[template<0] = 0 
    # mub_signal_vector = np.append(0.0, converter.miniToMicro(template))
    # chi2_mub = cern.Chi2_Inclusive(mub_signal_vector)

    else:
         chi2_mb = mbfit.CalculateChi2SignalExpectations(parameters)
    return -chi2_mb/2. # converting from chi^2 to gaussian likelihood
    
    # quote_chi2_unquote = np.sum((n_signal_vector - mbfit.mb_nominal_excess)**2/2/np.sqrt(mbfit.mb_nue_analysis_data))
    # return - quote_chi2_unquote/2

def lnprior(n_signal_vector, background_model=False):
    if np.any(n_signal_vector < 0):
        return -np.inf
    # if (np.sum(n_signal_vector) > 1000 or np.sum(n_signal_vector) < 11):
    #     return -np.inf
    # if any(n<0 for n in n_signal_vector):
    #     return -np.inf
    # else:
    return 0.0#np.sum(np.log(1/np.sqrt(n_signal_vector)))

def lnprob(n_signal_vector, nbins=11, background_model=False):

    # assign leftover high energy bins to have zero excess
    if nbins < 11 and not background_model:
        n_signal_vector = np.append(n_signal_vector, np.zeros(11-nbins))

    lp = lnprior(n_signal_vector, background_model=background_model)

    if not np.isfinite(lp):
        return -np.inf
    
    L = GetLikelihood(n_signal_vector, background_model=background_model)

    if np.isfinite(L):
        return lp + L
    else:
        return -np.inf

### MCMC business
def run_mcmc(nwalkers=50, 
            nsteps = 3000, 
            threads=4, 
            burnin_steps = 1000, 
            nbins=11, 
            background_model=False,
            appendix=''):
    tt = time.time()
    print("Initializing walkers")

    assert nbins <= 11, "Asked for too many bins, miniboone data only has 11 Enu bins."

    if background_model:
        p0_base = np.ones(nbins)
        p0_std = [0.3]*nbins

    else:
        # p0_base = mbfit.mb_nominal_excess[:nbins]
        p0_base = np.ones(nbins)*50
        p0_std = [5]*nbins

    assert(len(p0_base) == len(p0_std))
    ndim = len(p0_base)
    p0 = np.random.normal(p0_base, p0_std, size=[nwalkers, ndim])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=threads, args=[nbins,background_model])

    print("Running burn-in")
    assert nwalkers > 2*ndim, "Number of walkers too low, ensure nwalkers > 2*ndims"
    

    pos, prob, state = sampler.run_mcmc(p0, burnin_steps)
    sampler.reset() 
    pos, prob, state = sampler.run_mcmc(p0, nsteps)

    # for _ in tqdm.tqdm(sampler.sample(pos, iterations=burnin_steps), total=burnin_steps):
    #     pass
    # sampler.reset() 

    # print("Running proper mcmc")
    # for _ in tqdm.tqdm(sampler.sample(pos, iterations=nsteps), total=nsteps):
    #     pass
    # print("Time elapsed", time.time()-tt)

    # samples = sampler.chain[:, nwalkers:, :].reshape((-1, ndim))
    samples = sampler.get_chain(discard=burnin_steps, flat=True)#, thin=15)
    
    if background_model:
    
        complete_templates = sampler.flatchain.T

    else:
        if nbins < 11:
            complete_templates = np.concatenate((sampler.flatchain, np.zeros((np.shape(sampler.flatchain)[0],11-nbins))), axis=1).T
        else:
            complete_templates = sampler.flatchain.T

    if not appendix:
        if background_model:
            appendix='bkg'
        else:
            appendix='free'
    
    np.savetxt(f"{local_dir}/mcmc_results/miniboone_signal_chain_{nbins}_bins_{appendix}.dat", complete_templates)

    # labels=[f'{mbfit.bin_edges[i]} MeV - {mbfit.bin_edges[i+1]} MeV' for i in range(nbins)]
    labels=[f'bin {i}' for i in range(nbins)]
    
    fig = corner.corner(samples, show_titles=True,  plot_datapoints=False,
                                title_kwargs={"fontsize": 11}, 
                                smooth=True,labels=labels)
    fig.savefig(f"{local_dir}/mcmc_results/miniboone_signal_triangle_{nbins}_bins.png")
    plt.close()

    return sampler



######################  low-level functions #######################

def new_template(x, n=2):
    # assert x >= 1, "You tried upscaling, only downscaling allowed."
    # the nominal LEE
    BF_template = mbfit.mb_nue_analysis_data-mbfit.mb_nue_analysis_predicted_background
    # downscale the first n bins
    downscaled_bins = BF_template[:n]/x
    # distribute the difference evenly
    recipients = range(n,len(BF_template))
    bin_size_correction = mbfit.bin_width[recipients]/np.sum(mbfit.bin_width[recipients])
    # bin_size_correction = BF_template[recipients]/np.sum(BF_template[recipients])
    BF_template[recipients] += np.sum(BF_template[:n]-downscaled_bins)*bin_size_correction
    BF_template[:n] = downscaled_bins
    return BF_template

def get_templates_for_all_scalings(xrange, n=2):
    chi2s = []
    scales= np.linspace(*xrange, 1000)
    for x in scales:
        chi2s.append(mbfit.CalculateChi2SignalExpectations(new_template(x, n = n)))
        # chi2s.append(get_pvalue(new_template(x, n = n)))
    return scales, chi2s


def get_pvalue(template, ndf=8.7):
    chi2 = mbfit.CalculateChi2SignalExpectations(template)
    return scipy.stats.chi2.sf(chi2, ndf)

def get_pvalue_muboone(chi2):
    return scipy.stats.chi2.sf(chi2, 25 - 11)

def filter_samples(samples, p_threshold = 0.1):
    filtered = np.array([])
    for template in samples:
        if get_pvalue(template) > p_threshold: #and template[0]>mbfit.mb_nominal_excess[0]:
            filtered = np.append(filtered,template)
    return np.reshape(filtered, (int(len(filtered)/11), 11))
    
def get_max_avgenu_templates(samples, n=4):
    avgenus = np.sum((samples.T*mbfit.bin_centers).T/np.sum(samples, axis=0),axis=0)
    ind = np.argpartition(avgenus, -n)[-n:]
    return avgenus[ind], samples.T[ind]