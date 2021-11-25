import numpy as np
from scipy.linalg import inv
from scipy.special import sici

# Pointing to the right path of datafile
from pathlib import Path
local_dir = Path(__file__).parent
bstr = str(local_dir) + "/InclusiveData/DataRelease/"

Sets = ["nueCC_FC_", "nueCC_PC_", "numuCC_FC_", "numuCC_PC_", "numuCCpi0_FC_", "numuCCpi0_PC_", "NCpi0_"]
LEEStr, SigStr, BkgStr, ObsStr = "LEE.npy", "Sig.npy", "Bkg.npy", "Obs.npy"
LEESets = [np.load(bstr+si+LEEStr) for si in Sets]
SigSets = [np.load(bstr+si+SigStr) for si in Sets]
BkgSets = [np.load(bstr+si+BkgStr) for si in Sets]
ObsSets = [np.load(bstr+si+ObsStr) for si in Sets]

LEESetsF = np.concatenate(LEESets)
SigSetsF = np.concatenate(SigSets)
BkgSetsF = np.concatenate(BkgSets)
ObsSetsF = np.concatenate(ObsSets)

FCov = np.load(bstr+"MuBInclusive_FracCov_Square.npy")

SigTypes = ['nue', 'nue', 'numu', 'numu', 'numuPi0', 'numuPi0', 'NCPi0']
BEdges0 = [0.0 + 0.1*j for j in range(26)]
BEdges0.append(10.0)
Pi0BEdges0 = [0.0 + 0.1*j for j in range(11)]
Pi0BEdges0.append(10.0)
BEdges = [BEdges0, BEdges0, BEdges0, BEdges0, Pi0BEdges0, Pi0BEdges0, Pi0BEdges0]
LMBT = 0.4685 #Baseline length in kilometers

def ssqAvg(Emin, Emax, L, dmsq):
    if Emin == 0.0:
        Emin = 0.000001
    xmin, xmax = Emin/(1.267*dmsq*L), Emax/(1.267*dmsq*L)
    return 1.267*dmsq*L/(Emax-Emin)*((xmax*np.sin(1.0/xmax)**2 - sici(2.0/xmax)[0]) - (xmin*np.sin(1.0/xmin)**2 - sici(2.0/xmin)[0]))

def PeeAvg(Emin, Emax, L, dmsq, Ue4sq):
    return 1.0 - 4.0*Ue4sq*(1.0 - Ue4sq)*ssqAvg(Emin, Emax, L, dmsq)
def PmmAvg(Emin, Emax, L, dmsq, Um4sq):
    return 1.0 - 4.0*Um4sq*(1.0 - Um4sq)*ssqAvg(Emin, Emax, L, dmsq)
def PmeAvg(Emin, Emax, L, dmsq, Ue4sq, Um4sq):
    return 4.0*Ue4sq*Um4sq*ssqAvg(Emin, Emax, L, dmsq)
def PmsAvg(Emin, Emax, L, dmsq, Ue4sq, Um4sq):
    return 4.0*Um4sq*(1.0 - Ue4sq - Um4sq)*ssqAvg(Emin, Emax, L, dmsq)

def CNPStat(ni, mi):
    """Combined Neyman-Pearson Statistical Uncertainty
       Arguments:
          ni {int} -- Observation in bin i
          mi {float} -- Model Expectation in bin i
       Returns:
          [float] -- Statistical uncertainty of bin i
          See arXiv:1903.07185 for more details
    """
    if ni == 0.0:
        return mi/2.0
    else:
        return 3.0/(1.0/ni + 2.0/mi)

def muB_NoBkgOsc_Chi2(temp, constrained=False, Asimov=False):
    """Calculates the chi-squared from the full covariance matrix,
       Focuses only on the addition of a signal template above background, with no additional background oscillation

       "constrained" is an option of whether to apply the Covariance-Matrix-Constraint method on the nu_e CC fully-contained sample
       Default for our analyses will be "False"

       "Asimov" allows for calculation of Asimov-expected chi-squared, i.e. setting data to be equal to background expectation
    """
    CVStat = np.zeros(np.shape(FCov))
    CVSyst = np.zeros(np.shape(FCov))

    for ii in range(len(SigSetsF)):
        CVStat[ii][ii] = CNPStat(ObsSetsF[ii], SigSetsF[ii] + BkgSetsF[ii] + temp[ii])
        for jj in range(len(SigSetsF)):
            CVSyst[ii][jj] = FCov[ii][jj]*(SigSetsF[ii] + BkgSetsF[ii] + temp[ii] + 1.0e-2)*(SigSetsF[jj] + BkgSetsF[jj] + temp[jj] + 1.0e-2)
    CV = CVSyst + CVStat

    if constrained:
        CVYY = CV[26:,26:]
        CVXY = CV[:26,26:]
        CVYX = CV[26:,:26]
        CVXX = CV[:26,:26]

        if Asimov:
            nY = BkgSetsF[26:]+SigSetsF[26:]
        else:
            nY = ObsSetsF[26:]
        muY = BkgSetsF[26:]+SigSetsF[26:] + temp[26:]
        muX = BkgSetsF[:26]+SigSetsF[:26] + temp[:26]

        muXC = muX + np.dot(np.dot(CVXY, inv(CVYY)), nY-muY)
        CVXXc = CVXX - np.dot(np.dot(CVXY, inv(CVYY)), CVYX)

        if Asimov:
            nX = BkgSetsF[:26] + SigSetsF[:26]
        else:
            nX = ObsSetsF[:26]
        TS = np.dot(np.dot(nX[:25] - muXC[:25], inv(CVXXc[:25,:25])), nX[:25]-muXC[:25])
    else:
        if Asimov:
            nXY = BkgSetsF + SigSetsF
        else:
            nXY = ObsSetsF
        muXY = BkgSetsF + SigSetsF + temp

        TS = np.dot(np.dot(nXY - muXY, inv(CV)), nXY - muXY)

    return TS

def muB_OscChi2(Ue4sq, Um4sq, dm41, temp, constrained=False):
    """Calculates the chi-squared from the full covariance matrix,
       allowing for oscillated backgrounds (oscillating as a function of *reconstructed* neutrino energy)
       
       Still to be incorporated: templates "temp" that are adjusted based on Ue4sq, Um4sq, dm41

       "constrained" is an option of whether to apply the Covariance-Matrix-Constraint method on the nu_e CC fully-contained sample
       Default for our analyses will be "False"
    """
    CVStat = np.zeros(np.shape(FCov))
    CVSyst = np.zeros(np.shape(FCov))

    SSRW = []
    for SI in range(len(Sets)):
        ST = SigTypes[SI]
        BE = BEdges[SI]

        if ST == 'nue':
            RWVec = [PeeAvg(BE[kk], BE[kk+1], LMBT, dm41, Ue4sq) for kk in range(len(BE)-1)] 
        elif ST == 'numu':
            RWVec = [PmmAvg(BE[kk], BE[kk+1], LMBT, dm41, Um4sq) for kk in range(len(BE)-1)]
        elif ST == 'NCPi0' or ST == 'numuPi0':
            RWVec = [1.0 for kk in range(len(BE)-1)]

        SSRW.append(SigSets[SI]*RWVec)

    SSRWF = np.concatenate(SSRW)
    for ii in range(len(SigSetsF)):
        CVStat[ii][ii] = CNPStat(ObsSetsF[ii], SSRWF[ii] + BkgSetsF[ii] + temp[ii])
        for jj in range(len(SigSetsF)):
            CVSyst[ii][jj] = FCov[ii][jj]*(SSRWF[ii] + BkgSetsF[ii] + temp[ii] + 1.0e-2)*(SSRWF[jj] + BkgSetsF[jj] + temp[jj] + 1.0e-2)
    CV = CVSyst + CVStat

    if constrained:
        CVYY = CV[26:,26:]
        CVXY = CV[:26,26:]
        CVYX = CV[26:,:26]
        CVXX = CV[:26,:26]

        nY = ObsSetsF[26:]
        muY = BkgSetsF[26:]+SSRWF[26:] + temp[26:]
        muX = BkgSetsF[:26]+SSRWF[:26] + temp[:26]

        muXC = muX + np.dot(np.dot(CVXY, inv(CVYY)), nY-muY)
        CVXXc = CVXX - np.dot(np.dot(CVXY, inv(CVYY)), CVYX)

        nX = ObsSetsF[:26]
        TS = np.dot(np.dot(nX[:25] - muXC[:25], inv(CVXXc[:25,:25])), nX[:25]-muXC[:25])
    else:
        nXY = ObsSetsF
        muXY = BkgSetsF + SSRWF + temp

        TS = np.dot(np.dot(nXY - muXY, inv(CV)), nXY - muXY)

    return TS
