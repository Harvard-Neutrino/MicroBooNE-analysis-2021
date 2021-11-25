import numpy as np
from scipy.linalg import inv

bstr = "/Users/kjkelly/Documents/GitHub/uBvsmB/PyModules/InclusiveData/DataRelease/"
Sets = ["nueCC_FC_", "nueCC_PC_", "numuCC_FC_", "numuCC_PC_", "numuCCpi0_FC_", "numuCCpi0_PC_", "NCpi0_"]
LEEStr, SigStr, BkgStr, ObsStr = "LEE.npy", "Sig.npy", "Bkg.npy", "Obs.npy"
LEESetsF = np.concatenate([np.load(bstr+si+LEEStr) for si in Sets])
SigSetsF = np.concatenate([np.load(bstr+si+SigStr) for si in Sets])
BkgSetsF = np.concatenate([np.load(bstr+si+BkgStr) for si in Sets])
ObsSetsF = np.concatenate([np.load(bstr+si+ObsStr) for si in Sets])
FCov = np.load(bstr+"MuBInclusive_FracCov.npy")

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

def muBConstrainedChi2(temp):
    CVStat = np.zeros(np.shape(FCov))
    CVSyst = np.zeros(np.shape(FCov))

    for ii in range(len(SigSetsF)):
        CVStat[ii][ii] = CNPStat(ObsSetsF[ii], SigSetsF[ii] + BkgSetsF[ii] + temp[ii])
        for jj in range(len(SigSetsF)):
            CVSyst[ii][jj] = FCov[ii][jj]*np.sqrt((SigSetsF[ii] + BkgSetsF[ii] + temp[ii] + 1.0e-2)*(SigSetsF[jj] + BkgSetsF[jj] + temp[jj] + 1.0e-2))
    CV = CVSyst + CVStat

    CVYY = CV[26:,26:]
    CVXY = CV[:26,26:]
    CVYX = CV[26:,:26]
    CVXX = CV[:26,:26]

    nY = ObsSetsF[26:]
    muY = BkgSetsF[26:]+SigSetsF[26:] + temp[26:]
    muX = BkgSetsF[:26]+SigSetsF[:26] + temp[:26]

    muXC = muX + np.dot(np.dot(CVXY, inv(CVYY)), nY-muY)
    CVXXc = CVXX - np.dot(np.dot(CVXY, inv(CVYY)), CVYX)

    nX = ObsSetsF[:26]
    TS = np.dot(np.dot(nX[:25] - muXC[:25], inv(CVXXc[:25,:25])), nX[:25]-muXC[:25])
    return TS