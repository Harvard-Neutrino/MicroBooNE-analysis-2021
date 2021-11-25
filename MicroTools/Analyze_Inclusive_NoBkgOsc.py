'''
    This Python Script loads in a set of parameters {\Delta m_{41}^2, \sin^2(2\theta_{\mu e})} along with a set of MiniBooNE-excess-spectra for those parameters
    and then determines various MicroBooNE test statistics:
    a) Asimov Expectation when using the full Covariance Matrix
    b) Asimov Expectation using the Constrained Covariance Approach
    c) Data Result using the full Covariance Matrix
'''

import numpy as np
import ohioState as GoBlue
RHE = False
GBPC = GoBlue.MBtomuB(analysis='1eX_PC', remove_high_energy=RHE, unfold=True)
GBFC = GoBlue.MBtomuB(analysis='1eX', remove_high_energy=RHE, unfold=True)
from CERN_Inclusive_Analysis import muB_NoBkgOsc_Chi2
from pathlib import Path
local_dir = str(Path(__file__).parent)
import os

MiniBooNE_Signal_PANM = np.loadtxt(local_dir+"/MiniBooNETables/dm-sin-MB-events-table-less-points.dat")
Result = []
Pairs = []
for k0 in range(len(MiniBooNE_Signal_PANM)):
    if k0 % 100 == 0:
        print([k0, MiniBooNE_Signal_PANM[k0]])
    dm41, ssq2thmue = (MiniBooNE_Signal_PANM[k0])[0:2]
    p0 = [dm41, ssq2thmue]
    if p0 in Pairs:
        continue

    Pairs.append([dm41, ssq2thmue])
    MBSig = (MiniBooNE_Signal_PANM[k0])[2:]

    uBFC = GBFC.miniToMicro(MBSig)
    uBFC = np.insert(uBFC, 0, [0.0])
    uBFC = np.append(uBFC, 0.0)

    uBPC = GBPC.miniToMicro(MBSig)
    uBPC = np.insert(uBPC, 0, [0.0])
    uBPC = np.append(uBPC, 0.0)

    uBtemp = np.concatenate([uBFC, uBPC, np.zeros(85)])

    Result.append([dm41, ssq2thmue, muB_NoBkgOsc_Chi2(uBtemp, constrained=False, Asimov=True), muB_NoBkgOsc_Chi2(uBtemp, constrained=True, Asimov=True), muB_NoBkgOsc_Chi2(uBtemp, constrained=False, Asimov=False)])

np.savetxt(os.path.join(os.path.dirname(__file__), '../plots/Fig3/Inclusive_NoBkgOsc_Chi2.dat'), Result)
