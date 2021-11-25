import numpy as np
from CERN_Inclusive_Analysis import muB_OscChi2
# Pointing to the right path of datafile
from pathlib import Path
local_dir = Path(__file__).parent

ldm41Min, ldm41Max, ndm41 = -2.0, 3.0, 200
dm41Vec = [10**(ldm41Min + (ldm41Max-ldm41Min)*j/ndm41) for j in range(ndm41+1)]

lUe4Min, lUe4Max, nUe4 = -4.0, 0.0, 80
Ue4Vec = [10**(lUe4Min + (lUe4Max-lUe4Min)*j/nUe4) for j in range(nUe4+1)]

Results = []
for jj in dm41Vec:
    print(jj)
    for kk in Ue4Vec:
        Results.append([jj, kk, muB_OscChi2(kk, 0.0, jj, np.zeros(137), constrained=False), muB_OscChi2(0.0, kk, jj, np.zeros(137), constrained=False), muB_OscChi2(kk, 0.0, jj, np.zeros(137), constrained=True), muB_OscChi2(0.0, kk, jj, np.zeros(137), constrained=True)])

    np.savetxt(str(local_dir) + "/NoAppSlice/DisappChi2_Full.dat", Results)
np.savetxt(str(local_dir) + "/NoAppSlice/DisappChi2_Full.dat", Results)
