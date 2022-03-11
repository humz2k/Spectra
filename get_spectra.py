from astroquery.sdss import SDSS
from astropy import coordinates as coords
import numpy as np
import matplotlib.pyplot as plt
import sep
import os
from darepype.drp.datafits import DataFits
from astropy.stats import mad_std
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename

pos = coords.SkyCoord('0h8m05.63s +14d50m23.3s', frame='icrs')

print(pos)

xid = SDSS.query_region(pos, spectro=True)

sp = SDSS.get_spectra(matches=xid)

reference_star = sp[0]

x = 10.**reference_star[1].data['loglam']
y = reference_star[1].data['flux']

plt.plot(x,y)
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.show()




#print(spectra.columns)
