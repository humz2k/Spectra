# TODO: Hook this up to get_spectra


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

## Arguments for sep.Background.
bw, bh, fw, fh, maskthresh = 16, 16, 3, 3, 0.0

## Arguments for sep.extract.
# extract_thresh, extract_err, extract_err_str = 50, bkg.globalrms, 'bkg.globalrms'
extract_thresh = 2.0
extract_err_str = 'bkg.globalrms' # 'bkg.globalrms','bkg_rms', 'None'
deblend_nthresh = 256
bright_factor = 10.0

## Parameters used for data partition.
elim, starmadfactor, bright_lim, ps_lims, k2f_lims = 1.5, 2.0, 1e6,  (1.0, 5.0), (0.8, 1.25)
border_factor = 1.33

## Parameters for scaling overlays and images.
widthscale, heightscale, noisescale = 10, 10, (1, 60)        # 6, 6, (1, 60)

## Other parameters.
kfactor = 2.5

datapath = 'm16'

files = [f for f in os.listdir(datapath) if '.fit' in f]

files = sorted(files)              # On my mac with OSX 10.14.3, this is needed to assure alphabetical order.
for i in range(len(files)):
    print( i, files[i])

whichfile = -1
fname = files[whichfile]
filename = os.path.join(datapath,fname)

df = DataFits()                       # Opens a DataFits io object.
df.load(filename)                        # Loads the fits file into memory.
image = df.imageget() * 1.0

rows, cols = image.shape[0], image.shape[1]
print('rows =',rows,'   cols =',cols)

rows, cols = image.shape[0], image.shape[1]
print('rows =',rows,'   cols =',cols)

## Calculate some useful statistical properties.
imedian = np.nanmedian(image)
istd = np.nanstd(image)                      # Standard deviation of the pixel intensities.
imad = mad_std(image,ignore_nan=True)        # Median absolute deviation. A more "robust" estimator.
imin = np.nanmin(image)                      # Minimum intensity.
imax = np.nanmax(image)                      # Maximum intensity.

print('median =', imedian,'  std =', istd,'  mad_std =', imad)
print('min =', imin,'  max =', imax)

bkg = sep.Background(image,bw=bw, bh=bh, fw=fw, fh=fh)

bkg_image = bkg.back()   ## Background image.

bkg_rms = bkg.rms()   ## Noise image.

image_sub = image - bkg_image
imsubmed = np.nanmedian(image_sub)
imsubmad = mad_std(image_sub, ignore_nan = True)
imsubmin, imsubmax = np.nanmin(image_sub), np.nanmax(image_sub)

print(imsubmed, imsubmad, imsubmin, imsubmax)
print('')
# Print the "global" mean and noise of the image background:
print('Global background =', bkg.globalback)
print('Global rms of the background =', bkg.globalrms)

if extract_err_str == 'bkg.globalrms':
    extract_err = bkg.globalrms
elif extract_err_str == 'bkg_rms':
    extract_err = bkg_rms
elif extract_err_str == 'None':
    extract_err = None


sources = sep.extract(image_sub, extract_thresh, err=extract_err, deblend_nthresh=deblend_nthresh)

print('extract_thresh =',extract_thresh,'  extract_err =', extract_err_str)
numsources = len(sources)
print('Number of sources =',numsources)
cpeak_max = np.nanmax(sources['cpeak'])
print('cpeak_max =', cpeak_max)

source = sources[3]
print(source)

f = get_pkg_data_filename(filename)

hdu = fits.open(f)[0]

w = WCS(hdu.header)
sky = w.world_to_pixel(500,500)


'''
pos = coords.SkyCoord('0h8m05.63s +14d50m23.3s', frame='icrs')

xid = SDSS.query_region(pos, spectro=True)

sp = SDSS.get_spectra(matches=xid)

reference_star = sp[0]

x = 10.**reference_star[1].data['loglam']
y = reference_star[1].data['flux']

plt.plot(x,y)
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.show()

'''


#print(spectra.columns)
