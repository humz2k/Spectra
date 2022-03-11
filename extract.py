from astroquery.sdss import SDSS
from astropy import coordinates as coords
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sep
import os
from darepype.drp.datafits import DataFits
from astropy.stats import mad_std
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
from astropy.table import Table
from astropy.utils.data import download_file

defaults = {}
try:
    with open("defaults.txt","r") as f:
        raw = f.read().splitlines()
except:
    print("Can't find defaults.txt")
    exit(1)

for i in raw:
    temp = i.split("=")
    if "(" in temp[1]:
        val = [float(j) for j in temp[1][1:-1].split(",")]
    else:
        try:
            val = float(temp[1])
        except:
            val = temp[1][1:-1]
    defaults[temp[0]] = val

def extract_sources(file,bw=defaults['bw'],bh=defaults['bh'],
                        fw=defaults['fw'],fh=defaults['fh'],
                        maskthresh=defaults['maskthresh'],
                        extract_thresh=defaults['extract_thresh'],
                        extract_err_str=defaults['extract_err_str'],
                        deblend_nthresh=defaults['deblend_nthresh'],
                        bright_factor=defaults['bright_factor'],
                        elim=defaults['elim'],
                        starmadfactor=defaults['starmadfactor'],
                        bright_lim=defaults['bright_lim'],
                        ps_lims=defaults['ps_lims'],
                        k2f_lims=defaults['k2f_lims'],

                        border_factor=defaults['border_factor'],
                        widthscale=defaults['widthscale'],
                        heightscale=defaults['heightscale'],

                        noisescale=defaults['noisescale'],
                        kfactor=defaults['kfactor']):

    df = DataFits()
    df.load(file)
    image = df.imageget() * 1.0

    rows, cols = image.shape[0], image.shape[1]

    imedian = np.nanmedian(image)
    istd = np.nanstd(image)
    imad = mad_std(image,ignore_nan=True)
    imin = np.nanmin(image)
    imax = np.nanmax(image)


    bkg = sep.Background(image,bw=bw, bh=bh, fw=fw, fh=fh)

    bkg_image = bkg.back()

    bkg_rms = bkg.rms()

    image_sub = image - bkg_image


    imsubmed = np.nanmedian(image_sub)
    imsubmad = mad_std(image_sub, ignore_nan = True)
    imsubmin, imsubmax = np.nanmin(image_sub), np.nanmax(image_sub)
    if extract_err_str == 'bkg.globalrms':
        extract_err = bkg.globalrms
    elif extract_err_str == 'bkg_rms':
        extract_err = bkg_rms
    elif extract_err_str == 'None':
        extract_err = None


    sources = sep.extract(image_sub, extract_thresh, err=extract_err, deblend_nthresh=deblend_nthresh)
    numsources = len(sources)
    cpeak_max = np.nanmax(sources['cpeak'])

    return image_sub,sources
    '''
    fig, ax = plt.subplots()
    m, s = np.mean(image_sub), np.std(image_sub)
    im = ax.imshow(image_sub, interpolation='nearest', cmap='gray',
                   vmin=m-s, vmax=m+s, origin='lower')

    for i in range(len(sources)):
        e = Ellipse(xy=(sources['x'][i], sources['y'][i]),
                width=6*sources['a'][i],
                height=6*sources['b'][i],
                angle=sources['theta'][i] * 180. / np.pi)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax.add_artist(e)
    '''
    return sources

def sources_to_world(file,sources=None):
    if not isinstance(sources, np.ndarray):
        sources = extract_sources(file)
    hdu = fits.open(file)[0]
    w = WCS(hdu.header)
    out = [w.pixel_to_world(sources['x'][i],sources['y'][i]) for i in range(len(sources))]
    return out

def get_spectra(world_coord):
    #print(world_coord)
    xid = SDSS.query_region(world_coord, spectro=True)
    return
    #sp = SDSS.get_spectra(matches=xid)

    #reference_star = sp[0]
    #print(reference_star)
    #x = 10.**reference_star[1].data['loglam']
    #y = reference_star[1].data['flux']

file = "m51_g-band_128.0s_bin1_220202_114527_al_seo_124_WCS.fits"

#image_data = fits.getdata(image_file, ext=0)

#df = DataFits()
#df.load(file)
#image = df.imageget() * 1.0

#image = download_file('http://data.astropy.org/tutorials/FITS-images/HorseHead.fits', cache=True )
#print(image)

image = file

df = DataFits()
df.load(image)
data = df.imageget() * 1.0
m, s = np.mean(data), np.std(data)
plt.imshow(data, interpolation='nearest', cmap='gray', vmin=m-s, vmax=m+s, origin='lower')
plt.colorbar()

data_sub,objects = extract_sources(image)

fig, ax = plt.subplots()
m, s = np.mean(data_sub), np.std(data_sub)
im = ax.imshow(data_sub, interpolation='nearest', cmap='gray',
               vmin=m-s, vmax=m+s, origin='lower')

# plot an ellipse for each object
for i in range(len(objects[0:40])):
    e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                width=6*objects['a'][i],
                height=6*objects['b'][i],
                angle=objects['theta'][i] * 180. / np.pi)
    e.set_facecolor('none')
    e.set_edgecolor('red')
    ax.add_artist(e)
world_coords = sources_to_world(file,objects)
print(world_coords[10])
spectra = get_spectra(world_coords[10])

print(spectra)

plt.show()
