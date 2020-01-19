# coding: utf-8
"""I/O for MaNGA data.
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy import constants


class MaNGA_DAP:
    
    def __init__(self, filename):
        """MaNGA Data Analysis Pipeline (DAP) summary data. See

        - https://www.sdss.org/dr16/manga/manga-tutorials/dapall/
        - https://data.sdss.org/datamodel/files/MANGA_SPECTRO_ANALYSIS/DRPVER/DAPVER/dapall.html
        
        Parameters
        ----------
        filename: str
            Path to the DAP file.
        """
        self.dapall = Table.read(filename, 'DAPALL')
        
    def get_data(self, plateifu):
        """Return object data from DAP file.
        
        Parameters
        ----------
        plateifu: str
            MaNGA object ID in form PLATEID-IFUDESIGN
        
        Returns
        -------
        """
        selection = self.dapall['PLATEIFU']==plateifu
        
        # Do some checks for PLATEIFU. If multiple records are found, use the first.
        if not np.any(selection):
            raise ValueError('PLATEIFU {} not found in DAP file'.format(plateifu))
        if np.sum(selection) > 1:
            idx = np.argwhere(selection)
            selection[idx[1:]] = False
        return self.dapall[selection]


class MaNGA_HYB10:

    def __init__(self, filename):
        """MaNGA Data Analysis Pipeline pipeline output using HYB10 binning.
        For details see https://www.sdss.org/dr16/manga/manga-analysis-pipeline/.

        Parameters
        ----------
        filename: str
            Path to the HYB10 FITS file.
        """
        hdus = fits.open(filename)

        # Get plate+IFU IDs.
        self.plateifu = hdus['PRIMARY'].header['PLATEIFU']

        # Grab and store Ha velocity mask and maps.
        eline_vel_mask = hdus['EMLINE_GVEL_MASK'].data
        self.mask_vHa = eline_vel_mask[18]

        eline_vel = hdus['EMLINE_GVEL'].data
        self.vHa = np.ma.masked_array(eline_vel[18], self.mask_vHa > 0)

        eline_vel_ivar = hdus['EMLINE_GVEL_IVAR'].data
        self.ivar_vHa = np.ma.masked_array(eline_vel_ivar[18], self.mask_vHa > 0)

        self.wcs = WCS(hdus['EMLINE_GVEL'].header).celestial


class MaNGA_Pipe3D:

    def __init__(self, filename):
        """MaNGA Pipe3D value added catalog format. See

        https://www.sdss.org/dr14/manga/manga-data/manga-pipe3d-value-added-catalog/

        Parameters
        ----------
        filename: str
            Path to the Pipe3D FITS file.
        """
        hdus = fits.open(filename)
        org_hdr = hdus['ORG_HDR'].header

        # Get WCS and plate+IFU design.
        self.wcs = WCS(org_hdr).celestial
        self.plateifu = org_hdr['PLATEIFU']

        # Load SSP table, get stellar velocity map.
        ssp_hdr = hdus['SSP'].header
        ssp_dat = hdus['SSP'].data
        self.vStar = ssp_dat[13]
        self.dvStar = ssp_dat[14]

        # Load Ha velocity map.
        flux_hdr = hdus['FLUX_ELINES'].header
        flux_dat = hdus['FLUX_ELINES'].data
        self.vHa = flux_dat[102]
        self.dvHa = flux_dat[330]

        # Mask pixels.
        self.mask_vHa = np.logical_or(self.vStar == 0, self.vHa == 0)
        self.vHa = np.ma.masked_array(self.vHa, mask=self.mask_vHa)
        self.dvHa = np.ma.masked_array(self.dvHa, mask=self.mask_vHa)
