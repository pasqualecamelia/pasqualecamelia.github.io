# Third-party notices

This package contains or uses third-party scientific resources.

## Planck Legacy Archive data

The files `COM_PowerSpect_CMB-TT-full_R3_01.txt`,
`COM_PowerSpect_CMB-EE-full_R3_01.txt`, and
`COM_PowerSpect_CMB-TE-full_R3_01.txt` are Planck 2018 PR3 CMB
power-spectrum data products from the Planck Legacy Archive.

These files are third-party public scientific data products and are not
licensed under the CamCMB code licence.

Please cite:
- Planck Collaboration (Akrami et al.), Planck 2018 results. V. CMB power
  spectra and likelihoods, A&A 641, A5 (2020).
  doi:10.1051/0004-6361/201936386
- Planck Collaboration (Aghanim et al.), Planck 2018 results. VI.
  Cosmological parameters, A&A 641, A6 (2020).
  doi:10.1051/0004-6361/201833910
- Planck Legacy Archive: https://pla.esac.esa.int/

## CAMB

CAMB is not redistributed in this package. The script
`run_qgt_gate3_camb.py` requires CAMB as an external dependency
(pip install camb). Users must install CAMB separately and comply with
the CAMB licence (LGPL with additional conditions).

Please cite:
- A. Lewis, A. Challinor and A. Lasenby, Efficient computation of CMB
  anisotropies in closed FRW models, ApJ 538, 473 (2000).
  doi:10.1086/309179
- CAMB website: https://camb.info/

## Planck PR4/NPIPE reference spectra

The files `COM_PowerSpect_CMB-TT/EE/TE-full_R4_PR4.txt` are
CAMB-generated reference spectra produced from the Tristram et al.
PR4/NPIPE best-fit parameter vector. They are not official PR4 binned
spectra and are not an official PR4 likelihood product. The definitive
PR4 comparison requires the public PLA products and the full covariance
pipeline.

Please cite:
- M. Tristram et al., Planck constraints on the primordial power spectrum
  with Planck PR4/NPIPE, A&A 647, A128 (2021).
  doi:10.1051/0004-6361/202039585
