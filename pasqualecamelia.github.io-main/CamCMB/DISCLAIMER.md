# Disclaimer

## Diagnostic chi^2, not official Planck likelihood

The chi^2_nu values computed by `run_qgt_gate3_camb.py` use diagonal
Planck error bars from the binned power spectra. This is a diagnostic
reduced chi^2, not the official Planck likelihood, which requires:
  - the full covariance matrix
  - foreground marginalisation
  - nuisance parameter treatment

The diagnostic chi^2_nu is appropriate for comparing the structural
distance between the QGT parameter vector and the Planck data. It is
not a substitute for a full likelihood analysis.

## PR4 reference spectra

The files `COM_PowerSpect_CMB-*-full_R4_PR4.txt` are CAMB-generated
reference spectra from the Tristram+2021 best-fit parameter vector.
They are not official Planck PR4/NPIPE binned spectra. In particular,
chi^2_nu(LCDM PR4) ≈ 0 by construction, because the comparison
curve is generated from the same best-fit parameters used to produce
the reference data. This value should not be interpreted as a fit to
actual satellite observations.

## No warranty

This software is provided "as is" without warranty of any kind. Results
are numerical outputs of a diagnostic pipeline and should be verified
independently before use in scientific publications.

## CAMB dependency

This package requires CAMB as an external dependency. The authors of
CamCMB are not responsible for CAMB behaviour, accuracy, or licence
compliance. Users are responsible for complying with the CAMB licence.
