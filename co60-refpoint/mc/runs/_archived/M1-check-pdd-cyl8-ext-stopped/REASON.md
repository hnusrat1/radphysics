# Stopped after 2 minutes (no batch finished), 2026-10-10 20:25 UTC

Superseded by `M1-check-boundary-free-R0`. This extension used the depth-scan configuration (importance 8 in the
column, ROI over the whole column) and would have bounded the boundary effect on D at R0 only to about 0.4%.
The replacement uses the main run's variance reduction with the depth scan restricted to 4.9-5.1 cm, which reaches
about 0.15% in similar time. The spec is kept in `mc/geant4/runspecs/M1-check-pdd-cyl8-ext.yaml`.
