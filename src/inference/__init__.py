"""Inference on a scan nobody has cached: the path the app takes.

Everything here REUSES the training code rather than restating it -- the volume
is prepared by `src.data.scan`, sites are located by `src.data.site_scoring`,
patches are cut by `site_dataset.patch_centre` / `cut_patch`, outputs are
converted by `targets.to_report_units`, and feasibility comes from
`targets.derived_feasible`. A second copy of any of those would be free to
drift, and the failure would look like a working model.
"""
