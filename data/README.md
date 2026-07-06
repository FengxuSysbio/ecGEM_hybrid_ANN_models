# Data

Place the input workbook in the repository root when running the scripts from
the repository root.

The scripts are configured to read:

```text
Supplementary Data 3.xlsx
```

Expected workbook structure:

- Sheet 1: benchmark model data, including `qP_mmol_L_h`
- Sheet 2: eciFX1172 core metabolic reactions and partial flux data, including `r_0013`

This `data/` directory is kept as a placeholder for documentation or small
example files. The current scripts look for the workbook in the current working
directory, so the recommended location is:

```text
ecGEM_hybrid_ANN_models/Supplementary Data 3.xlsx
```

The Excel workbook is not included in this GitHub-ready folder because raw data
files can be large and may require separate sharing or journal-specific handling.
