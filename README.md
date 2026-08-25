# mRMR for dendritic morphology

This tutorial uses 22 dendritic morphology features to find a compact,
complementary feature set for distinguishing Cortex, Thalamus, and CP neurons.
The main notebook implements count-based mutual information and greedy MID mRMR,
then compares feature rankings with visualization and classification.

## Run

From the repository root in an Anaconda Prompt or PowerShell:

```powershell
conda env create -f environment.yml
conda activate mrmr_demo
Expand-Archive -Path "data/subset_dendrite_resampled_2um.zip" -DestinationPath "data/subset_dendrite_resampled_2um"
python scripts/install_pymrmr.py
jupyter lab notebooks/mrmr_neuromorphology.ipynb
```

Alternatively, open `notebooks/mrmr_neuromorphology.ipynb` with the VS Code
Jupyter extension and select the `mrmr_demo` environment.

`pymrmr` requires a C/C++ compiler. The installation helper handles its legacy
Cython source and platform-specific compiler flags. The notebook uses the
included data and runs offline after setup.

## Notebook

The workflow:

1. loads 300 neurons and 22 Vaa3D morphology features;
2. discretizes continuous features and calculates mutual information;
3. implements MID mRMR and checks it against `pymrmr`;
4. compares relevance-only, mRMR, and all-feature classifiers;
5. explores sensitivity to discretization and alternative mRMR libraries.

## Data

- `data/metadata.csv`: 100 manually checked neurons from each region.
- `data/morphology_features.csv`: 22 Vaa3D global features from soma+dendrite
  morphologies resampled at 2 µm.
- `data/subset_dendrite_resampled_2um/`: the 300 prepared dendritic SWC files.

This educational subset comes from **SEU-A1876**, a collection of 1,876 complete
neuron morphologies registered to Allen CCFv3. The 2025 *Scientific Data*
descriptor accompanies the 2024 *Nature Communications* study.

- Dataset: [10.5281/zenodo.13944322](https://doi.org/10.5281/zenodo.13944322)
- Dataset paper: [10.1038/s41597-025-04379-0](https://doi.org/10.1038/s41597-025-04379-0)
- Dataset paper: [10.1038/s41467-024-54745-6](https://doi.org/10.1038/s41467-024-54745-6)

## Rebuild the prepared data

Download `Full_morphology_CCFv3.zip` from the dataset record, then run:

```powershell
python scripts/prepare_subset.py downloaded_data/metadata.xlsx --swc-archive downloaded_data/Full_morphology_CCFv3.zip --all-swc-dir downloaded_data/subset_swc
python scripts/prepare_compartments.py downloaded_data/subset_swc --vaa3d "D:/path/to/vaa3d" --workers 4
python scripts/extract_vaa3d_features.py downloaded_data/subset_dendrite_resampled_2um --vaa3d "D:/path/to/vaa3d" --workers 4 --output data/morphology_features.csv
```

The Vaa3D installation must include the `resample_swc` and
`global_neuron_feature` plugins. See the
[Vaa3D-x 1.1.4 release](https://github.com/Vaa3D/release/releases/tag/v1.1.4).

