# gradient-decoding
Meta-analytic decoding of the cortical gradient of functional connectivity

## Summary
This repository contains all code required to reproduce the analyses and figures of the 
"Meta-analytic decoding of the cortical gradient of functional connectivity" paper.
See the preprint version for more details: https://doi.org/10.1101/2023.08.01.551505

## Workflow

The workflow consists of the following steps:

1. Functional Connectivity Gradients:
    * HCP S1200 resting-state fMRI data were used to generate functional connectivity and compute 
    the affinity matrix.
    * Diffusion map embedding was applied to identify the principal gradient of functional 
    connectivity.
2. Segmentation and Gradient Maps:
    * Whole-brain gradient maps were segmented to divide the gradient spectrum into a finite number 
    of brain maps.
    * Three different segmentation approaches were evaluated: percentile-based (PCT), k-means 
    (KMeans), and KDE. 
    * Individual segments were transformed into “activation” brain maps for decoding. 
    * The three segmentation approaches were evaluated using the silhouette score. 
3. Meta-analytic Functional Decoding:
    * Six different meta-analytic decoding strategies were implemented on surface space, derived 
    from three sets of meta-analytic maps (i.e., term-based (Term), LDA, and GCLDA) and two 
    databases (i.e., NS: [Neurosynth](https://github.com/neurosynth/neurosynth-data) 
    and NQ: [NeuroQuery](https://github.com/neuroquery/neuroquery_data)). 
4. Performance of Decoding Strategies:
    * The resultant 18 different decoding strategies were evaluated using four performance metrics,
    assessed by comparing correlation profiles, semantic similarity metrics (i.e., information 
    content (IC) and TFIDF), and signal-to-noise ratio (SNR). 
5. Visualization of the Decoded Maps:
    * Finally, we select the strategy with better performance across metrics for visualization, 
    where the non-functional terms were removed from the model.

![Fig-01](https://github.com/NBCLab/gradient-decoding/assets/52050407/a61f4998-23f8-4657-bf6c-00528775be06)

## How to use

### 1. Install dependencies

In order to execute the workflow (`workflow.py`), you will need to install all of the Python libraries 
that are required. The required library and associated versions are available in `requirements.txt`.

The easiest way to install the requirements is with Conda.

```python
conda create -p /path/to/gradientdec_env pip python=3.9
conda activate /path/to/gradientdec_env
pip install -r requirements.txt
```
### 2. Download data files

The analysis workflow is computationally intensive. If a user would like to skip any step, they will need to 
download the necessary files in `data` and `results` from our OSF page: https://osf.io/xzfrt/. 

### 3. Run the workflow

To run the workflow, create a project directory `PROJECT_DIR`, and execute the command:

```
python workflow.py --project_dir ${PROJECT_DIR} --n_cores 1
```

Alternatively, users can adapt and use our SLURM submission script: `./jobs/run_workflow.sh`.

```
sbatch ./jobs/run_workflow.sh
```

## Citation

If you use this code in your research, please acknowledge this work by citing the paper: https://doi.org/10.1101/2023.08.01.551505.

## Note

The code in this repository should be used only for reproducibility purposes of the linked paper.
In order to perform the proposed analysis in your data, please refer to the Python package 
[Gradec](https://github.com/JulioAPeraza/gradec).
