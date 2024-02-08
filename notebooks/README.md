# A numerical variability approach to results stability tests and its application to neuroimaging

## Paper

Arxiv preprint is available [here](https://arxiv.org/pdf/2307.01373.pdf).

## Minimal dataset to reproduce notebook images

The dataset is stored on [OSF](https://osf.io/), you'll need an OSF account to download the data.
Your username is your `Public Profile` which can be found under `My Profile`.
You can create a Token in `Settings > Personal access tokens`.

Install Python dependencies

```bash
pip install osfclient
```

To reproduce images from Notebook, run:

```bash
export OSF_USERNAME=<osf-username>
export OSF_TOKEN=<osf-token>
./download-minimal.sh
```

Jupyter notebook `paper.ipynb`

## Full reproduction

### Reproduce the fMRIprep executions

Create scripts to launch fMRIprep executions with

```bash
./scripts/make_script.sh
```

Execute `./scripts/launch_fmriprep_fuzzy_all.sh`

```bash
usage: ./launch_fmriprep_fuzzy_anat_only_all.sh --inputs --random-mode
         --inputs=file      JSON inputs file
         --samples=n        Number of samples to run
         --random-mode=mode Random mode (rr,rs,rr.rs,ieee)
```

### Reproduce the stability tests

Install Python dependencies

```bash
pip install osfclient
pip install git+https://github.com/yohanchatelain/stabilitest.git
```

Rerun executions or used precomputed results by downloading the full dataset:

```bash
export OSF_USERNAME=<osf-username>
export OSF_TOKEN=<osf-token>
./download-full.sh
```

Create configurations for the stability tests

```bash
cd numerical-variability-data
python3 ./scripts/create_config.py --inputs=inputs.json
```

### Run stability tests

Reproducing stability tests takes a while, in particular the Leave-One-Out tests (20h on 16 CPUs).
To reproduce the analysis, run:

```bash
./scripts/run_analysis.sh
```

Results are located into:

```bash
stability-test-results
├── loo
├── subjects
├── templates
└── versions
```

## Reproduce figures

Jupyter notebook `paper.ipynb`

