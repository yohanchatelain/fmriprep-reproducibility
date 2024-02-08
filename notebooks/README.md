# A numerical variability approach to results stability tests and its application to neuroimaging

## Paper

Arxiv preprint is available [here](https://arxiv.org/pdf/2307.01373.pdf).

## Clone dataset

### OSF
The dataset is stored on [OSF](https://osf.io/), you'll need an OSF account to download the data.
Your username is your `Public Profile` which can be found under `My Profile`.
You can create a Token in `Settings > Personal access tokens`.

### Minimal dataset to reproduce notebook images

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

### Full reproduction

Install Python dependencies

```bash
pip install osfclient
pip install git+https://github.com/yohanchatelain/stabilitest.git
```

If you want to reproduce stability test results, download the full dataset using:

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

