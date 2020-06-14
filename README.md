
# Cadmium paper scripts

These scripts download, process, analyze and generate the figures for the "Integrating MNase-seq and RNA-seq time series data to study chromatin and transcription dynamics under cadmium stress" paper.

## Setup

Create conda environment to establish required libraries for the scripts.
```bash
conda env create --file cadmium_env.yml
```

Setup a `config.py` for the scripts. Use `config.example.py` as a starting point.

## Usage

Command-line usage:
Scripts can be run straight from the command-line, in which case set `USE_SLURM=False`
```bash
python 1_data_initialize.py
python 2_data_preprocessing.py
python 4_analysis.py
python 3_chrom_metrics.py
python 5_figures.py
```

Slurm submission:
Scripts can be run straight from the command-line, in which case set `USE_SLURM=True`. And the following variables need to be filled in inside `config.py`:

```python
USE_SLURM = True
SLURM_WORKING_DIR
CONDA_PATH
CONDA_ENV
```

Submit to slurm queue using sbatch:
```bash
sbatch -D </path/to/slurm/logs> \
    exports=SLURM_WORKING_DIR=</path/to/python/scripts/>,CONDA_PATH<path/to/conda.sh>=c,CONDA_ENV=<conda_env_name> \
    scripts/run_pipeline.sh
```
