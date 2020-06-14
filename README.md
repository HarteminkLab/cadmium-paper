
# Cadmium paper scripts

These scripts download, process, analyze and generate the figures for the "Integrating MNase-seq and RNA-seq time series data to study chromatin and transcription dynamics under cadmium stress" paper.

## Setup

Create conda environment:
```bash
conda env create --file cadmium_env.yml
```

## Usage

Command-line usage:
```bash
python 1_data_initialize.py
python 2_data_preprocessing.py
python 4_analysis.py
python 3_chrom_metrics.py
python 5_figures.py
```

Slurm submission:
```bash
sbatch -D </path/to/slurm/logs> \
    exports=SLURM_WORKING_DIR=</path/to/python/scripts/>,CONDA_PATH<path/to/conda.sh>=c,CONDA_ENV=<conda_env_name> \
    scripts/run_pipeline.sh
```
