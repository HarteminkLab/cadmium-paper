
# Cadmium paper scripts

These scripts download, process, analyze and generate the figures for the "Integrating MNase-seq and RNA-seq time series data to study chromatin and transcription dynamics under cadmium stress" paper.

## Setup

Create conda environment to establish required libraries for the scripts.
```bash
conda env create --file cadmium_env.yml
```

Setup a `config.py` for the scripts. Using `config.example.py` as a starting point.

Download the [MEME-Suite](http://meme-suite.org/doc/download.html) and its
[motif database](http://meme-suite.org/meme-software/Databases/motifs/motif_databases.12.19.tgz).

Download and extract the [R64 - sacCer3 reference genome](http://sgd-archive.yeastgenome.org/sequence/S288C_reference/genome_releases/S288C_reference_genome_R64-1-1_20110203.tgz)

Make sure to configure your config.py with the correct paths:
```bash
FIMO_PATH = '/path/to/fimo'
FIMO_GENOME_FSA = "path/to/sacCer3/genome.fsa"
MACISAAC_MEME_PATH = 'path/to/macisacc_yeastdata/fimo/macisaac_yeast.v1.meme'
SACCER3_REFERENCE = 'path/to/extract/sacCer3/files/'
```

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
