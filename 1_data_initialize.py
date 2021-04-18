
from src.utils import mkdirs_safe
from src.utils import print_fl
from src.timer import Timer
from config import *

# global timer
timer = Timer()

rna_bam_rep1_dir = '%s/rna_seq/rep1' % BAM_DIR
rna_bam_rep2_dir = '%s/rna_seq/rep2' % BAM_DIR
mnase_bam_rep1_dir = '%s/mnase_seq/rep1' % BAM_DIR
mnase_bam_rep2_dir = '%s/mnase_seq/rep2' % BAM_DIR

def download_bam():
    
    from src.read_bam import sra_download_convert_bam
    # SRA IDs available at
    # https://www.ncbi.nlm.nih.gov/Traces/study/?acc=SRP269500&o=acc_s%3Aa

    mnase_rep1_files = [
        ("SRR12124866", "DM498_MNase_rep1_0_min"),
        ("SRR12124867", "DM499_MNase_rep1_7.5_min"),
        ("SRR12124868", "DM500_MNase_rep1_15_min"),
        ("SRR12124869", "DM501_MNase_rep1_30_min"),
        ("SRR12124870", "DM502_MNase_rep1_60_min"),
        ("SRR12124871", "DM503_MNase_rep1_120_min"),
    ]

    mnase_rep2_files = [
        ("SRR12124872", "DM504_MNase_rep2_0_min"),
        ("SRR12124873", "DM505_MNase_rep2_7.5_min"),
        ("SRR12124874", "DM506_MNase_rep2_15_min"),
        ("SRR12124875", "DM507_MNase_rep2_30_min"),
        ("SRR12124876", "DM508_MNase_rep2_60_min"),
        ("SRR12124877", "DM509_MNase_rep2_120_min"),
    ]

    rna_rep1_files = [
        ("SRR12124878", "RNA_rep1_0_min"),
        ("SRR12124879", "RNA_rep1_7.5_min"),
        ("SRR12124880", "RNA_rep1_15_min"),
        ("SRR12124881", "RNA_rep1_30_min"),
        ("SRR12124882", "RNA_rep1_60_min"),
        ("SRR12124883", "RNA_rep1_120_min")
    ]

    rna_rep2_files = [
        
        ('SRR13253046', "RNA_rep2_0_min"),
        ('SRR13253047', "RNA_rep2_7.5_min"),
        ('SRR13253048', "RNA_rep2_15_min"),
        ('SRR13253049', "RNA_rep2_30_min"),
        ('SRR13253050', "RNA_rep2_60_min"),
        ('SRR13253051', "RNA_rep2_120_min"),
    ]

    sra_id = mnase_rep1_files[0][0]
    filename = mnase_rep1_files[0][1]

    for sra_id, filename in mnase_rep1_files:
        sra_download_convert_bam(mnase_bam_rep1_dir, sra_id, filename)

    for sra_id, filename in mnase_rep2_files:
        sra_download_convert_bam(mnase_bam_rep2_dir, sra_id, filename)

    for sra_id, filename in rna_rep1_files:
        sra_download_convert_bam(rna_bam_rep1_dir, sra_id, filename)

    for sra_id, filename in rna_rep2_files:
        sra_download_convert_bam(rna_bam_rep2_dir, sra_id, filename)


def init_mnase_seq():

    from src.read_bam import read_mnase_set

    rep1_filenames = [
        "DM498_MNase_rep1_0_min.bam",
        "DM499_MNase_rep1_7.5_min.bam",
        "DM500_MNase_rep1_15_min.bam",
        "DM501_MNase_rep1_30_min.bam",
        "DM502_MNase_rep1_60_min.bam",
        "DM503_MNase_rep1_120_min.bam",
    ]

    rep2_filenames = [
        "DM504_MNase_rep2_0_min.bam",
        "DM505_MNase_rep2_7.5_min.bam",
        "DM506_MNase_rep2_15_min.bam",
        "DM507_MNase_rep2_30_min.bam",
        "DM508_MNase_rep2_60_min.bam",
        "DM509_MNase_rep2_120_min.bam",
    ]

    # Read Replicate 1 BAM files
    print_fl("Reading Rep1 MNase-seq BAM...", end='')
    rep1_mnase = read_mnase_set(mnase_bam_rep1_dir, rep1_filenames, 'dm498_503', 
        debug=DEBUG)
    print_fl("Done.")
    timer.print_time()
    print_fl()

    # Read Replicate 2 BAM files
    print_fl("Reading Rep2 MNase-seq BAM...", end='')
    rep2_mnase = read_mnase_set(mnase_bam_rep2_dir, rep2_filenames, 'dm504_509', 
        debug=DEBUG)
    print_fl("Done.")
    timer.print_time()
    print_fl()

    # depth of each dataset
    rep1_depth = rep1_mnase[['chr', 'time']].groupby('time').count()\
        .rename(columns={'chr':'count'})
    rep2_depth = rep2_mnase[['chr', 'time']].groupby('time').count()\
        .rename(columns={'chr':'count'})

    print_fl("Rep1 read depth:\n" + str(rep1_depth), end='\n\n')
    print_fl("Rep2 read depth:\n" + str(rep2_depth), end='\n\n')

    from src.chromatin import sample_mnase
    sample_rep1_depth = rep1_depth['count'].min()
    sample_rep2_depth = rep2_depth['count'].min()

    print_fl("Sampling Rep1 to %d for each time point..." % 
        sample_rep1_depth, end='')
    rep1_mnase_sampled = sample_mnase(rep1_mnase, sample_rep1_depth)
    print_fl("Done.")
    timer.print_time()
    print_fl()

    print_fl("Sampling Rep2 to %d for each time point..." % 
        sample_rep2_depth, end='')
    rep2_mnase_sampled = sample_mnase(rep2_mnase, sample_rep2_depth)
    print_fl("Done.")
    timer.print_time()
    print_fl()

    # Merge replicates
    print_fl("Merging MNase-seq files...", end='')
    merged_mnase_all = rep1_mnase.append(rep2_mnase)
    merged_mnase_all = merged_mnase_all[['chr', 'start', 
    'stop', 'length', 'mid', 'time', 'source']].sort_values(
        ['source', 'time', 'chr', 'start'])
    print_fl("Done.")
    timer.print_time()
    print_fl()

    # Merge sampled replicates
    print_fl("Merging MNase-seq files...", end='')
    merged_mnase_sampled = rep1_mnase_sampled.append(rep2_mnase_sampled)
    merged_mnase_sampled = merged_mnase_sampled[['chr', 'start', 
    'stop', 'length', 'mid', 'time', 'source']].sort_values(
        ['source', 'time', 'chr', 'start'])
    print_fl("Done.")
    timer.print_time()
    print_fl()

    # Save to disk
    save_path = '%s/mnase_seq_merged_all.h5.z' % mnase_dir
    print_fl("Saving merged MNase-seq to %s..." % save_path, end='')
    merged_mnase_all.to_hdf(save_path, 'mnase_data', mode='w', complevel=9,
        complib='zlib')
    print_fl("Done.")
    timer.print_time()
    print_fl()

    # Save to disk
    save_path = '%s/mnase_seq_merged_sampled.h5.z' % mnase_dir
    print_fl("Saving merged MNase-seq to %s..." % save_path, end='')
    merged_mnase_sampled.to_hdf(save_path, 'mnase_data', mode='w', complevel=9,
        complib='zlib')
    print_fl("Done.")
    timer.print_time()
    print_fl()

def init_rna_seq():

    rna_seq_rep1_filenames = [
        "DM538_RNA_rep1_0_min.bam",
        "DM539_RNA_rep1_7.5_min.bam",
        "DM540_RNA_rep1_15_min.bam",
        "DM541_RNA_rep1_30_min.bam",
        "DM542_RNA_rep1_60_min.bam",
        "DM543_RNA_rep1_120_min.bam"
    ]

    rna_seq_rep2_filenames = [
        "DM1450_RNA_rep2_0_min.bam",
        "DM1451_RNA_rep2_7.5_min.bam",
        "DM1452_RNA_rep2_15_min.bam",
        "DM1453_RNA_rep2_30_min.bam",
        "DM1454_RNA_rep2_60_min.bam",
        "DM1455_RNA_rep2_120_min.bam"
    ]

    from src.read_bam import read_rna_seq_set
    from src.transcription import sample_rna

    # Read replicate 1
    print_fl("Reading Replicate 1 RNA-seq BAM...", end='')
    rna_seq_rep1 = read_rna_seq_set(rna_bam_rep1_dir, rna_seq_rep1_filenames,
        source='dm538_dm543',
        debug=DEBUG)
    print_fl("Done.")
    timer.print_time()
    print_fl()

    # Read replicate 2
    print_fl("Reading Replicate 2 RNA-seq BAM...", end='')
    rna_seq_rep2 = read_rna_seq_set(rna_bam_rep2_dir, rna_seq_rep2_filenames,
        source='dm1450_dm1455',
        debug=DEBUG)
    print_fl("Done.")
    timer.print_time()
    print_fl()

    # depth of each dataset
    rep1_depth = rna_seq_rep1[['chr', 'time']].groupby('time').count()\
        .rename(columns={'chr':'count'})
    rep2_depth = rna_seq_rep2[['chr', 'time']].groupby('time').count()\
        .rename(columns={'chr':'count'})

    print_fl("Rep1 read depth:\n" + str(rep1_depth), end='\n\n')
    print_fl("Rep2 read depth:\n" + str(rep2_depth), end='\n\n')

    sample_rep1_depth = rep1_depth['count'].min()
    sample_rep2_depth = rep2_depth['count'].min()

    print_fl("Sampling Rep1 to %d for each time point..." % 
        sample_rep1_depth, end='')
    rep1_sampled = sample_rna(rna_seq_rep1, sample_rep1_depth)
    print_fl("Done.")
    timer.print_time()
    print_fl()

    print_fl("Sampling Rep2 to %d for each time point..." % 
        sample_rep2_depth, end='')
    rep2_sampled = sample_rna(rna_seq_rep2, sample_rep2_depth)
    print_fl("Done.")
    timer.print_time()
    print_fl()

    # Merge replicates
    print_fl("Merging RNA-seq files...", end='')
    merged_rna = rna_seq_rep1.append(rna_seq_rep2)
    merged_rna = merged_rna[['chr', 'start', 
    'stop', 'length', 'strand', 'time', 'source']].sort_values(
        ['source', 'time', 'chr', 'start'])
    print_fl("Done.")
    timer.print_time()
    print_fl()

    # Merge sampled replicates
    print_fl("Merging RNA-seq files...", end='')
    merged_rna_sampled = rep1_sampled.append(rep2_sampled)
    merged_rna_sampled = merged_rna_sampled[['chr', 'start', 
    'stop', 'length', 'strand', 'time', 'source']].sort_values(
        ['source', 'time', 'chr', 'start'])
    print_fl("Done.")
    timer.print_time()
    print_fl()

    # Save to all RNA-seq data to disk
    save_path = '%s/rnase_seq_all.h5.z' % rna_dir
    print_fl("Saving merged RNase-seq to %s..." % save_path, end='')
    merged_rna.to_hdf(save_path, 'rna_seq_data', mode='w', complevel=9,
        complib='zlib')

    # Save merged data to disk
    save_path = '%s/rnase_seq_merged_sampled.h5.z' % rna_dir
    print_fl("Saving merged RNase-seq to %s..." % save_path, end='')
    merged_rna_sampled.to_hdf(save_path, 'rna_seq_data', mode='w', complevel=9,
        complib='zlib')
    print_fl("Done.")
    timer.print_time()
    print_fl()

    # convert to pileup dataframe
    mkdirs_safe([pileup_chrom_dir])

    print_fl("Calculating RNA-seq pileup...", end='')
    from src.rna_seq_pileup import calculate_rna_seq_pileup
    pileup = calculate_rna_seq_pileup(merged_rna_sampled, timer)
    print_fl("Done.")
    timer.print_time()
    print_fl()

    save_path = pileup_path
    print_fl("Saving RNA-seq pileup to %s..." % save_path, end='')
    pileup.to_hdf(save_path, 'pileup', mode='w', complevel=9, complib='zlib')
    print_fl("Done.")
    timer.print_time()
    print_fl()


def main():
    """
    Initial BAM data download and conversion to dataframes

    1. Download RNA-seq and MNase-seq BAM files to disk
    2. Read RNA-seq BAM convert to dataframe and save
    3. Compute RNA-seq pileup and save
    4. Read MNase-seq BAM duplicates
    5. Merged, sample, and save

    Inputs:
    - output directory path
    - RNA-seq BAM files path
    - MNase-seq BAM files path

    Output:
    - RNA-seq data frame of reads
    - RNA-seq pileup data frame
    - MNase-seq merged data frame

    """

    print_fl("*******************************")
    print_fl("* 1      Initialization       *")
    print_fl("*******************************")

    # ------ Setup -------------

    print_preamble()


    # Make directories
    mkdirs_safe([
        rna_bam_rep1_dir, mnase_bam_rep2_dir, mnase_bam_rep1_dir,
        rna_dir, mnase_dir
        ])

    # ------- Download BAM files to disk ------------

    print_fl("\n------- Downloading dataset ----------\n")
    download_bam()

    print_fl("\n------- RNA-seq ----------\n")
    init_rna_seq()

    print_fl("\n------- MNase-seq ----------\n")
    init_mnase_seq()

    print_fl("Data initialization done. Time elapsed: %s" % timer.get_time())


if __name__ == '__main__':
    main()
