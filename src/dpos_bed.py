
import sys
sys.path.append('.')

from config import *
import pandas as pd


def create_bed_for_dpos(mnase_data, save_path, filter_len=None):
    """
    Create bed file for dpos. Requires paired end reads as consecutively named reads ending in 1/2.
    Filtered to nucleosome length reads
    """
    
    # filter data by length selecting relevant columns
    if filter_len is not None:
        mnase_data = mnase_data[(mnase_data['length'] <= filter_len[1]) & 
                                          (mnase_data['length'] >= filter_len[0])]
    mnase_data = mnase_data.reset_index(drop=True).reset_index()[['chr', 'start', 'stop', 'index']]
    mnase_data = mnase_data.rename(columns={'index': 'name'})
    
    # set default score and designate + mate
    mnase_data['score'] = 255
    mnase_data['strand'] = '+'
    
    # name read
    mnase_data['name'] = 'read' + mnase_data['name'].astype(str)

    # create mate read
    mate = mnase_data.copy()
    mate['strand'] = '-'
    
    # name reads
    mnase_data['name'] = mate['name'] + '_1'
    mate['name'] = mate['name'] + '_2'

    #save to bed
    mnase_data = mnase_data.append(mate).sort_values('name').reset_index(drop=True)
    mnase_data.to_csv(save_path, sep='\t', header=False, index=False)


def main():

    from src.read_bam import read_mnase_bam

    print("Loading MNase-seq")
    filename = 'data/bam/mnase_seq/hm/DM504_sacCer3_m1_2020-05-20-18-48.bam'
    time = 0.0
    mnase_data = read_mnase_bam(filename, time, debug=False)
    print("Done.")

    print("Creating bed file...")
    save_path = "%s/bed/MNase_0_0_250.bed" % OUTPUT_DIR
    create_bed_for_dpos(mnase_data, save_path, filter_len=(0, 250))
    print("Done. Saved to %s" % save_path)


if __name__ == '__main__':
    main()