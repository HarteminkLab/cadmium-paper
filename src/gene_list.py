
from src.met4 import all_genes

def get_gene_list():
    return [
            'MBP1', 'HSP26', 'PDC6', 'JLP1', 'CYS3', 'GRE1', 
            'RPS7A', 'RPL31B', 'CKB1', 'RPS7A','TAD2', 'PEX2',
            'MCD4', 'APJ1',
            'SUR4', 'UTR2', 'CTR1', 'TRM5', 'SRO9', # antisense activated sense repressed
            'ATG17', 'HOR2', 'PIC2', 'YLL056C', 'YBR241C', 'GUT2',  # sense and antisense activation
            ]

def get_paper_list():
    return ['HSP26', 'RPS7A', 'CKB1', 'MET32', 'MET31', 'PDC6', 'MCD4', 'YBR241C', 'UTR2']
