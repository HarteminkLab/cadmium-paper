
import pandas

def lookup_sequence(chrom, start, end, genome):
    return genome[chrom][start:end]

def peaks_sequences(peaks_data, genome):
    """
    Determine the sequences at the given peaks. Append to input peaks data frame
    """

    data={'chr':[], 'start': [], 'end': [], 'sequence': []}

    for index, row in peaks_data.iterrows():

        chrom = row.chr
        start = int(row.start)
        end = int(row.end)

        data['chr'].append(chrom)
        data['start'].append(start)
        data['end'].append(end)
        data['sequence'].append(lookup_sequence(chrom, start, end, genome))

    data_frame = pandas.DataFrame(data=data)
    return data_frame

def create_fna(peaks_sequences, output_path, name_key=None):

    with open(output_path, 'wb') as output_file:
        for index, row in peaks_sequences.iterrows():
            
            chrom = row['chr']
            start = row['start']
            end = row['end']
            sequence = row['sequence']

            if name_key is None:
                name = ">chr{}:{}-{}_peak{}".format(chrom, start, end, index)
            else: name = ">{}".format(row[name_key])

            output_file.write(name + '\n')
            output_file.write(str(sequence) + '\n')