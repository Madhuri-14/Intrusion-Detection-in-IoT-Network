from FeatureExtractor import *
import numpy as np
import time
import pandas as pd
import sys

# File location
# filename = benign-dec.pcap
# packet_size = 137396
# python process.py mirai1.pcap mirai2.pcap
if len(sys.argv) < 2:
    print("Need File name")
    exit(0)

number_of_files = len(sys.argv) - 1

for i in range(1, number_of_files+1):
    filename = sys.argv[i]
    file_path = "../Packets/" + filename.rstrip() #the pcap, pcapng, or tsv file to process.
    packet_limit = np.Inf #the number of packets to process
    result = list()

    FE_obj = FE(file_path,packet_limit)
    packet_size = len(FE_obj.scapyin)
    for i in range(packet_size):
        x = FE_obj.get_next_vector()
        result.append(list(x))

    df = pd.DataFrame(result, columns=FE_obj.nstat.getNetStatHeaders())
    file, pcap = filename.split('.')
    df.to_csv('../Data/' + file + '.csv', index=False)
