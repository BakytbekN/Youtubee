import pandas as pd
import numpy as np
import io
allFiles = ["DENIScutted.txt","OSIPOVcutted.txt","VADIMcutted.txt"]
for name in allFiles:
	data_file=pd.read_csv(name, header=None, delimiter="\t", quoting=3)
	data_file.columns = ["Sentiment","Text"]
	print(name)
	print(data_file.shape[0])
	print( "positivnyh")
	print(np.count_nonzero(data_file.Sentiment))#positivnyh
	print( "negativnyh")
	print(len(data_file.Text)-np.count_nonzero(data_file.Sentiment))#negativnyh