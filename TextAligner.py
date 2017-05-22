import pymorphy2
import json
import io
import re
from operator import itemgetter
from collections import OrderedDict
from collections import defaultdict
names=["DENIS","FILEV","OSIPOV","VADIM"]

def cutter(name):
	arrayOfWords2= defaultdict(int)
	print("started")
	morph = pymorphy2.MorphAnalyzer()
	text=["DummuTextForFirstString"]
	with io.open(name+"cutted.txt",'r',encoding='utf8') as f:
		print("reading")
		for line in f:
			if (line[0]=="1" or line[0]=="0" ) and (len(line)>2):
				text.append(line)
			else:
				text[len(text)-1]=text[len(text)-1].rstrip()+line

	with io.open(name+"cutted.txt",'w',encoding='utf8') as f:
		print("writing")
		for w in text:
			f.write(w)

names=["DENIS","FILEV","OSIPOV","VADIM"]
for name in names:
	cutter(name)