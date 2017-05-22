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
	text=[]
	with io.open(name+"cutted.txt",'r',encoding='utf8') as f:
		for line in f:
			text.append(line)
	for currentComment in text:
		for w in currentComment.split():
			w= ''.join(ch for ch in w if ch.isalnum())
			word=morph.parse(w)[0]
			if {'VERB'} in word.tag or {'NOUN'} in word.tag or {'ADJF'} in word.tag or {'ADVB'} in word.tag:
				w=morph.parse(w)[0].normal_form
				arrayOfWords2[w] += 1
	arrayOfWords2= OrderedDict(sorted(arrayOfWords2.items(), key=itemgetter(1)))
	with io.open(name+"counted.txt",'w',encoding='utf8') as f:
		print("writing")
		for w in sorted(arrayOfWords2, key=arrayOfWords2.get, reverse=True):
			f.write(str(w))
			f.write("---")
			f.write(str(arrayOfWords2[w]))
			f.write("\n")

names=["DENIS","FILEV","OSIPOV","VADIM"]
for name in names:
	cutter(name)