from bs4 import BeautifulSoup
import codecs
# 
names=["DENIS","FILEV","OSIPOV","VADIM"]

def cutter(name):
	tekst=""
	f = codecs.open(name+".txt", 'r', 'UTF-8')
	for line in f:
		tekst+=line
	f.close()
	print(len(tekst))
	vernut=""
	# with open("DENIS.html", 'rb') as f:
		# lines = [x.decode('utf8').strip() for x in f.readlines()]
	# for i in lines:
		# tekst += str(i) + "-"
	soup = BeautifulSoup(tekst, 'html.parser')
	element=soup.findAll("div", { "class" : "comment-renderer-text-content" })
	for hit in soup.findAll("div", { "class" : "comment-renderer-text-content" }):
		vernut+="1\t"+hit.get_text().lstrip()+"\r\n"
	#    vernut+=hit.get_text().lstrip()+"\r\n"
	with open(name+"cutted.txt", "wb") as fileF:
		fileF.write(vernut.encode("utf-8"))

for name in names:
	cutter(name)