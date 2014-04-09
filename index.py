from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords

from os import walk
import sys, json, getopt, math, string
sys.path.append("./ET/elementtree/")
import ElementTree as ET

#######################################################################
# Sample command:
#	python index.py -i D:\training\ -d dictionary.txt -p postings.txt
#######################################################################


#######################################################################
# Read filenames - Creating posting lists - Generate outputs
#######################################################################

def index(docs_dir, dict_file, posting_file):
	print 'start indexing'	

	filenames = []
	for (root, dirs, files) in walk(docs_dir):
		files = [ (x) for x in files ]
		filenames.extend(files)
	filenames.sort()

	posting_array_dict = generate_posting_array_dict(filenames)
	print '\nstart calculating document vector lengths'

	lengths_dict = generate_document_vector_length(filenames, posting_array_dict)
	print "\nstart processing output"

	generate_output(posting_array_dict, lengths_dict, dict_file, posting_file)

	print "\nmission completed :)"


def generate_output(dict, lengths, dict_file, posting_file):
	
	# init stdout
	curr = 0
	
	# init tokens list (sorted) and the list length
	sorted_keys = sorted(dict.iterkeys())
	total = len(sorted_keys)
	
	# init content for dictionary file
	dict_content = ""

	# init content for postings file
	# header: file lengths
	posting_header = json.dumps(lengths) + '\n'
	posting_f = open(posting_file, 'wb')
	posting_f.write(posting_header)
	
	# preparing content to be written into files
	for key in sorted_keys:
		posting_pos = write_posting(dict[key], posting_f)
		dict_line = "%(token)s %(length)s %(start)s %(end)s\n"%{'token':str(key), 'length':str(len(dict[key])), 'start':str(posting_pos[0]), 'end':str(posting_pos[1])}
		dict_content = dict_content + dict_line
		curr += 1
		sys.stdout.write("\rprocessing: " + ("%.2f" % (100.0 * curr / total)) + '%')
		sys.stdout.flush()

	# finished writing positing file 
	posting_f.close()

	# write into dictionary file
	dict_f = open(dict_file, 'wb')
	dict_f.write(str(dict_content))
	dict_f.close()

# build SkipList for every tokens from its posting array, build skips.
# pickled each SkipList into string and write into posting files. 
# return starting and ending position of the SkipList in the positng file

def write_posting(posting_dict, posting_f):
	posting_start = posting_f.tell()
	posting_f.write(json.dumps(posting_dict) + '\n')
	posting_end = posting_f.tell()
	return posting_start, posting_end

#######################################################################
# Generating Posting SkipLists
#######################################################################

# returns a dictionary
# key: tokens
# value: posting array
def generate_posting_array_dict(filenames):

	# init stdout 
	curr = 0

	# init empty posting array 
	posting_array_dict = {}
	
	# totol length of files to be processed 
	total = len(filenames)
	for filename in filenames:
		# genearate tokenized and stemmed (stopwords and numbers can be filtered by option) from a file
		try:
			title, desc = generate_tokens(str(filename))
			# store tokens and filnames into posting_array_dict
			# append filenames at end of the posting array if token key existed in the dictionary
			for t in title:
				key = str(t)
				if key in posting_array_dict:
					if filename in posting_array_dict[key]:
						posting_array_dict[key][filename]['title'] = posting_array_dict[key][filename]['title'] + 1
					if not filename in posting_array_dict[key]:
						posting_array_dict[key][filename] = {'title': 1, 'desc': 0}
				else:
					posting_array_dict[key] = {filename: {'title': 1, 'desc': 0}}

			for t in desc:
				key = str(t)
				if key in posting_array_dict:
					if filename in posting_array_dict[key]:
						posting_array_dict[key][filename]['desc'] = posting_array_dict[key][filename]['desc'] + 1
					if not filename in posting_array_dict[key]:
						posting_array_dict[key][filename] = {'title': 0, 'desc': 1}
				else:
					posting_array_dict[key] = {filename: {'title': 0, 'desc': 1}}

			curr += 1
			sys.stdout.write("\rindexing: " + ("%.2f" % (100.0 * curr / total)) + '%')
			sys.stdout.flush()
		except ET.ParseError:
			pass
	return posting_array_dict

def generate_document_vector_length(filenames, dict):
	# init stdout 
	curr = 0
	total = len(filenames)
	lengths_dict = {}

	for filename in filenames:
		try:
			axis = {}
			title, desc = generate_tokens(str(filename))
			for token in title:
				if token in axis:
					axis[token]['title'] += 1
				else:
					axis[token] = {'title': 1, 'desc': 0}

			for token in desc:
				if token in axis:
					axis[token]['desc'] += 1
				else:
					axis[token] = {'desc': 1, 'title': 0}
			sum_of_squared_title = 0
			sum_of_squared_desc = 0
			for token, freq in axis.iteritems():
				d_tf_title = float(1) + math.log(freq['title'], 10) if freq['title'] != 0 else 0
				d_tf_desc = float(1) + math.log(freq['desc'], 10) if freq['desc'] != 0 else 0
				d_idf = 1
				sum_of_squared_title += (d_tf_title * d_idf) ** 2
				sum_of_squared_desc += (d_tf_desc * d_idf) ** 2
			lengths_dict[filename] = {'title': math.sqrt(sum_of_squared_title), 'desc': math.sqrt(sum_of_squared_desc) }
			curr += 1
			sys.stdout.write("\rprocessing: " + ("%.2f" % (100.0 * curr / total)) + '%')
			sys.stdout.flush()
		except ET.ParseError:
			pass
	return lengths_dict

#######################################################################
# Tokenization and Stemming 
#######################################################################

# generate array of tokens from file
def generate_tokens(filename):
	title, desc = tokenize_sentences(filename)
	title_tokens = filter_stopwords(stemming_words(title))
	desc_tokens = filter_stopwords(stemming_words(desc))
	return (title_tokens, desc_tokens)

# tokenize sentences in a file into array of words
def tokenize_sentences(filename):
	file_dir = docs_dir + str(filename)
	f = open(file_dir, 'r')

	root = ET.parse(f).getroot()
	tags = root.getiterator('str')

	# read the relevant tags
	title_string = ''
	desc_string = ''
	for tag in tags:
		if tag.get('name')  == 'Title' :
			title_string = filter(lambda x: x in string.printable, tag.text.lower().strip())

		elif tag.get('name') == 'Abstract':
			desc_string = filter(lambda x: x in string.printable, tag.text.lower().strip().replace('relevant documents will describe', ''))

	f.close()

	sentences = sent_tokenize(title_string)
	title_words = []
	for s in sentences:
		title_words = title_words + word_tokenize(s)

	sentences = sent_tokenize(desc_string)
	desc_words = []
	for s in sentences:
		desc_words = desc_words + word_tokenize(s)

	
	return (title_words, desc_words)

stemmer = PorterStemmer()
def stemming_words(words):
	tokens = []
	for w in words:
		tokens.append(stemmer.stem(w))
	return tokens

stop = stemming_words(stopwords.words('english'))
def filter_stopwords(words):
	return [i for i in words if i not in stop]


#######################################################################
# Main
#######################################################################

def usage():
    print "usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file"

docs_dir = dict_file = posting_file = None
try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError, err:
    usage()
    sys.exit(2)
for o, a in opts:
    if o == '-i':
        docs_dir = a
    elif o == '-d':
        dict_file = a
    elif o == '-p':
        posting_file = a
    else:
        pass # no-op
if docs_dir == None or dict_file == None or posting_file == None:
    usage()
    sys.exit(2)

index(docs_dir, dict_file, posting_file)
