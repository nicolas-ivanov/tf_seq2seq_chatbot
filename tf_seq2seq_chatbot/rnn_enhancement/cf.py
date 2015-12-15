# -*- coding: utf-8 -*-
import numpy as np
import re, itertools, language_check, os, nltk, sys, traceback, h5py
from gensim.models import Phrases
import cPickle as pickle
from collections import defaultdict
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
import traceback


'''this is a list of common functions that may be useful, but they are not necessary. They are not well organized'''



'''beginning functions'''

def beginning_function():
	print
	print'------------------WELCOME TO TENSORFLOW, SEQ 2 SEQ LEARNING ----------------'
	print


def define_beginning_variables():
	iteration = 1
	number_of_epochs_completed = 0
	y_train_loss_line_list = []
	y_val_loss_line_list = []
	return (iteration, number_of_epochs_completed, y_train_loss_line_list, y_val_loss_line_list)

'''Other Useful Functions'''

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1



'''fitting model and processing predictions'''


def set_learning_rate(hist, learning_rate = 0, activate_halving_learning_rate = False, new_loss =0, past_loss = 0, counter = 0, save_model_dir='', iteration = 0):
	if activate_halving_learning_rate and (learning_rate>=0.000001):
	    if counter == 0:
	        new_loss = hist.history['loss'][0]
	        if new_loss>=(past_loss * 0.999): #you want at least a 0.5% loss decrease compared to the previous iteration
	            learning_rate = float(learning_rate)/float(3)
	            print 'you readjusted the learning rate to', learning_rate
	            with open('models/'+save_model_dir+'/'+'history_report.txt', 'a+') as history_file:
	                history_file.write('For next Iteration, Learning Rate has been reduced to '+str(learning_rate)+'\n\n')
    	            with open('history_reports/'+save_model_dir+'_'+'history_report.txt', 'a+') as history_file:
	                history_file.write('For next Iteration, Learning Rate has been reduced to '+str(learning_rate)+'\n\n')
    	            with open('history_reports/'+save_model_dir+'_'+'learning_rate_report.txt', 'a+') as history_file:
	                history_file.write('For Iteration: '+(str(iteration+1))+ ' Learning Rate has been reduced to '+str(learning_rate)+'\n\n')
	        	
	        past_loss = new_loss
        return (learning_rate, new_loss, past_loss)

def sample(a, temperature=1.0):

    a = np.log(a)/temperature
    a = np.exp(a)/np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1,a,1))

def convert_preds_to_integer(preds2d, temperature = 1.0):
	newlist = []
	for eachprobability_array in preds2d.tolist():
		newlist.append(sample(np.asarray(eachprobability_array)))
	return newlist


def sample(a, temperature=1.0):

    a = np.log(a)/temperature
    a = np.exp(a)/np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1,a,1))

def calculate_most_similar_word(vector, word2vecmodel):
	most_similar_word = word2vecmodel.most_similar_to_word_vec(vector, topn=1)[0]
	return most_similar_word[0] #get just the word

def vector_to_words(list_of_vectors, word2vecmodel):
	for eachvector in list_of_vectors:
		if np.linalg.norm(eachvector) == 0:
			pass
		else:
			sys.stdout.write(calculate_most_similar_word(eachvector, word2vecmodel))
			sys.stdout.write(' ')


def convert_predicted_vectors_to_actual_vectors(list_of_vectors, word2vecmodel):
	'''constructs 3d matrix for input into the predict function, zeros are left padded'''

	threedarrayforinput = np.zeros((1,len(list_of_vectors),len(list_of_vectors[0])))
	for i,eachvector in enumerate(list_of_vectors):
		if np.linalg.norm(eachvector) == 0:
			pass
		else:
			threedarrayforinput[0,i,:] = word2vecmodel[calculate_most_similar_word(eachvector, word2vecmodel)] #returns array of word
	return threedarrayforinput


def average_hidden_states(decoder_states):
	mean_decoder_states = np.mean(decoder_states)
	final_decoder_state = (0.5*decoder_states[-1])+0.5*mean_decoder_states




'''common functions'''

def create_directory(directory):
	if not os.path.exists(directory):
			print 'You created a new directory because it did not exist:', directory
        		os.mkdir(directory)


def ceil(n):
    return round(n+.5)

def intceil(number):
    return int(round(number+.5))



'''split sets of text'''




'''text manipulation'''
def tokenize_words(string_of_words):
	return nltk.word_tokenize(string_of_words)

def convert_sentence_tok_to_sentence_word_tok(quadgram_sentence_one_list):
	tok_sentence_tok_words = []
	for eachsentence in quadgram_sentence_one_list:
		wordlist =nltk.word_tokenize(eachsentence)
		tok_sentence_tok_words.append(wordlist)
	return tok_sentence_tok_words

def convert_to_one_list(quadgram_lol):
	return list(itertools.chain(*quadgram_lol))

#filter out unwanted wikipedia text
def bookfilter(text):
    text = unicode(text, errors='ignore')
    text = text.encode('ascii', errors='ignore')
    text = re.sub(r'(Further reading|Glossary|Glossaries|Systematics|Bibliography|Books|Dictionaries|External links|See also|Sources|References|Cited texts)\n\n(- .+\n|  .+\n)*', r'',text)
    text = re.sub(r'(Further reading|Glossary|Glossaries|Systematics|Bibliography|Books|Dictionaries|External links|See also|Sources|References|Cited texts)\n(- .+\n|  .+\n)*', r'',text)
    text = re.sub('(Footnotes|References|Notes)\n\n\[\d*\].(.+\n|.+\n)*', '',text)
    # text = re.sub(r'\[\d*\].(.+\n|.+\n)*', '',text)
    text = re.sub('Main article:.+\n', '',text)
    text = re.sub(r'(²|³|¹|⁴|⁵|⁶|⁷|⁸|⁹|⁰)', '',text)
    text = text.lower()
    # text = re.sub(r'[^\x00-\x7F]+', r'', text) #remove consecutive non-ascii chars
    text = re.sub('(     |    |   |  )', '',text)
    text = re.sub('(\n\n\n\n|\n\n\n\n\n|\n\n\n)', '\n\n',text)
    text = re.sub(r'\.([a-zA-Z])', r'. \1', text) #ensures space after every period (cuts down on word diversity)
    text = re.sub('`', '',text)
    text = re.sub('---','--',text)
    text = re.sub(r'(___|____|===)','',text)
    # text = re.sub(r'\n[0-9a-z_]{,10}\n', r'',text) 
    # text = re.sub(r'\n*\.\.\.\.+\n',r'',text

    text = re.sub('\n\n', ' NEWPARAGRAPH ',text)
    text = re.sub('\n', ' ',text) #this is for the indents
    
    text = re.sub('(all rights reserved|click here to order|buy now|money-back)','',text)
    text = re.sub(r'\n.+(\.{4,})(.+)\n','',text) #deletes all periods for indexes
    text = re.sub(r'([ a-zA-ZäöüßÄÖÜ-])+, ([ 0-9a-zA-ZäöüßÄÖÜ,-])*(\d)+\.*\n','',text)
    text = re.sub(r'\n(\.|\?|!|)+\n','\n',text) #remove lines with just periods
    # text = re.sub(r'\d{3,}\n','',text) #delete lines that have 2 or more digits in them 
    text = re.sub(r'(http://|http:|https:|https:/)*([a-zA-ZäöüßÄÖÜ-])+\. *([a-z]){2,4}','',text) #remove websites
    text = re.sub(r'\d*\.*.+(et\.|et\. al\.).+\n','',text) #remove sources
    text = re.sub(r'\n.*\d+:\d*.*\d*\.*','',text) #remove lines with indexes at end of soures.

    return text

def grammarchecklanguagecheck(text):
    tool = language_check.LanguageTool('en-US')
    matches = tool.check(text)
    return language_check.correct(text, matches).lower()

def runthroughgrammarchecklanguage(tokenized_sentences):
	start_time = time.time()
	print 'Running through grammar checker -- this will take a long time'
	tokenized_sentences_grammar_checked =[]
	progress_counter = 0 
	for eachsentence in tokenized_sentences:
	    tokenized_sentences_grammar_checked.append(grammarchecklanguagecheck(eachsentence))
	    progress_counter = progress_counter +1
	    if progress_counter % 7000 == 0:
	        print 'you are '+str(int((progress_counter*100)/len(tokenized_sentences)))+' percent complete'

	print("For Grammar Language Check It Took --- %s seconds ---" % (time.time() - start_time))
	return tokenized_sentences_grammar_checked


def grammarcheck(text):
    text = text.replace(' .','.')
    text = text.replace(' ?','?')
    text = text.replace(' !','!')
    text = text.replace(' )',')')
    text = text.replace('( ','(')
    text = text.replace(' ,',',')
    text = text.replace(" 's","'s")
    text = text.replace(", ,",",")
    text = text.replace(", ,",",")
    text = text.replace(' ,',',')
    text = text.replace('    ',' ')
    text = text.replace('   ',' ')
    text = text.replace('  ',' ')
    text = text.replace(',.','.')
    text = text.replace('?.','.')
    text = text.replace('!.','.')
    text = text.replace(', .','.')
    text = text.replace('? .','.')
    text = text.replace('! .','.')
    text = text.replace('.,','.')
    text = text.replace('.?','.')
    text = text.replace('.!','.')
    text = text.replace('. ,,','.')
    text = text.replace('. ?','.')
    text = text.replace('. !','.')
    return text

def convert_to_word_vectors_lol_input(quadgram_lol,modelvec):
	quadgram_lolol_numbers =[]
	for eachsentence in quadgram_lol:
		new_list_of_words = []
		for eachword in eachsentence:
		    new_list_of_words.append(modelvec[eachword])
		quadgram_lolol_numbers.append(new_list_of_words)
	return quadgram_lolol_numbers


def convert_to_word_vectors(quadgram_one_list, modelvec):
	quadgram_vectors_one_list = []
	for eachword in quadgram_one_list:
		quadgram_vectors_one_list.append(modelvec[eachword])
	return quadgram_vectors_one_list


def quad_gram_words(tokenized_sentences_tokenized_words, minimum_count_for_vectorization):
    print "performing bi gram"
    bigram = Phrases((tokenized_sentences_tokenized_words), min_count=minimum_count_for_vectorization, delimiter='_', threshold = 10)
    print "performing tri gram"
    trigram = Phrases((list(bigram[tokenized_sentences_tokenized_words])), min_count=minimum_count_for_vectorization, delimiter='_', threshold = 10)
    print "performing quad gram"
    quadgram = Phrases((list(trigram[list(bigram[tokenized_sentences_tokenized_words])])), min_count=minimum_count_for_vectorization, delimiter='_', threshold = 10)
    quadgramprocessed = (quadgram[list(trigram[list(bigram[tokenized_sentences_tokenized_words])])])
    return quadgramprocessed

def bi_gram_words(tokenized_sentences_tokenized_words, minimum_count_for_vectorization):
    print "performing bi gram"
    bigram = Phrases((tokenized_sentences_tokenized_words), min_count=minimum_count_for_vectorization, delimiter='_', threshold = 10)
    bigramprocessed = (bigram[tokenized_sentences_tokenized_words])
    return bigramprocessed







'''text classifier'''
def tokenize_newline_tokenize_words(text):
	tok_newline_word = []
	tokenized_newline = text.split('\n')
	for eachsentence in tokenized_newline:
    		tokenized_words_untouched = nltk.word_tokenize(text)
		tok_newline_word.append(tokenized_words_untouched)
	return tok_newline_word

def replace_new_lines(text):
	text = unicode(text, errors='replace')
	text = re.sub('(\n\n\n\n|\n\n\n\n\n|\n\n\n)', '\n\n',text)
	return text

def filter_for_positive_text_classifier(texttoread):
	text = texttoread.read()
	text = grammarcheck(text)
	text = replace_new_lines(text)
	text = re.sub('\n\n', ' NEWPARAGRAPH ',text)
	tok_newline_tok_word = tokenize_newline_tokenize_word(text)
	return tok_newline_tok_word

def filter_for_negative_text_classifier(texttoread):
	text = texttoread.read()
	text = grammarcheck(text)
	text = replace_new_lines(text)
	text = re.sub('\n\n', '',text)
	tok_newline_tok_word = tokenize_newline_tokenize_word(text)
	return tok_newline_tok_word






'''saving & loading material'''


def pickle_dump(file_to_save, variable_name = 'filename', save_model_dir = ''):
	 with open('models/'+save_model_dir+'/'+variable_name+'.p','wb+') as pkl_file:
		pickle.dump(file_to_save, pkl_file, pickle.HIGHEST_PROTOCOL)


def pickle_load(variable_name = 'filename', load_model_dir = ''):
	with open('models/'+load_model_dir+'/'+variable_name+'.p','rb') as pkl_file:
		return pickle.load(pkl_file)

def pickle_dump_matrix(file_to_save, variable_name = '', save_model_dir = '', counter = 0):
	with open('models/'+save_model_dir+'/'+variable_name+'_matrix_'+str(counter)+'.p','wb+') as pkl_file:
		pickle.dump(file_to_save, pkl_file, pickle.HIGHEST_PROTOCOL)

def pickle_load_matrix(variable_name = '', load_model_dir = '', counter = 0):
	with open('models/'+load_model_dir+'/'+variable_name+'_matrix_'+str(counter)+'.p','rb') as pkl_file:
		return pickle.load(pkl_file)

def hdf5_dump_matrix(file_to_save, variable_name= '', save_model_dir = '', counter = 0):
	h5f = h5py.File('models/'+save_model_dir+'/'+variable_name+'_matrix_'+str(counter)+'.h5', 'w')
	h5f.create_dataset('dataset_1', data=file_to_save)
	h5f.close()

def hdf5_load_matrix(variable_name= '', load_model_dir = '', counter = 0):
	h5f = h5py.File('models/'+load_model_dir+'/'+variable_name+'_matrix_'+str(counter)+'.h5', 'r')
	return h5f['dataset_1'][:] #potentially need add colon at the end. 
	h5f.close()

'''k means material'''

def make_set_of_words(text_lol):
	set_of_all_words = set()
	for eachsentence in text_lol:
		for eachword in eachsentence:
			set_of_all_words.add(eachword)
	return set_of_all_words

def int_squareroot(number):
	return int(round((number)**(0.50)))


def reduce_initial_cluster(number):
	return int(number*.80)

def create_word2vec_dict(vocabkeys, modelvec):
 	word2vec_dict={}
	for i in vocabkeys:
		try:
		    word2vec_dict[i]=modelvec[i]
		except:    
		    pass
	return word2vec_dict

def calculate_number_of_too_large_clusters(cluster_dict, max_cluster_size):
	above_cluster_limit = 0
	for eachkey in cluster_dict:
	# print len(cluster_dict[eachkey])
		if len(cluster_dict[eachkey])>max_cluster_size:
		    above_cluster_limit = above_cluster_limit +1
	print 'the number of clusters above the cluster limit is: '+str(above_cluster_limit)

def calculate_largest_number_for_matrix(cluster_dict, max_cluster_size):
	nb_clusters = len(cluster_dict)
	if nb_clusters >= max_cluster_size:
		return nb_clusters
	else:
		return max_cluster_size

def create_cluster_dict(y, clusters, word_freq):
    cluster_dict=defaultdict(list)
    for word,label in zip(y,clusters.labels_):
        cluster_dict[label].append( (word, word_freq[word], label) )
    return cluster_dict

def sort_by_word_freq(cluster_dict):
	for key,val in cluster_dict.iteritems():
		val.sort(key=lambda word_tup: -word_tup[1])
	return cluster_dict

def create_big_dict(cluster_dict):
	big_dict = {}
	for key,val in cluster_dict.iteritems():
	# val.sort(key=lambda word_tup: -word_tup[1])
		for i, foo in enumerate(val):
		    big_dict[foo[0]] = {
		        'word': foo[0],            
		        'cluster_id': foo[2],
		        'word_id': i,
		        'freq': foo[1]
		    } 
	return big_dict

def readjust_cluster_size(cluster_dict, max_cluster_size):

    key_number = len(cluster_dict)
    total_overflow_bins = 0
    for eachkey in range(key_number):
        cluster = cluster_dict[eachkey]  
        cluster_size = len(cluster)  
        if cluster_size > max_cluster_size:        
            overflow_bins = int(ceil((cluster_size - max_cluster_size)/max_cluster_size))
            # For each of the overflow bins...
            for i in range(overflow_bins):
                cluster_index = key_number + total_overflow_bins + i            
                # Fill it to max_cluster_size...
                for j in range(max_cluster_size):
                    # Or until the cluster has been reduced to max_cluster_size.
                    if len(cluster) == max_cluster_size:
                        break
                    else:
                        word_tuple = cluster.pop(max_cluster_size)
                        new_word_tuple = (word_tuple[0], word_tuple[1], cluster_index)
                        cluster_dict[cluster_index].append(new_word_tuple)
            total_overflow_bins += overflow_bins  
    return cluster_dict

def convert_text_to_dual_numbers(quadgram_one_list, big_dict):
	print 'len of full text:', len(quadgram_one_list)
	text_to_dual_numbers = []
	for eachword2 in quadgram_one_list:
		text_to_dual_numbers.append(big_dict[eachword2]['cluster_id'])
		text_to_dual_numbers.append(big_dict[eachword2]['word_id'])
	print 'len of full text coded with cluster id and word id', len(text_to_dual_numbers)
	return text_to_dual_numbers

def convert_lol_to_dual_numbers(set_of_list_of_lists, big_dict):
	lol_to_dual_numbers = []
	for eachsentence in set_of_list_of_lists:
		new_list = []
		for eachword in eachsentence:
			new_list.append(big_dict[eachword]['cluster_id'])
			new_list.append(big_dict[eachword]['word_id'])
		lol_to_dual_numbers.append(new_list)
	return lol_to_dual_numbers

def fixate_length_for_lol(lol_to_dual_numbers, y_sent_len, big_dict):
	lol_to_dual_numbers_fixed_len = []

	for eachsentence in lol_to_dual_numbers:
		list_of_words_in_dual_numbers = []
		if len(eachsentence) >= y_sent_len*2:
			list_of_words_in_dual_numbers = (eachsentence[:y_sent_len*2])
		elif len(eachsentence) < y_sent_len*2:
			remainder_of_eols = (y_sent_len*2 - len(eachsentence))/2
			list_of_words_in_dual_numbers = eachsentence
			for t in range(remainder_of_eols):
				list_of_words_in_dual_numbers.append(big_dict['EOL']['cluster_id'])
				list_of_words_in_dual_numbers.append(big_dict['EOL']['word_id'])
		lol_to_dual_numbers_fixed_len.append(list_of_words_in_dual_numbers)

	return lol_to_dual_numbers_fixed_len

'''prepare matrix lists'''
#if you specify a number in pop, it will pop from the left. If you don't, it will pop from the end!





def create_y_targets_list(text_to_dual_numbers, x_maxlen, y_maxlen, timestep = 1):
    y_targets_list = []
    for i in range(x_maxlen+1, len(text_to_dual_numbers) - y_maxlen, timestep):
        y_targets_list.append(text_to_dual_numbers[i + y_maxlen]) #this is just one word -- it will NOT BE A POS if i+maxlen remains odd. 
    print 'nb sequences generated for y target!:', len(y_targets_list) #keep this here
    return y_targets_list


def one_hot_2d_matrix(matrix2d):
	maxinteger = matrix2d.max()

	nb_samples, timesteps = matrix2d.shape
	new_3d_matrix = np.zeros((nb_samples, timesteps, maxinteger+1),dtype=np.bool)
	for i in range(nb_samples):
		for t in range(timesteps):
			new_3d_matrix[i, t, matrix2d[i,t]] = 1	
	return new_3d_matrix

def create_temporary_y_dual_numbers(counter, number_of_sequences_per_batch, text_to_dual_numbers):
	temporary_y_dual_numbers = []
        for p in range(number_of_sequences_per_batch+(y_maxlen/2)):
                

                temporary_y_dual_numbers.append(text_to_dual_numbers.pop(0))
                temporary_y_dual_numbers.append(text_to_dual_numbers.pop(0))

	return (temporary_y_dual_numbers, text_to_dual_numbers)


def create_y_list_of_lists(temporary_y_dual_numbers, y_maxlen):
	y_lol = []
	sequence_number_total= ((len(temporary_y_dual_numbers)/2)-(y_maxlen/2))
	for c in range(sequence_number_total): #this is the number of list with one word shifted over
		newlist = temporary_y_dual_numbers[0:y_maxlen]
		y_lol.append(newlist)
		temporary_y_dual_numbers.pop(0)
		temporary_y_dual_numbers.pop(0)
	return y_lol


def create_y_2d(y_lol, y_maxlen):
	y_2d = np.zeros((len(y_lol), y_maxlen), dtype=np.int32)
        for i,eachsequence in enumerate(y_lol):
            for t,eachnumber in enumerate(eachsequence):
                y_2d[i,t] = eachnumber
	return y_2d

def create_x_embed_list_of_lists(temporary_x_embed_vectors_lol, x_maxlen):
	x_embed_lolol = []
	sequence_number_total = len(temporary_x_embed_vectors_lol)-x_maxlen
	for c in range(sequence_number_total):
		newlist = temporary_x_embed_vectors_lol[0:x_maxlen]
		x_embed_lolol.append(newlist)
		temporary_x_embed_vectors_lol.pop(0)
	return x_embed_lolol

def appropriate_lolol_vector_list(quadgram_vectors_lolol, x_maxlen):
	vectors_lolol_maxlen_appropriated = []
	total_num_sequences = len(quadgram_vectors_lolol) +1 -x_maxlen

	for t in range(total_num_sequences):
		newsentlist = quadgram_vectors_lolol[:x_maxlen]
		mergedlist = [j for i in newsentlist for j in i]
		quadgram_vectors_lolol.pop(0)
		vectors_lolol_maxlen_appropriated.append(mergedlist)
	return vectors_lolol_maxlen_appropriated


def x_embed_build_array(x_embed, x_embed_lolol):
	for s,eachsequence in enumerate(x_embed_lolol):
            for w, eachword in enumerate(eachsequence): #reverse input here by adding [::-1], make sure you adjust processing below though!
                for a, eachvector in enumerate(eachword):
                    x_embed[s,w,a] = eachvector
	return x_embed

def x_embed_build_array_reverse(x_embed, x_embed_lolol):
	for s,eachsequence in enumerate(x_embed_lolol):
            for w, eachword in enumerate(eachsequence[::-1]): #you only reverse the list, not the sublist below by doing the [::-1]
                for a, eachvector in enumerate(eachword):
                    x_embed[s,w,a] = eachvector
	return x_embed


def x_embed_build_array_trim_sent(x_embed, x_embed_lolol, x_sent_len, x_maxlen = 1, word2vec_dimension = 16):
	
	total_num_sequences = len(x_embed_lolol) +1 -x_maxlen
	lolol_of_fixed_vector_size = []
	for s,eachsequence in enumerate(x_embed_lolol):
		if len(eachsequence)>=x_sent_len:
			eachsequence = eachsequence[:x_sent_len]

		else:
			num_of_eols = x_sent_len - len(eachsequence)
			for p in range(num_of_eols):
				eachsequence.append(np.zeros((word2vec_dimension),dtype=np.float32)) 
		# print 'pay attention to len of eachsequence', len(eachsequence)
		lolol_of_fixed_vector_size.append(eachsequence)

	print 'len of lolol_of_fixed_vector_size', len(lolol_of_fixed_vector_size)
#so now you have each sequence up to exactly x_maxlen
	vectors_lolol_maxlen_appropriated = []
	for t in range(total_num_sequences):
		newsentlist = lolol_of_fixed_vector_size[:x_maxlen]
		# print 'len of newsentlist is', len(newsentlist)
		mergedlist = [j for i in newsentlist for j in i]
		# print 'len of merged list is', len(mergedlist) #this should be 60
		lolol_of_fixed_vector_size.pop(0)
		vectors_lolol_maxlen_appropriated.append(mergedlist)
	print 'len of vectors_lolol_maxlen_appropriated', len(vectors_lolol_maxlen_appropriated)
#now build the matrix
	for s,eachsequence in enumerate(vectors_lolol_maxlen_appropriated):
		for w, eachword in enumerate(eachsequence): #reverse input here by adding [::-1], make sure you adjust processing below though!
    			for a, eachvector in enumerate(eachword):
    				# print 'newline'
    				# print s
    				# print w
    				# print a
				x_embed[s,w,a] = eachvector
	return x_embed


def appropriate_lolol_vector_list(quadgram_vectors_lolol, x_maxlen):
	vectors_lolol_maxlen_appropriated = []
	total_num_sequences = len(quadgram_vectors_lolol) +1 -x_maxlen

	for t in range(total_num_sequences):
		newsentlist = quadgram_vectors_lolol[:x_maxlen]
		mergedlist = [j for i in newsentlist for j in i]
		quadgram_vectors_lolol.pop(0)
		vectors_lolol_maxlen_appropriated.append(mergedlist)
	return vectors_lolol_maxlen_appropriated

def replace_zeros_with_number(x_embed, number = 5):
	x_embed[x_embed == 0.] = number
	return x_embed


'''prediction functions'''
def print_matrix_info(x_embed,y):
		
	print 'the size of x embed is   ', x_embed.nbytes
	print 'the size of y is         ', y.nbytes
	print 'the shape of x embed     ', x_embed.shape
	print 'the shape of y           ', y.shape

def update_history_info(save_model_dir, hist, iteration):
    with open('models/'+save_model_dir+'/'+'history_report.txt', 'a+') as history_file:
		history_file.write('Iteration number is: '+str(iteration)+'\n\n')
		history_file.write(str(hist.history))
		history_file.write('\n\n')
    with open('history_reports/'+save_model_dir+'_'+'history_report.txt', 'a+') as history_file:
		history_file.write('Iteration number is: '+str(iteration)+'\n\n')
		history_file.write(str(hist.history))
		history_file.write('\n\n')

def tf_update_history_info(name_of_plot, loss, val_loss, current_step, current_moving_average):
    with open('history_reports/'+name_of_plot+'_'+'rpt.txt', 'a+') as history_file:
		history_file.write('Step is: '+str(current_step)+'\n\n')
		history_file.write('Loss is: '+str(loss)+'     Val Loss is: '+ str(val_loss)+'   Moving Average: '+str(current_moving_average)+'\n\n\n')

def dual_numbers_to_words(generated, cluster_dict):
	list_of_words = []
    	for index,eachword3 in enumerate(generated[:-1]):
                if index % 2 == 0:
                	try:	
                		word_created = (cluster_dict[eachword3][generated[index+1]])[0]
				list_of_words.append(word_created)
				sys.stdout.write(word_created)
				sys.stdout.write(' ')
        		except IndexError:
        			sys.stdout.write('NO EXIST ')
        			list_of_words.append((cluster_dict[0][0])[0])
	
        sys.stdout.flush
    	return list_of_words

def vector_list_for_list_of_words(list_of_words, modelvec):
	vector_list = []
	for eachword in list_of_words:
			vector_list.append(modelvec[eachword]) #all words should be modelvec compatible at this point
	return vector_list

def vector_list_for_list_of_words_EOL(list_of_words, modelvec, word2vec_dimension):
	vector_list = []
	for eachword in list_of_words:
		if eachword == 'EOL':
			vector_list.append(np.zeros((word2vec_dimension),dtype=np.float32))
		else:
			vector_list.append(modelvec[eachword]) #all words should be modelvec compatible at this point
	return vector_list

def convert_3d_matrix_to_list(x_embed_sample):
	return x_embed_sample[0].tolist()

def join_new_words_to_x_embed_sample(x_embed_sample, new_vector_list, y_maxlen, word2vec_dimension):
	x_embed_sample_list = convert_3d_matrix_to_list(x_embed_sample)
	for p in range(y_maxlen/2):
		x_embed_sample_list.pop(0)
		#you did not add these lists on correctly
	x_embed_sample_list.extend(new_vector_list)
	x_embed_sample = np.zeros((1,len(x_embed_sample_list),word2vec_dimension), dtype = np.float32)
	for i,eachword in enumerate(x_embed_sample_list):
		for t,eachvector in enumerate(eachword):
			x_embed_sample[0,i,t] = eachvector
	return x_embed_sample
 	
def join_new_words_to_x_embed_sample_reverse(x_embed_sample, new_vector_list, y_maxlen, word2vec_dimension):
	x_embed_sample_list = convert_3d_matrix_to_list(x_embed_sample)
	for p in range(y_maxlen/2):
		x_embed_sample_list.pop(0)
		#you did not add these lists on correctly
	x_embed_sample_list.extend(new_vector_list)
	x_embed_sample = np.zeros((1,len(x_embed_sample_list),word2vec_dimension), dtype = np.float32)
	for i,eachword in enumerate(x_embed_sample_list[::-1]): #right here, you edit it to make it reverse!
		for t,eachvector in enumerate(eachword):
			x_embed_sample[0,i,t] = eachvector
	return x_embed_sample

def join_new_words_to_x_embed_sample_sentence_level(x_embed_sample, new_vector_list, y_maxlen, word2vec_dimension, x_sent_len = 30, x_maxlen = 1):
	x_embed_sample_list = convert_3d_matrix_to_list(x_embed_sample)
	for p in range(x_sent_len*y_maxlen):
		x_embed_sample_list.pop(0)
		#you did not add these lists on correctly
	x_embed_sample_list.extend(new_vector_list)
	x_embed_sample = np.zeros((1,len(x_embed_sample_list),word2vec_dimension), dtype = np.float32)
	for i,eachword in enumerate(x_embed_sample_list):
		for t,eachvector in enumerate(eachword):
			x_embed_sample[0,i,t] = eachvector
	return x_embed_sample


def sample_with_temperature_axis_one(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1), axis=1)


def sample_with_temperature(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def sample_with_temperature_divide_100(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = a/100
    print a
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


'''plotting functions'''

def update_graph_data(hist, number_of_epochs_completed, y_train_loss_line_list, y_val_loss_line_list):
	number_of_epochs_completed = number_of_epochs_completed + 1
	y_train_loss_line_list.append(hist.history['loss'][0])
	y_val_loss_line_list.append(hist.history['val_loss'][0])
	return (number_of_epochs_completed, y_train_loss_line_list, y_val_loss_line_list)


def create_plot(number_of_epochs_completed, y_train_loss_line_list, y_val_loss_line_list, name_of_plot):
	try:
		x_axis = np.linspace(1,number_of_epochs_completed, number_of_epochs_completed)
		y_train_loss_line = np.asarray(y_train_loss_line_list)
		y_val_loss_line = np.asarray(y_val_loss_line_list)


		tracetrainloss = go.Scatter(
			x = x_axis,
			y = y_train_loss_line,
			name = 'Training Loss',
			line = dict(
				color = ('rgb(205, 12, 24)'),
				width = 3)
		)

		tracevalloss = go.Scatter(
			x = x_axis,
			y = y_val_loss_line,
			name = 'Validation Loss',
			line = dict(
				color = ('rgb(22, 96, 167)'),
				width = 3)
		)


		data = [tracetrainloss, tracevalloss]
		py.plot(data, filename=name_of_plot , auto_open=False)
	except:
		traceback.print_exc()
		pass


def moving_average_past_five(y_val_loss_line_list):
	return np.mean(y_val_loss_line_list[-5:])


#color picker here: http://www.colorpicker.com/c390d4
def create_plot_by_step(number_of_epochs_completed, y_train_loss_line_list, y_val_loss_line_list, name_of_plot, step_size, batch_size, y_learning_rate_list, y_val_loss_moving_average_list, y_val_loss_line_list_25,
            y_val_loss_line_list_75,
            y_val_loss_line_list_100,
            train_size,
            starting_learning_rate):
	try:
		x_axis = np.linspace(step_size,number_of_epochs_completed, len(y_train_loss_line_list))
		y_train_loss_line = np.asarray(y_train_loss_line_list)
		y_val_loss_line = np.asarray(y_val_loss_line_list)


		tracetrainloss = go.Scatter(
			x = x_axis,
			y = y_train_loss_line,
			name = 'Training Loss',
			line = dict(
				color = ('rgb(205, 12, 24)'), #red
				width = 5)
		)

		tracevalloss = go.Scatter(
			x = x_axis,
			y = y_val_loss_line,
			name = 'Validation Loss (5,10)',
			line = dict(
				color = ('rgb(22, 96, 167)'), #blue
				width = 2)
		)
		tracevalloss25 = go.Scatter(
			x = x_axis,
			y = y_val_loss_line_list_25,
			name = 'Validation Loss (13,18)',
			line = dict(
				color = ('rgb(96, 246, 236)'), #cyan
				width = 2)
		)

		tracevalloss75 = go.Scatter(
			x = x_axis,
			y = y_val_loss_line_list_75,
			name = 'Validation Loss (25,30)',
			line = dict(
				color = ('rgb(207, 194, 12)'), #yellow
				width = 2)
		)

		tracevalloss100 = go.Scatter(
			x = x_axis,
			y = y_val_loss_line_list_100,
			name = 'Validation Loss (60,60)',
			line = dict(
				color = ('rgb(192, 192, 192)'), #Grey
				width = 2)
		)


		tracelearningrate = go.Scatter(
			x = x_axis,
			y = y_learning_rate_list,
			name = 'Learning Rate (Jump = LR Decrease)',
			line = dict(
				color = ('rgb(180, 41, 227)'), #purple
				width = 3,
				shape = 'hv')
		)

		tracevallossmovingaverage = go.Scatter(
			x = x_axis,
			y = y_val_loss_moving_average_list,
			name = 'Val Loss (13,18) Moving Average (Past 5)',
			line = dict(
				color = ('rgb(32, 173, 10)'), #green
				width = 5)
		)



		
		layout = dict(title = name_of_plot,
		              xaxis = dict(title = 'Steps (Batch Size = '+str(batch_size)+')<br>'+'Epoch = '+str(int(train_size/batch_size))+' Steps. Train Size = '+str(int(train_size))+'<br>'+'Starting LR = '+str(starting_learning_rate)),
		              yaxis = dict(title = 'Loss (Goal = 1.6)'),
		              )

		data = [tracetrainloss, tracevalloss, tracevalloss25, tracevalloss75, tracevalloss100, tracevallossmovingaverage, tracelearningrate]
		fig = dict(data=data, layout=layout)
		py.plot(fig, filename=name_of_plot , auto_open=False)
	except:
		traceback.print_exc()
		pass







'''model construction'''

def create_bd_lstms():
	lstmenc = LSTM(output_dim=hidden_variables_encoding/2, return_sequences = True)
	gruenc = GRU(output_dim=hidden_variables_encoding/2, return_sequences = True) 
	lstmenc2 = LSTM(output_dim=hidden_variables_encoding/2, return_sequences = False)
	gruenc2 = GRU(output_dim=hidden_variables_encoding/2, return_sequences = False) 
	lstmdec = LSTM(output_dim=hidden_variables_decoding/2, return_sequences = True)
	grudec = GRU(output_dim=hidden_variables_decoding/2, return_sequences = True) 
	lstmdec2 = LSTM(output_dim=hidden_variables_decoding/2, return_sequences = True)
	grudec2 = GRU(output_dim=hidden_variables_decoding/2, return_sequences = True) 

def create_readout_gru():
	print 'constructing readout GRU'
	readout = Sequential()
	readout.add(Dense(y_matrix_axis, input_shape = (hidden_variables_decoding,), activation='softmax'))
	gru_wr = GRUwithReadout(readout, return_sequences = True)