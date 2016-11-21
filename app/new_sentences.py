import numpy as np

from load_network import load_network
from layer import *
from network import * 

vocab = [u'all', u'set', u'just', u'show', u'being', u'money', u'over', u'both', u'years', u'four', u'through', u'during', u'go', u'still', u'children', u'before', u'police', u'office', u'million', u'also', u'less', u'had', u',', u'including', u'should', u'to', u'only', u'going', u'under', u'has', u'might', u'do', u'them', u'good', u'around', u'get', u'very', u'big', u'dr.', u'game', u'every', u'know', u'they', u'not', u'world', u'now', u'him', u'school', u'several', u'like', u'did', u'university', u'companies', u'these', u'she', u'team', u'found', u'where', u'right', u'says', u'people', u'house', u'national', u'some', u'back', u'see', u'street', u'are', u'year', u'home', u'best', u'out', u'even', u'what', u'said', u'for', u'federal', u'since', u'its', u'may', u'state', u'does', u'john', u'between', u'new', u';', u'three', u'public', u'?', u'be', u'we', u'after', u'business', u'never', u'use', u'here', u'york', u'members', u'percent', u'put', u'group', u'come', u'by', u'$', u'on', u'about', u'last', u'her', u'of', u'could', u'days', u'against', u'times', u'women', u'place', u'think', u'first', u'among', u'own', u'family', u'into', u'each', u'one', u'down', u'because', u'long', u'another', u'such', u'old', u'next', u'your', u'market', u'second', u'city', u'little', u'from', u'would', u'few', u'west', u'there', u'political', u'two', u'been', u'.', u'their', u'much', u'music', u'too', u'way', u'white', u':', u'was', u'war', u'today', u'more', u'ago', u'life', u'that', u'season', u'company', u'-', u'but', u'part', u'court', u'former', u'general', u'with', u'than', u'those', u'he', u'me', u'high', u'made', u'this', u'work', u'up', u'us', u'until', u'will', u'ms.', u'while', u'officials', u'can', u'were', u'country', u'my', u'called', u'and', u'program', u'have', u'then', u'is', u'it', u'an', u'states', u'case', u'say', u'his', u'at', u'want', u'in', u'any', u'as', u'if', u'united', u'end', u'no', u')', u'make', u'government', u'when', u'american', u'same', u'how', u'mr.', u'other', u'take', u'which', u'department', u'--', u'you', u'many', u'nt', u'day', u'week', u'play', u'used', u"'s", u'though', u'our', u'who', u'yesterday', u'director', u'most', u'president', u'law', u'man', u'a', u'night', u'off', u'center', u'i', u'well', u'or', u'without', u'so', u'time', u'five', u'the', u'left']

net = load_network()

def generate_sentence(number_of_words):

    input_vector = np.zeros((1,250*3))
    expansion_matrix = np.eye(250)

    input_words = np.array([np.random.randint(250,size=3)])
    words = [input_words[0,0],input_words[0,1],input_words[0,2]]

    # Convert input integers into vectors
    input_vector[:,0:250]   = expansion_matrix[input_words[:,0]]
    input_vector[:,250:500] = expansion_matrix[input_words[:,1]]
    input_vector[:,500:750] = expansion_matrix[input_words[:,2]]

    for i in range(number_of_words-3):
        next_word_vector = net.forward_prop(input_vector)

        # roll a number between 0 and 0.8
        roll = np.random.rand(1)[0]*0.8
        winner = 0.5

        chances = np.sort(next_word_vector[0])[::-1]
        for chance in chances:
            roll -= chance
            if roll <= 0:
                winner = chance
                break

        next_word = np.where(next_word_vector == winner)[1][0]

        words.append(next_word)

        input_vector[:,0:250] = input_vector[:,250:500]
        input_vector[:,250:500] = input_vector[:,500:750]
        input_vector[:,500:750] =  expansion_matrix[next_word]

    speech = ""
    for word in words:
        speech += vocab[word] + " "

    # Clean up a bit    
    sentences = speech.split("? ")
    sentences = [sentence[0].upper()+sentence[1:] for sentence in sentences if len(sentence) > 2]
    speech = ""
    for sentence in sentences:
        speech += sentence + "? "

    sentences = speech.split(". ")
    sentences = [sentence[0].upper()+sentence[1:] for sentence in sentences if len(sentence) > 2]
    speech = ""
    for sentence in sentences:
        speech += sentence + ". "

    speech = speech.replace(" :",":")
    speech = speech.replace(" ) ","")
    speech = speech.replace(" ( ","")
    speech = speech.replace(" '","'")
    speech = speech.replace(" ?","?")
    speech = speech.replace(" .",".")
    speech = speech.replace(" ,",",")
    speech = speech.replace(" nt "," not ")
    speech = speech.replace(" i "," I ")
    speech = speech.replace(" american "," American ")
    speech = speech.replace(" new york "," New York ")
    speech = speech.replace(" york "," New York ")

    return speech