""" 
Authors: Ya'Kuana Davis, Roger Trejo
Course Name: CSCI 3725
Assignment Name: PQ4
Date: November 19, 2020
Description: This file contains all of the functions neccessary to train and run a simple Neural Network.
"""

import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import csv
import numpy as np
import time

"""TODO: add more here..."""
def get_raw_training_data(filename):
    """This function takes in the file, which would be the raw data and organizes it 
    to be in the format of a dictionary that has the artist as the key, and the sentences
    from the arist as the values in a list"""
    #training data variable
    training_data = {}

    #open file and read lines
    with open(filename, newline='') as file:
        raw_data = csv.reader(file)
        for row in raw_data:
            if training_data.get(row[0]) != None:
                training_data[row[0].lower()] += [row[1].lower()]
            else:
                training_data[row[0]] = [row[1].lower()]
                
    #return formatted training data 
    return training_data 


def organize_raw_training_data(raw_training_data, stemmer):
    """This function uses the raw_training_data to create a classes, a list of artits names,
    full_list_of_words, a list of words that each artist uses, and documents, 
    a list of tuples of all the words from the artist tupled with the artist name. """
    classes = []                # List of artists name 
    documents = []              # List of tuples (words[], artist_name)
    full_list_of_words = []     # List of lists of words 

    # Loop through raw_training_data dictionary
    for artist in raw_training_data:
        #append artists name
        classes.append(artist)
        #retrieve phrases from the arist
        phrases = raw_training_data.get(artist)
        #list of all the words from the artists 
        words_of_artist = []
        #loop through the phrases the from the artist
        for phrase in phrases:  
            # using nltk to tokenize each phrase
            new_words = nltk.word_tokenize(phrase)
            for word in new_words:
                #appending word after nltk tokenization
                words_of_artist.append(word)
        #add list of words of artist to list of all the words from the artists
        full_list_of_words.append(words_of_artist)
        
    #loop through arists and their words list
    for i in range(0, len(classes) - 1):
        #create (words, artist) tuple for documents
        document = tuple(full_list_of_words[i], classes[i])
        documents.append(document)
    #return list of words, documents, and classes   
    return full_list_of_words, documents, classes


def create_training_data(stems, classes, documents, stemmer):
    """Create the training data and output arrays using the stems, classes, and documents provided.""" 
    training_data = [] 
    output = [] 
    number_of_classes = len(classes)
    
    for artist_tuple in documents: 
        # Stores found words as 0's and 1's 
        artist_training_data = [] 
        
        for word in artist_tuple[0]: 
            # If the word is in stems add a one. Otherwise add a 0. 
            if word in stems: 
                artist_training_data.append(1)
            else: 
                artist_training_data.append(0)
        
        # Initialize the artist output list to have all 0s 
        artist_output = [0 for i in range(number_of_classes)]

        # Example: 
        # [0, 0, 0] 
        # ['jack', 'bob', 'tom']
        # artist_tuple[1] == 'tom'
        # at index i == 2 change 0  ->  1
        # [0, 0, 1]

        # Iterate through the artists and find the artist that said the sentence 
        for i, artist in classes: 
            if artist_tuple[1] == artist: 
                artist_output[i] = 1
        
        # Append final data to respective lists 
        training_data.append(artist_training_data)
        output.append(artist_output)

    return training_data, output  


def preprocess_words(words, stemmer):
    """Return a list of stems given an initial list of lists of words.""" 
    final_words_list = []
    
    # Iterate through every list of words 
    for words_list in words: 
        # Iterate through every word 
        for word in words_list:
            # Remove punctuation and special characters  
            word.strip(',!\'s\n!?.')
            # Append stemmed word to final words list 
            if (len(word) > 0 and final_words_list.count(word) == 0):
                final_words_list.append(stemmer.stem(word))

    # Ensure there are not any duplicates 
    return list(set(final_words_list))


def sigmoid(z):
    """Return the result of the sigmoid function."""
    sigmoid_result = 1 / (1 + np.exp(-z))
    return sigmoid_result 


def sigmoid_output_to_derivative(output):
    """Convert the sigmoid function's output to its derivative."""
    return output * (1-output)


"""* * * TRAINING * * *"""
def init_synapses(X, hidden_neurons, classes):
    """Initializes our synapses (using random values)."""
    # Ensures we have a "consistent" randomness for convenience.
    np.random.seed(1)

    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, len(classes))) - 1

    return synapse_0, synapse_1


def feedforward(X, synapse_0, synapse_1):
    """Feed forward through layers 0, 1, and 2."""
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0, synapse_0))
    layer_2 = sigmoid(np.dot(layer_1, synapse_1))
    return layer_0, layer_1, layer_2


def get_synapses(epochs, X, y, alpha, synapse_0, synapse_1):
    """Update our weights for each epoch."""
    # Initializations.
    last_mean_error = 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    synapse_0_direction_count = np.zeros_like(synapse_0)

    prev_synapse_1_weight_update = np.zeros_like(synapse_1)
    synapse_1_direction_count = np.zeros_like(synapse_1)

    # Make an iterator out of the number of epochs we requested.
    for j in iter(range(epochs+1)):
        layer_0, layer_1, layer_2 = feedforward(X, synapse_0, synapse_1)

        # How much did we miss the target value?
        layer_2_error = y - layer_2

        if (j% 10000) == 0 and j > 5000:
            # If this 10k iteration's error is greater than the last iteration,
            # break out.
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break

        # In what direction is the target value?  How much is the change for layer_2?
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # How much did each l1 value contribute to the l2 error (according to the weights)?
        # (Note: .T means transpose and can be accessed via numpy!)
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # In what direction is the target l1?  How much is the change for layer_1?
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

        # Manage updates.
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

        if j > 0:
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))

        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update

        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    return synapse_0, synapse_1


def save_synapses(filename, words, classes, synapse_0, synapse_1):
    """Save our weights as a JSON file for later use."""
    now = datetime.datetime.now()

    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
              }
    synapse_file = "synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print("Saved synapses to:", synapse_file)


def train(X, y, words, classes, hidden_neurons=10, alpha=1, epochs=50000):
    """Train using specified parameters."""
    print("Training with {0} neurons and alpha = {1}".format(hidden_neurons, alpha))

    synapse_0, synapse_1 = init_synapses(X, hidden_neurons, classes)

    # For each epoch, update our weights
    synapse_0, synapse_1 = get_synapses(epochs, X, y, alpha, synapse_0, synapse_1)

    # Save our work
    save_synapses("synapses.json", words, classes, synapse_0, synapse_1)


def start_training(words, classes, training_data, output):
    """Initialize training process and keep track of processing time."""
    start_time = time.time()
    X = np.array(training_data)
    y = np.array(output)

    train(X, y, words, classes, hidden_neurons=20, alpha=0.1, epochs=100000)

    elapsed_time = time.time() - start_time
    print("Processing time:", elapsed_time, "seconds")


"""* * * CLASSIFICATION * * *"""

def bow(sentence, words):
    """Return bag of words for a sentence."""
    stemmer = LancasterStemmer()

    # Break each sentence into tokens and stem each token.
    sentence_words = [stemmer.stem(word.lower()) for word in nltk.word_tokenize(sentence)]

    # Create the bag of words.
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
    return (np.array(bag))


def get_output_layer(words, sentence):
    """Open our saved weights from training and use them to predict based on
    our bag of words for the new sentence to classify."""

    # Load calculated weights.
    synapse_file = 'synapses.json'
    with open(synapse_file) as data_file:
        synapse = json.load(data_file)
        synapse_0 = np.asarray(synapse['synapse0'])
        synapse_1 = np.asarray(synapse['synapse1'])

    # Retrieve our bag of words for the sentence.
    x = bow(sentence.lower(), words)
    # This is our input layer (which is simply our bag of words for the sentence).
    l0 = x
    # Perform matrix multiplication of input and hidden layer.
    l1 = sigmoid(np.dot(l0, synapse_0))
    # Create the output layer.
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2


def classify(words, classes, sentence):
    """Classifies a sentence by examining known words and classes and loading our calculated weights (synapse values)."""
    error_threshold = 0.2
    results = get_output_layer(words, sentence)
    results = [[i,r] for i,r in enumerate(results) if r>error_threshold ]
    results.sort(key=lambda x: x[1], reverse=True)
    return_results =[[classes[r[0]],r[1]] for r in results]
    print("\nSentence to classify: {0}\nClassification: {1}".format(sentence, return_results))
    return return_results


def main():
    """TODO: more instructions here..."""
    stemmer = LancasterStemmer()

    raw_training_data = get_raw_training_data('dialogue_data.csv')
    words, classes, documents = organize_raw_training_data(raw_training_data, stemmer)
    stems = preprocess_words(words, stemmer)
    training_data, output = create_training_data(stems, classes, documents , stemmer)

    # Comment this out if you have already trained once and don't want to re-train.
    start_training(words, classes, training_data, output)

    # Classify new sentences.
    classify(words, classes, "will you look into the mirror?")
    classify(words, classes, "mithril, as light as a feather, and as hard as dragon scales.")
    classify(words, classes, "the thieves!")



if __name__ == "__main__":
    main()