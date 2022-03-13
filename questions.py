import nltk
import sys
import os
import string
from math import log

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """

    text_dict = {}

    for file in os.listdir(directory):
        with open(os.path.join(directory, file)) as f:
            file_text = f.read()
        text_dict[file] = file_text

    return text_dict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    # Use nltk's tokenizer to convert sentence into words
    tokens = nltk.word_tokenize(document)
    # Process tokens to remove punctuation and stopwords
    processed_tokens = []
    for token in tokens:
        if token.lower() in string.punctuation:
            continue
        if token.lower() in nltk.corpus.stopwords.words("english"):
            continue
        processed_tokens.append(token.lower())

    return processed_tokens


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    # Set up idf calculation
    document_count = len(documents)
    word_bank = {}

    # Loop through each file and word in file to get count of each word
    for file in documents:
        # Create 'set' of unique words from document
        document_words = set(documents[file])
        for word in document_words:
            # Tally occurences of each word in the word_bank dictionary
            word_bank[word] = word_bank[word] + 1 if word in word_bank else 1

    # Loop through each word to calculate inverse document frequency
    idf_bank = {
        word: log(document_count / word_bank[word])
        for word in word_bank
    }

    return idf_bank


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """

    top_files = {}

    # Calculate tfidf in each document for words in query
    for file in files:
        top_files[file] = {}
        for word in files[file]:
            if word in query:
                if word in top_files[file]:
                    top_files[file][word]['tf'] += 1
                else:
                    top_files[file][word] = {'tf': 1, 'idf': idfs[word]}

        # Combine tf and idf into tfidf
        for word in top_files[file]:
            top_files[file][word]['tfidf'] = (top_files[file][word]['tf'] *
                                              top_files[file][word]['idf'])

    # Loop through each file summing idf values of words in the query
    for file in files:
        file_score = 0
        for word in top_files[file]:
            file_score += top_files[file][word]['tfidf']
        top_files[file]['file_score'] = file_score

    top_files = sorted(top_files.keys(),
        key=lambda x: top_files[x]['file_score'], reverse=True)

    return top_files[:FILE_MATCHES]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
