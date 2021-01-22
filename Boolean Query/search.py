import re
import nltk
import sys
import getopt

import linecache
import _pickle as pickle
from collections import namedtuple
import math


def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")


def shunting_yard(query):
    """
    Returns Reverse Polish Notation queue (postfix  notation/AST) after
    parsing the input query in infix notation.
    Query terms in the return value are processed using stemming and case folding.
    """
    stemmer = nltk.stem.porter.PorterStemmer()

    output_queue = []
    operator_stack = []

    tokens = nltk.word_tokenize(query)
    for token in tokens:
        # Check order of precedence when processing
        if token == '(':
            operator_stack.append(token)

        elif token == ')':
            while len(operator_stack) > 0 and operator_stack[-1] != '(':
                output_queue.append(operator_stack.pop())

            operator_stack.pop()  # Remove '('

        elif token == 'NOT':
            operator_stack.append(token)

        elif token == 'AND':
            while len(operator_stack) > 0 and operator_stack[-1] == 'NOT':
                output_queue.append(operator_stack.pop())

            operator_stack.append(token)

        elif token == 'OR':
            while (len(operator_stack) > 0 and \
                   (operator_stack[-1] == 'NOT' or operator_stack[-1] == 'AND')):
                output_queue.append(operator_stack.pop())

            operator_stack.append(token)

        else:
            # Apply stemming and case folding as done during indexing
            output_queue.append(stemmer.stem(token).lower())

    while len(operator_stack) > 0:
        output_queue.append(operator_stack.pop())

    return output_queue


def read_postings_list_from_disk(word, dictionary, postings_fo):
    """
    Returns the postings list of the `word`, read from the postings file.
    """
    # TODO: time - use pickle to load serialized data instead.
    # Store skip pointers on disk
    # Use f.tell() to store offsets in index.py
    (num_docs, offset, size, skip_len) = dictionary[word]

    postings_fo.seek(offset, 0)
    postings = postings_fo.read(size)
    postings = [int(docID) for docID in postings[:-1].split(',')]  # Remove trailing space and convert to list

    return postings


# Wrapper tuple class to store list of document IDs and the length of the skip pointers for this list
QueryResult = namedtuple('QueryResult', ['doc_Ids', 'skip_len'])


def get_query_result(operand, dictionary, postings_fo):
    """
    Returns `operand` if it is a `QueryResult`.
    Otherwise, read from the disk and return a QueryResult containing the postings list
    of document IDs and length of skip pointers.
    """
    if type(operand) is QueryResult:
        return operand

    if operand not in dictionary:
        return QueryResult([], 0)

    postings_list = read_postings_list_from_disk(operand, dictionary, postings_fo)

    return QueryResult(postings_list, dictionary[operand][-1])


def has_skip(idx, skip_len, total_len):
    """
    Returns True if a non-zero logical skip pointer can be used to
    skip a few steps forward.
    """
    if skip_len == 0:
        return False

    # sqrt(total_len) skip pointers are evenly placed
    if idx % skip_len == 0 and (idx + skip_len) < total_len:
        return True

    return False


def perform_not_query(operand, dictionary, postings_fo):
    """
    Returns the QueryResult containing doc IDs from the documents collection that
    are not in the postings list of operand.
    NOT list_1 - All elements of the list of all document IDs that are not present
    in list_1, i.e., all_docId_list AND NOT list_1.
    """
    ALL_DOC_IDS = '## all doc IDs ##'  # Special token that has a postings list containing all doc IDs
    all_docs_res = get_query_result(ALL_DOC_IDS, dictionary, postings_fo)

    operand_res = get_query_result(operand, dictionary, postings_fo)

    return perform_and_not_query(all_docs_res, operand_res, dictionary, postings_fo)


def perform_and_not_query(operand_1, operand_2, dictionary, postings_fo):
    """
    Returns the QueryResult containing doc IDs that are in the postings list of
    operand_1 but not operand_2.
    list_1 AND NOT list_2 - All elements of list_1 that are not present in list_2
    """
    (list_1, skip_1) = get_query_result(operand_1, dictionary, postings_fo)
    (list_2, skip_2) = get_query_result(operand_2, dictionary, postings_fo)

    i = 0
    j = 0

    results = []
    while i < len(list_1) and j < len(list_2):
        if list_1[i] < list_2[j]:
            results.append(list_1[i])
            i += 1
        elif list_1[i] == list_2[j]:
            i += 1
            j += 1
        elif list_1[i] > list_2[j]:
            # Check if skip pointer can be used
            if has_skip(j, skip_2, len(list_2)) and list_2[j + skip_2] <= list_1[i]:
                j += skip_2

                while has_skip(j, skip_2, len(list_2)) and list_2[j + skip_2] <= list_1[i]:
                    j += skip_2
            else:
                j += 1

    if i < len(list_1):
        results.extend(list_1[i:])

    return QueryResult(results, int(math.sqrt(len(results))))


def perform_and_query(operand_1, operand_2, dictionary, postings_fo):
    """
    Returns the QueryResult containing doc IDs that are in both the postings
    lists of operand_1 and operand_2.
    list_1 AND list_2 - All elements of list_1 that are present in list_2
    """
    (list_1, skip_1) = get_query_result(operand_1, dictionary, postings_fo)
    (list_2, skip_2) = get_query_result(operand_2, dictionary, postings_fo)

    i = 0
    j = 0

    results = []
    while i < len(list_1) and j < len(list_2):
        if list_1[i] == list_2[j]:
            results.append(list_1[i])
            i += 1
            j += 1
        elif list_1[i] < list_2[j]:
            # Check if skip pointer can be used
            if has_skip(i, skip_1, len(list_1)) and list_1[i + skip_1] <= list_2[j]:
                i += skip_1

                while has_skip(i, skip_1, len(list_1)) and list_1[i + skip_1] <= list_2[j]:
                    i += skip_1
            else:
                i += 1
        elif list_2[j] < list_1[i]:
            # Check if skip pointer can be used
            if has_skip(j, skip_2, len(list_2)) and list_2[j + skip_2] <= list_1[i]:
                j += skip_2

                while has_skip(j, skip_2, len(list_2)) and list_2[j + skip_2] <= list_1[i]:
                    j += skip_2
            else:
                j += 1

    return QueryResult(results, int(math.sqrt(len(results))))


def perform_or_query(operand_1, operand_2, dictionary, postings_fo):
    """
    Returns the QueryResult containing doc IDs that are either in the
    postings lists of operand_1 or in that of operand_2.
    list_1 OR list_2 - Set of unique elements in list_1 and list_2
    """
    (list_1, skip_1) = get_query_result(operand_1, dictionary, postings_fo)
    (list_2, skip_2) = get_query_result(operand_2, dictionary, postings_fo)

    if len(list_1) == 0 or len(list_2) == 0:
        if len(list_1) > 0:
            return QueryResult(list_1, 0)
        else:
            return QueryResult(list_2, 0)

    i = 0
    j = 0

    results = []
    while i < len(list_1) and j < len(list_2):
        if list_1[i] < list_2[j]:
            results.append(list_1[i])
            i += 1
        elif list_2[j] < list_1[i]:
            results.append(list_2[j])
            j += 1
        elif list_1[i] == list_2[j]:
            results.append(list_1[i])
            i += 1
            j += 1

    if i < len(list_1):
        results.extend(list_1[i:])
    if j < len(list_2):
        results.extend(list_2[j:])

    return QueryResult(results, int(math.sqrt(len(results))))


def perform_search_query(query, dictionary, postings_fo):
    """
    Returns list of unique document IDs that match the input search query
    in Reverse Polish Notation.
    """
    stack = []
    i = 0
    while i < len(query):
        token = query[i]

        # TODO: optimisation using De Morgan's rule, merge smaller ANDs, AND NOT, NOT NOT?
        if token == 'NOT':
            if (i + 1) < len(query) and query[i + 1] == 'AND':  # AND NOT
                i += 1
                operand_2 = stack.pop()
                operand_1 = stack.pop()
                query_res = perform_and_not_query(operand_1, operand_2, dictionary, postings_fo)
            elif (i + 1) < len(query) and query[i + 1] == 'NOT':  # NOT NOT
                i += 1
            else:
                operand = stack.pop()
                query_res = perform_not_query(operand, dictionary, postings_fo)
        elif token == 'AND':
            operand_2 = stack.pop()
            operand_1 = stack.pop()
            query_res = perform_and_query(operand_1, operand_2, dictionary, postings_fo)
        elif token == 'OR':
            operand_2 = stack.pop()
            operand_1 = stack.pop()
            query_res = perform_or_query(operand_1, operand_2, dictionary, postings_fo)
        else:
            query_res = get_query_result(token, dictionary, postings_fo)

        stack.append(query_res)
        i += 1

    search_results = stack.pop()
    if type(search_results) is not QueryResult:
        search_results = get_query_result(search_results, dictionary, postings_fo)

    return search_results.doc_Ids


def run_search(dict_file, postings_file, queries_file, results_file):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    print('running search on the queries...')

    # { word_type : (num_docs, offset_bytes, size_bytes, skip_len) }
    f = open(dict_file, 'rb')
    dictionary = pickle.load(f)
    f.close()

    postings_fo = open(postings_file, "rt")
    results_fo = open(results_file, "wt")

    line_num = 1
    line = linecache.getline(queries_file, line_num)
    while line != '':
        query = shunting_yard(line)
        search_results = perform_search_query(query, dictionary, postings_fo)

        # Write results to output file
        search_results_str = ' '.join([str(docID) for docID in search_results]) + '\n'
        results_fo.write(search_results_str)

        line_num += 1
        line = linecache.getline(queries_file, line_num)

    postings_fo.close()
    results_fo.close()


dictionary_file = postings_file = file_of_queries = output_file_of_results = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    else:
        assert False, "unhandled option"

if dictionary_file is None or postings_file is None or file_of_queries is None or file_of_output is None:
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
