#ifndef	_TYPE_H
#define	_TYPE_H
#include <map>
#include <vector>
#include<iostream>

using namespace std;

//      Buffers used later to read and write into files
#define	BUFF_SIZE_LONG	1000000
#define	BUFF_SIZE_SHORT	512

// struct (word-id, word-sentiment-prob-vector, word-polarity)
struct Word_atr {
	int id; // vocabulary index
        int polarity;
};

// struct (string, probability)
struct stringProba {
	string value; // string
	double proba;
};

// map of words / ids [string => int]
typedef map<string, int> mapword2id;

// map of ids / words [int => string]
typedef map<int, string> mapid2word;

// map of words / attributes_of_words [string => word_attr]
typedef map<string, Word_atr> mapword2atr;

// map of stamps / stamp-ids [double => int]
typedef map<double, int> mapstamp2atr;

// Document type: vector of words, length, id and timestamp-id
class document {
    public:
        int * words;
        int stamp;
        string docID;
        int len;
	
    // Constructor. Retrieve the length of the document and allocate memory for storing the documents
    document(int length) {
        this->len = length;
        docID = "";
        words = new int[length]; // words stores the word token ID at each position of document, which is integer
    }
    
    // Destructor
    ~document() {
        if (words != NULL){
                delete [] words;
                words = NULL;
        }
    }
};

#endif
