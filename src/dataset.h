#ifndef	_DATASET_H
#define	_DATASET_H

#include "types.h"
#include <fstream>
#include <sstream>

using namespace std; 

class dataset {

public:
    mapword2atr word2atr;
    mapstamp2atr stamp2atr;
    mapid2word id2word; 

    document ** pdocs; // documents
    vector<string> titles;      // document titles (used to label topics))
    ifstream fin;

    int nbDocs;
    int vocabSize;
    int nbStamps;
    vector<double> stampLabels; // stampLabels[stamp id] = stamp value
    int corpusSize;

    vector<string> docs; // for buffering dataset
    
    // functions 
    dataset();
    ~dataset(void);

    void deallocate ();  
    void addDoc (document * doc, int idx);
    int loadFromVect (vector<string> &docs, vector<string> &docTitles, bool hndleTime);
};

#endif
