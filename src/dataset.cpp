#include "dataset.h"
#include "tokenizer.h"
#include <stdlib.h>

using namespace std; 

// Object to store dataset matrix and vocabulary
dataset::dataset() {
    pdocs = NULL;
    word2atr.clear();
    nbDocs = 0;
    vocabSize = 0;
    nbStamps = 0;
    corpusSize = 0;
}

// Delete object
dataset::~dataset(void) {
    deallocate();
}

// De-allocate object
void dataset::deallocate() {
    if (pdocs) {
        for (int i = 0; i < nbDocs; i++) 
            delete pdocs[i];		
        delete [] pdocs;
        pdocs = NULL;
    }
}
    
// Add a document entry to the dataset matrix
void dataset::addDoc(document * doc, int idx) {
    if (0 <= idx && idx < nbDocs)
        pdocs[idx] = doc;
}

// Load data from an array of documents accompanied with an array of titles
int dataset::loadFromVect(vector<string> &docs, vector<string> &docTitles, bool hndleTime) {
    // Init data stats
    corpusSize = 0;
    nbDocs = docs.size();
    pdocs = new document*[ nbDocs ];
    
    // feature extraction
    for (int d=0; d < nbDocs; d++) {      // loop over docs
        // Tokenize document
        tokenizer st ( ((string) (docs[d])).c_str(), " " );        
        // loop over words
        for (int w = 0; w < st.nbTokens() - 2; w++) {
            // Retrieve word ID from vocabulary
            string token = st.tokenAt(w + 2).c_str();            
            mapword2atr::iterator it;
            it = word2atr.find(token);
            if (it == word2atr.end() ) {  // new word
                int id = word2atr.size();
                word2atr.insert( pair<string, Word_atr>(token, (Word_atr) {id, -1} ) );
                id2word.insert (pair<int, string>(id, token));
            }
        }
        // timestamps
        if (hndleTime) {       // Read timestamps and put them into a dictionary
            double stamp = atof ( st.tokenAt(1).c_str() );
            mapstamp2atr::iterator its;
            its = stamp2atr.find(stamp);
            if (its == stamp2atr.end() ) {  // new word
                int id = stamp2atr.size();
                stamp2atr.insert( pair<double, int>(stamp, id) );
            }
        }
        // Titles
        this->titles.push_back( ((string) docTitles[d]) );
    }
    
    printf("\nfinished feature extraction");
    // Update stamp number
    if (hndleTime)
    {
        nbStamps = stamp2atr.size();
        // Sort timestamps
        int i = 0;
        for (mapstamp2atr::iterator itss = stamp2atr.begin(); itss != stamp2atr.end(); itss++)
            itss->second = i++;
        // Save timestamp labels into an array to write it later (in psi_file)
        stampLabels.resize (nbStamps);
        for (mapstamp2atr::iterator iterator = stamp2atr.begin(); iterator != stamp2atr.end(); iterator++)
            stampLabels[iterator->second] = iterator->first;
    }
    else
    {
        stamp2atr.insert( pair<double, int>(0.0, 0) );
        nbStamps = 1;
    }
    printf("\nfinished handle time");
    
    // Insert feature-preprocessing here (stem, frequency, etc)

    // Fill data matrix
    for (int d=0; d < nbDocs; d++) {      // loop over docs
        
        // Tokenize document
        tokenizer st(docs[d].c_str(), " ");        
        
        // Allocate new document in data matrix
        pdocs[d] = new document(st.nbTokens() - 2);
        pdocs[d]->docID = docs[d];   // unique ID

        if (hndleTime)
            pdocs[d]->stamp = stamp2atr.at( atof(st.tokenAt(1).c_str()) );  // timestamp id (int)
        else
            pdocs[d]->stamp = 0.0;
        
        corpusSize += pdocs[d]->len;

        // loop over words
        for (int w = 0; w < pdocs[d]->len; w++) {
            // Retrieve word ID from vocabulary
            string token = st.tokenAt(w + 2).c_str();            
            int tokID = word2atr.at(token).id;
            // Update data matrix
            pdocs[d]->words[w] = tokID;
        }
    }    
    
    // Set vovab size
    vocabSize = word2atr.size();
    return 0;
}
