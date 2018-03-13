#include "model.h"
#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include <iostream>
#include <map>
using namespace std;

int main(int argc, char ** argv) {
    
    if (argc <= 1)
    {
        printf ("\n****** Config file not set, please use '-est config_file.txt' !");
        return 1;
    }
    
    printf("\n***** Welcome to TTS 1.0 ! *****\n");
    bool verbose = true;
    
    // Instanciate a new model
    model tts;
    
    // Load parameters from file 
    tts.read_params_from_file (argv[2]);
    
    // Dataset 
    dataset *data = new dataset();
    // Data
    vector<string> docs;
    vector<string> titles;    
    
    // data : 1    // From File
    if (verbose)
        printf("\nLoading data...");
    ifstream fin;
    fin.open(tts.datasetFile.c_str(), ifstream::in);
    string line;
    char buff[BUFF_SIZE_LONG];
    docs.clear();
    int counter=0;
   
    while (fin.getline(buff, BUFF_SIZE_LONG)) {
    //while (true) {
        line = buff;
        if(!line.empty()) {
            docs.push_back(line);
            titles.push_back("Title-of-doc");
            counter++;
        }
    }
    
    printf("\nRead %d docs...",counter);

    // Load data
    printf("\nLoading from vector...");
    data->loadFromVect (docs, titles, tts.handleTime);

    // Set id2word dictionary for topic-top-word generation
    printf("\nSetting id2word dictionary...");
    tts.id2word = &data->id2word;

    // Execute model
    printf("\nExecuting TTS...");
    tts.learn (*data, verbose);

    return 0;
}
