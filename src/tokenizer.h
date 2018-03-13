#ifndef _TOKENIZER_H
#define _TOKENIZER_H

#include <string>
#include <vector>

using namespace std;

class tokenizer {
protected:
    vector<string> tokens;

public:
    tokenizer(string str, string seperators = " ");    
    void parse(string str, string seperators);
    int nbTokens();
    string tokenAt(int i);
    void deleteAt(int p);
};

#endif

