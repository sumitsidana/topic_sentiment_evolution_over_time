#include <string>
#include "tokenizer.h"
 
using namespace std;

// Default constructor
tokenizer::tokenizer (string str, string seperators) {
    parse(str, seperators);
}

// Tokenize a string
void tokenizer::parse (string str, string seperators) {
    int n = str.length();
    int start, stop;
    start = str.find_first_not_of (seperators);
    while (start >= 0 && start < n) {
        stop = str.find_first_of (seperators, start);
        if (stop < 0 || stop > n)
            stop = n;
        tokens.push_back (str.substr(start, stop - start));	
        start = str.find_first_not_of (seperators, stop + 1);
    }    
}

// Returns number of tokens
int tokenizer::nbTokens() {
    return tokens.size();
}

// Delete a token at position p
void tokenizer::deleteAt (int p) {
    tokens.erase( tokens.begin() + p );
}

// Returns token at position i
string tokenizer::tokenAt (int i) {
    if (i >= 0 && i < (int)tokens.size())
        return tokens[i];
    else
        return "";
}
