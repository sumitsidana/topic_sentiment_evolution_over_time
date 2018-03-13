#include <stdio.h>
#include <math.h>
#include <vector>

#include "dataset.h"
#include "polya_fit_simple.h"
#include "tokenizer.h"

using namespace std;

// Function "sort_pred" for sorting word probs
struct pairSort {
    bool operator () (const std::pair<int,double> &left, const std::pair<int,double> &right) {
        return left.second > right.second;
    }
};

// Default constructor
class model {    
    public:
        model(void);
        ~model(void);

        // Lexicon file path;
        string sentLexFile;
        string datasetFile;
        string outputFolder;
        string name;
        
        // Dataset stats
        int nbTopics;
        int nbSentLabs;
        
        // Learn params
        bool handleSent;       // Activate "sentiment" option
        bool handleTime;       // Actiate "time" option
        
        int nbTopWords;
        
        int maxIters;
        int updatParamStep;
        bool updatGamma;
        
        // id2word map file (used to generate topic top words)
        mapid2word* id2word;        
        
        // Hyperparameters
        double _alpha;
        double _beta;
        double _gammaPos, _gammaNeg;
        double _mu;
        
        // Current Gamma pace
        double gammaPace;

        // launch function
        int learn(dataset &data, bool verbose = false);
        
        // Getters
        vector<vector<vector<stringProba> > > getTopicSentWrds (int nbWords=10);
        vector<vector<stringProba> > processTopicTitles (string* titles, int nbTitles=10);
        vector<vector<double> > getTopicSents ();
        vector<vector<vector<int> > > getTopicSentEvol ();
        vector<vector< pair<int, double> > > processTopicDocs();
        vector<vector<stringProba> > processTopicWrds(int nbWords=10);
        vector<vector<double> > getDocTopics();
        
        int read_params_from_file (string filename);

    private:
        // Dataset stats
        int nbDocs;
        int corpusSize;
        int vocabSize;
        int nbStamps;
        
        ifstream fin;
        int currentIter;
        
        // Count variables
        vector<int> nd;
        vector<vector<int> > ndz;
        vector<vector<vector<int> > > nzlw;
        vector<vector<vector<int> > > nzlt;
        vector<vector<int> > nzl;
        vector<int> nz;

        // Topic and label assignments
        vector<vector<double> > p;
        vector<vector<int> > z;
        vector<vector<int> > l;

        // Model parameters (distributions))
        vector<vector<double> > pi_zl;
        vector<vector<double> > theta_dz;
        vector<vector<vector<double> > > phi_zlw;
        vector<vector<vector<double> > > psi_zlt;

        // Hyperparameters 
        vector<double> alpha_z;
        double alphaSum;
        vector<vector<vector<double> > > beta_zlw;
        vector<vector<vector<double> > > mu_zlt;
        vector<vector<double> > betaSum_zl;
        vector<vector<double> > muSum_zl;
        vector<double> gamma_l;
        double gammaSum;
        double _gt;
        vector<vector<double> > lambda_lw;

        vector<vector<double> > opt_alpha_lz;
        
        // Methods
        int initModelParams(dataset &data, bool verbose);
        int initEstimate(dataset &data);
        int estimate(dataset &data, bool verbose);
        int prior2beta(dataset &data);
        int sampling(dataset &data, int m, int n, int& sentiLab, int& topic);
        int loadLexicon(string lexiconPath, mapword2atr &word2atr, int nbSents, bool verbose);

        // Compute parameter functions
        void computePi(); 
        void computeTheta(); 
        void computePhi(); 
        void computePsi();
        
        // Compute Gamma Criterion
        double computeGammaFromGT(bool verbose, double negPortion);

        // Update parameter functions
        int updateAlphaML (bool verbose);
        int updateGammaML (bool verbose);
        int updateGammaGT (bool verbose);

        // Save model parameter funtions 
        int save(dataset &data, string model_name, bool verbose);
        int saveWrdTopics(dataset &data, string filename);
        int saveTopicSents(string filename);
        int saveDocTopics(string filename);
        int saveTopicSentEvol(double *stampLabels, string filename);
        int saveTopicSentWrds(string filename);
        int saveTopicWrds (string filename);
        
};
