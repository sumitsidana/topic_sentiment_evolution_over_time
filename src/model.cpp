#include <cmath>
#include <algorithm>
#include "model.h"

using namespace std;

// Allocate a new model with default parameters
model::model (void) {
    // Corpus statistics
    nbTopics = 30;
    nbSentLabs = 2;
    vocabSize = 0;
    nbDocs = 0;
    corpusSize = 0;
    name = "tts_test";

    // Model parameters
    maxIters = 300;
    currentIter = 0;
    updatParamStep = 1;
    updatGamma = true;
    nbTopWords = 20;
    
    // Choose what information to be handled, from "sentiments" and "time"
    handleSent = true;
    handleTime = true;
        
    // Hyperparameters (if set to -1, they'll be reset to default in the param-initializaion step)
    _alpha  = -1.0;
    _beta = -1.0;
    _gammaPos = _gammaNeg = -1.0;
    _mu = -1.0;
    _gt = 0.5;
    
    gammaPace = 100.0;
}

// Default constructor
model::~model(void) {
}

// Execute model
int model::learn(dataset &data, bool verbose) {
    if (data.corpusSize == 0 || data.nbDocs == 0 || data.vocabSize == 0)
    {
        if (verbose)
            printf("\nError : data not found !\n");
        return 1;
    }
    
    if (verbose)
        printf("\nBegin learning..\n");
    
    // Set sentiment lexicon (if any)
    if (handleSent)
        loadLexicon(sentLexFile, data.word2atr, nbSentLabs, verbose);
    else        // Ignore sentiment lexicon
        nbSentLabs = 1;
    
    // Initialize model parameters
    if ( initModelParams(data, verbose) )
        return 1;
    // Initialize word-topic-sentiment assignments
    if ( initEstimate(data) )
        return 1;
    
    // Estimate
    if ( estimate(data, verbose) )
        return 1;
    
    // Close IO stream
    fin.close();
    if (verbose)
        printf ("Success !\n");
    return 0;
}

// Initialize model parameters with start values
int model::initModelParams (dataset &data, bool verbose) {
    if (verbose)
        printf("\nInit model parameters :");

    nbDocs = data.nbDocs;
    vocabSize = data.vocabSize;
    nbStamps = data.nbStamps;
    corpusSize = data.corpusSize;
    
    // Count variables
    nd.resize(nbDocs);
    for (int m = 0; m < nbDocs; m++)
            nd[m]  = 0;

    ndz.resize(nbDocs);
    for (int m = 0; m < nbDocs; m++) {
        ndz[m].resize(nbTopics);
        for (int z = 0; z < nbTopics; z++)
            ndz[m][z] = 0;
    }

    nzlw.resize(nbTopics);
    for (int z = 0; z < nbTopics; z++) {
        nzlw[z].resize(nbSentLabs);
        for (int l = 0; l < nbSentLabs; l++) {
            nzlw[z][l].resize(vocabSize);
            for (int r = 0; r < vocabSize; r++)
                nzlw[z][l][r] = 0;
        }
    }

    nzlt.resize(nbTopics);
    for (int z = 0; z < nbTopics; z++) {
        nzlt[z].resize(nbSentLabs);
        for (int l = 0; l < nbSentLabs; l++) {
            nzlt[z][l].resize(nbStamps);
            for (int r = 0; r < nbStamps; r++)
                nzlt[z][l][r] = 0;
        }
    }

    nz.resize(nbTopics);
    for (int z = 0; z < nbTopics; z++)
        nz[z] = 0;

    nzl.resize(nbTopics);
    for (int z = 0; z < nbTopics; z++) {
        nzl[z].resize(nbSentLabs);
        for (int l = 0; l < nbSentLabs; l++)
            nzl[z][l] = 0;
    }
    
    // Proba distributions
    pi_zl.resize(nbTopics);
    for (int z = 0; z < nbTopics; z++)
        pi_zl[z].resize(nbSentLabs);

    theta_dz.resize(nbDocs);
    for (int m = 0; m < nbDocs; m++)
        theta_dz[m].resize(nbTopics);

    phi_zlw.resize(nbTopics);
    for (int z = 0; z < nbTopics; z++) {
        phi_zlw[z].resize(nbSentLabs);
        for (int l = 0; l < nbSentLabs; l++)
                phi_zlw[z][l].resize(vocabSize);
    }

    psi_zlt.resize(nbTopics);
    for (int z = 0; z < nbTopics; z++) {
        psi_zlt[z].resize(nbSentLabs);
        for (int l = 0; l < nbSentLabs; l++)
                psi_zlt[z][l].resize(nbStamps);
    }

    // Posterior P
    p.resize(nbTopics);
    for (int z = 0; z < nbTopics; z++)
        p[z].resize(nbSentLabs);

    // Hyperparameters
    // "alpha"
    if (_alpha <= 0)    // Set "alpha" to default
        _alpha =  50 / (double) (nbTopics);

    alpha_z.resize(nbTopics);
    for (int z = 0; z < nbTopics; z++) {
        alpha_z[z] = _alpha;
        alphaSum += alpha_z[z];	
    }
    
    // "beta"
    if (_beta <= 0)     // Set "beta" to default
        _beta = 1.0/nbTopics;
    
    beta_zlw.resize(nbTopics);
    betaSum_zl.resize(nbTopics);
    for (int z = 0; z < nbTopics; z++) {
        beta_zlw[z].resize(nbSentLabs);
        betaSum_zl[z].resize(nbSentLabs);
        for (int l = 0; l < nbSentLabs; l++) {
            betaSum_zl[z][l] = 0.0;
            beta_zlw[z][l].resize(vocabSize);
            for (int r = 0; r < vocabSize; r++) {
                beta_zlw[z][l][r] = _beta;
                betaSum_zl[z][l] = _beta;
            }
        }
    }

    // "mu"
    if (_mu <= 0)       // Set "mu" to default
        _mu = 0.01;
    mu_zlt.resize(nbTopics);
    muSum_zl.resize(nbTopics);
    for (int z = 0; z < nbTopics; z++) {
        mu_zlt[z].resize(nbSentLabs);
        muSum_zl[z].resize(nbSentLabs);
        for (int l = 0; l < nbSentLabs; l++) {
            muSum_zl[z][l] = 0.0;
            mu_zlt[z][l].resize(nbStamps);
            for (int r = 0; r < nbStamps; r++) {
                mu_zlt[z][l][r] = _mu;
                muSum_zl[z][l] += _mu;
            }
        } 		
    }

    // Word prior transformation matrix (default lambda=1, otherwise take it from lexicon)
    lambda_lw.resize(nbSentLabs);
    for (int l = 0; l < nbSentLabs; l++) {
        lambda_lw[l].resize(vocabSize);
        for (int r = 0; r < vocabSize; r++)
            lambda_lw[l][r] = 1;
    }

    // Incorporate prior information into "beta"
    this->prior2beta(data);
    
    // Find best "gamma"
    if (_gammaPos < 0 || _gammaNeg < 0) {  // Find best "gamma" by greedy search
        _gammaPos = 1;
        _gammaNeg = 1;
        if (verbose)
                printf("\n  _gamma will be set automatically using GT = 0.5");
    }
    else {      // Symmetric "gamma" fixed by user
        if (verbose)
                printf("\n  _gamma(-) fixed by user to %.3f", _gammaNeg);
    }
    
    // Set "gamma"
    gamma_l.resize(nbSentLabs);
    gammaSum = _gammaNeg  + _gammaPos;
    gamma_l[0] = _gammaNeg;
    gamma_l[1] = _gammaPos;      // pos
    
    // Print out 
    if (verbose) {
        printf("\n  nbSentiLabs = %d", nbSentLabs);
        printf("\n  nbTopics = %d", nbTopics);
        printf("\n  maxIters = %d", maxIters);
        printf("\n  updatParamStep = %d", updatParamStep);
        printf("\n  handleSent = %d", handleSent);
        printf("\n  handleTime = %d", handleTime);
        printf("\n  sentLexFile = %s", sentLexFile.c_str() );
        printf("\n  datasetFile = %s", datasetFile.c_str() );
        printf("\n  outputFolder = %s", outputFolder.c_str() );
        printf("\n  _alpha = %.3f", _alpha);
        printf("\n  _beta = %.3f", _beta);
        printf("\n  _gamma(+) = %.3f", _gammaPos);
        printf("\n  _gamma(-) = %.3f", _gammaNeg);
        printf("\n  _mu = %.3f\n", _mu);
        printf("\n  _gt = %.3f\n", _gt);
    }
    return 0;
}

// Incorporate sentiment lexicon (multiply beta by lambda)
int model::prior2beta(dataset &data) {
    mapword2atr::iterator wordIt;
    
    for (wordIt = data.word2atr.begin(); wordIt != data.word2atr.end(); wordIt++) {
        // Adapting lambda values to the lexicon content
        if (wordIt->second.polarity != -1) { // Word in lexicon
            // Activer "lambda" sur la polarité du mot et la désactiver ailleurs
            for (int j = 0; j < nbSentLabs; j++)
                if (j != wordIt->second.polarity)
                    lambda_lw[j][wordIt->second.id] = 0;
        }
    }

    // Updating beta values according to lambda
    for (int k = 0; k < nbTopics; k++) {
      for (int l = 0; l < nbSentLabs; l++) {
          betaSum_zl[k][l] = 0.0;
          for (int r = 0; r < vocabSize; r++) {
                beta_zlw[k][l][r] = beta_zlw[k][l][r] * lambda_lw[l][r];
                betaSum_zl[k][l] += beta_zlw[k][l][r];
            }
        }
    }
    return 0;
}

void model::computePhi() {
    for (int z = 0; z < nbTopics; z++)
        for (int l = 0; l < nbSentLabs; l++)
            for(int r = 0; r < vocabSize; r++)
                phi_zlw[z][l][r] = (nzlw[z][l][r] + beta_zlw[z][l][r]) / (nzl[z][l] + betaSum_zl[z][l]);
}

void model::computePsi() {
    for (int z = 0; z < nbTopics; z++)
        for (int l = 0; l < nbSentLabs; l++)
            for(int r = 0; r < nbStamps; r++)
                psi_zlt[z][l][r] = (nzlt[z][l][r] + mu_zlt[z][l][r]) / (nzl[z][l] + muSum_zl[z][l]);
}

void model::computePi() {
    for (int z = 0; z < nbTopics; z++)
        for (int l = 0; l < nbSentLabs; l++)
            pi_zl[z][l] = (nzl[z][l] + gamma_l[l]) / (nz[z] + gammaSum);
}

void model::computeTheta() {
    for (int m = 0; m < nbDocs; m++)
        for (int z = 0; z < nbTopics; z++)
            theta_dz[m][z] = (ndz[m][z] + alpha_z[z]) / (nd[m] + alphaSum);
}

// Save model to different files
int model::save (dataset &data, string model_name, bool verbose) {
    ostringstream ss;
    ss << model_name << "_T" << nbTopics << "_a" << _alpha << "_b" << _beta << "_g-" << _gammaNeg << "_g+" << _gammaPos << "_m" << _mu << "_iter" << currentIter;
    string model_name_long = ss.str();
    if (verbose)
        printf("\n\nSaving the final model to '%s'\n", (outputFolder + name).c_str());
    //
    if (saveTopicSentWrds(model_name_long + ".twords")) 
        return 1;
    if (saveTopicWrds(model_name_long + ".topics")) 
        return 1;
    if (saveTopicSents(model_name_long + ".pi")) 
        return 1;
    if (saveTopicSentEvol(&data.stampLabels[0], model_name_long + ".psi"))
        return 1;
    //if (saveDocTopics(model_name_long + ".theta"))
        //return 1;
    return 0;
}

// Save model : word-topic assignments
int model::saveWrdTopics (dataset &data, string filename) {
    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
        printf("Cannot save file %s!\n", filename.c_str());
        return 1;
    }
    for (int m = 0; m < data.nbDocs; m++) {
        fprintf(fout, "%s \n", data.pdocs[m]->docID.c_str());
        for (int n = 0; n < data.pdocs[m]->len; n++)
            fprintf(fout, "%d:%d:%d ", data.pdocs[m]->words[n], l[m][n], z[m][n]); //  wordID:sentiLab:topic
        fprintf(fout, "\n");
    }
    fclose(fout);
    return 0;
}

// Save model : topic top words
int model::saveTopicSentWrds(string filename) 
{
    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
        printf("Cannot save file %s!\n", filename.c_str());
        return 1;
    }
    
    // Get topic top words
    vector<vector<vector<stringProba> > > topWords = getTopicSentWrds (nbTopWords);
    // Write out
    for (int k = 0; k < nbTopics; k++) {
        for (int l = 0; l < nbSentLabs; l++) {
            if (handleSent) // Pas besoin d'écrie Label
                fprintf(fout, "Topic : s%d_t%d\n", l, k); // Topic : s0_t0
            else
                fprintf(fout, "\nTopic : %d\n", k);
            for (int i = 0; i < (int) topWords[k][l].size(); i++) { 
                fprintf(fout, "\t%s\t%.6f\n", topWords[k][l][i].value.c_str(), topWords[k][l][i].proba);
            }
        }
    }
    fclose(fout);      
    return 0;    
}

// Save model : topic words
int model::saveTopicWrds(string filename) 
{
    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
        printf("Cannot save file %s!\n", filename.c_str());
        return 1;
    }
    
    // Get topic top words
    vector<vector<stringProba> > topWords = processTopicWrds (vocabSize);
    // Write out
    for (int k = 0; k < nbTopics; k++) {
        fprintf(fout, "Topic : s0_t%d*s1_t%d\n", k, k);
        for (int i = 0; i < (int) topWords[k].size(); i++) { 
            fprintf(fout, "\t%s\t%.6f\n", topWords[k][i].value.c_str(), topWords[k][i].proba);
        }
    }
    fclose(fout);      
    return 0;    
}

// Save model : pi
int model::saveTopicSents(string filename) {
    vector<vector<double> > topicSents = getTopicSents();
    if ( topicSents.size() ) {  // size may be "0" if modelType == "0"
        FILE * fout = fopen(filename.c_str(), "w");
        if (!fout) {
            printf("Cannot save file %s!\n", filename.c_str());
            return 1;
        }

        for (int z = 0; z < nbTopics; z++) {
            fprintf(fout, "Topic : s0_t%d*s1_t%d\n", z, z);
            for (int l = 0; l < nbSentLabs; l++)
                fprintf(fout, "\t%.6f\n", topicSents[z][l]);
        }
        fclose(fout);
    }
    return 0;
}

// Save model : theta
int model::saveDocTopics(string filename) {
    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
        printf("Cannot save file %s!\n", filename.c_str());
        return 1;
    }
    for(int m = 0; m < nbDocs; m++) {
        fprintf(fout, "Document %d\n", m);
            for (int z = 0; z < nbTopics; z++) {
                fprintf(fout, "\t%f ", theta_dz[m][z]);
                fprintf(fout, "\n");
            }
    }
    fclose(fout);
    return 0;
}

// Save model : psi
int model::saveTopicSentEvol(double *stampLabels, string filename) {
    vector<vector<vector<int> > > distribs = getTopicSentEvol ();
    if ( distribs.size() ) {      // size may be "0" if modelType in {0, 1}
        FILE * fout = fopen(filename.c_str(), "w");
        if (!fout) {
            printf("Cannot save file %s!\n", filename.c_str());
            return 1;
        }
        // Generate timestamp labels
        vector<string> stampLabs;
        stampLabs.resize(nbStamps);
        //
        for (int i = 0; i < nbStamps; i++)
        {
            struct tm * dt;
            time_t ts = (int) ( stampLabels[i] / 1000);
            dt = localtime(&ts);
            //printf("\n%d-%d-%d", dt->tm_mday, dt->tm_mon+1, dt->tm_year+1900);

            //fprintf(fout, " \t%.0f", data.stampLabels[i]);
            std::ostringstream sstream;
            sstream << dt-> tm_mday << "-" << (dt->tm_mon+1) << "-" <<  (dt->tm_year+1900) << " " << (dt->tm_hour) << ":" << (dt->tm_min);
            string str_t = sstream.str();
            stampLabs[i] = str_t.c_str();
        }
        //
        for (int z = 0; z < nbTopics; z++) {
            for (int l = 0; l < nbSentLabs; l++) {
                if (handleSent)
                    fprintf(fout, "Topic : s%d_t%d\n", l, z);
                for (int r = 0; r < nbStamps; r++) {
                    //fprintf(fout, "\t%.6f\n", psi_zlt[z][l][r]);       // just a proba (no much meaning)
                    //fprintf(fout, "\t%s\t%.6f\n", stampLabs[r].c_str(), distribs [z][l][r]);    // real-size estimation
                    fprintf(fout, "\t%s\t%.6f\n", stampLabs[r].c_str(), psi_zlt[z][l][r]);    // real-size estimation
            }
           }
        }
        fclose(fout);
    }
    return 0;
}

// Initialize counts for estimation (random initialization of topic-word assignment)
int model::initEstimate (dataset &data) {

    int sentiLab, topic;
    srand(time(0)); // Initialize for random number generation
    z.resize(nbDocs);
    l.resize(nbDocs);
    // Loop over words
    for (int m = 0; m < nbDocs; m++) {
        int docLength = data.pdocs[m]->len;
        z[m].resize(docLength);
        l[m].resize(docLength);

        for (int w = 0; w < docLength; w++) {
            // Look for this word in the lexicon
            string word = id2word->at(data.pdocs[m]->words[w]);
            mapword2atr::iterator wordIt;
            wordIt = data.word2atr.find(word);
            if (wordIt->second.polarity != -1) {
                sentiLab = wordIt->second.polarity; // incorporate prior information into the model
            }
            else {
                sentiLab = (int) (((double)rand() / RAND_MAX) * nbSentLabs);
                if (sentiLab == nbSentLabs)
                    sentiLab = nbSentLabs -1;  // to avoid over array boundary
            }
            l[m][w] = sentiLab;
            
            // Random initialize of topic assginment
            topic = (int)(((double)rand() / RAND_MAX) * nbTopics);
            if (topic == nbTopics)  
                topic = nbTopics - 1; // to avoid over array boundary
            z[m][w] = topic;

            // Count-variable update
            nd[m]++;
            ndz[m][topic]++;
            nzlw[topic][sentiLab][data.pdocs[m]->words[w]]++;
            nzlt[topic][sentiLab][data.pdocs[m]->stamp]++;      // The same stamp for all token in document
            nzl[topic][sentiLab]++;
            nz[topic]++;
        }
    }
    return 0;
}

// Estimate by Gibbs sampling algorithm
int model::estimate (dataset &data, bool verbose) {
    int sentiLab, topic;
    
    if (verbose)
        printf ("\nSampling %d iterations..\n", maxIters);
    for (currentIter = 1; currentIter <= maxIters; currentIter++) 
    {
        if (verbose) {
            printf ("%d..", currentIter);
            fflush(stdout);
        }
        for (int m = 0; m < nbDocs; m++) {
            for (int n = 0; n < data.pdocs[m]->len; n++) {
                sampling (data, m, n, sentiLab, topic); // Gibbs sampling
                l[m][n] = sentiLab;
                z[m][n] = topic;
            }
        }
        // Update "alpha" and "gamma" hyperparameters (first 10 iterations then less frequently)
        if (updatParamStep > 0 && currentIter % updatParamStep == 0)
        {
            // Update Alpha "ML"
            this->updateAlphaML (verbose);
            
            // ML-based update
            // this->updateGammaML (verbose);
            
            // GT-based update
            if (this->updatGamma) {
                if (verbose)
                    printf("\n  Optimising parameter \"gamma\" using GT..");
                
                this->updateGammaGT (verbose);
                
                if (verbose)
                    printf("\n  newGamma [0] : %.4f , newGamma [1] : %.4f\n", gamma_l[0], gamma_l[1]);
            }
            // Compute posterior distribs.
            computePi();
            computeTheta();
            computePhi();
            computePsi();
        }
    }
    
    // If not just saved, re-compute distribs
    if (currentIter % updatParamStep != 1)      // when the 'for' loop ends, it increments currentIter
    {
        computePi();
        computeTheta();
        computePhi();
        computePsi();
    }

    // Save model
    save (data, outputFolder + name, verbose);
    
    if (verbose)
        printf("Gibbs sampling completed!\n");
    return 0;
}

// Gibbs sampling procedure
int model::sampling (dataset &data, int m, int n, int& sentiLab, int& topic) {
    topic = z[m][n];
    sentiLab = l[m][n];
    int w = data.pdocs[m]->words[n]; // the ID/index of the current word token in vocabulary 
    int t = data.pdocs[m]->stamp; // the index of current document's stamp
    double u;

    // Exclude current token
    nd[m]--;
    ndz[m][topic]--;
    nz[topic]--;
    nzlw[topic][sentiLab][w]--;
    nzlt[topic][sentiLab][t]--;
    nzl[topic][sentiLab]--;

    // Do multinomial sampling via cumulative method
    for (int k = 0; k < nbTopics; k++) {
        for (int l = 0; l < nbSentLabs; l++) {
            p[k][l] =       (nzlw[k][l][w] + beta_zlw[k][l][w]) / (nzl[k][l] + betaSum_zl[k][l]) *
            ( pow ((nzlt[k][l][t] + mu_zlt[k][l][t]) / (nzl[k][l] + muSum_zl[k][l]), 1/(nd[m]+1)) ) *
            (ndz[m][k] + alpha_z[k]) / (nd[m] + alphaSum) *
            (nzl[k][l] + gamma_l[l]) / (nz[k] + gammaSum);
        }
    }

    // Accumulate multinomial parameters
    for (int k = 0; k < nbTopics; k++)  {
        for (int l = 0; l < nbSentLabs; l++) {
            if (l==0)  {
                if (k==0) 
                    continue;
                else 
                    p[k][l] += p[k-1][nbSentLabs-1]; // accumulate the sum of the previous array
            }
            else
                p[k][l] += p[k][l-1];
        }
    }
    
    // Probability normalization
    u = ((double) rand() / RAND_MAX) * p[nbTopics-1][nbSentLabs-1];
    
    // Sample sentiment label l, where l \in [0, S-1]
    bool loopBreak=false;
    for (topic = 0; topic < nbTopics; topic++) {
        for (sentiLab = 0; sentiLab < nbSentLabs; sentiLab++) {
            if (p[topic][sentiLab] > u) {
                loopBreak = true;
                break;
            }
        }
        if (loopBreak == true) {
            break;
        }
    }
    // To avoid over array boundary
    if (topic == nbTopics) 
        topic = nbTopics - 1;
    if (sentiLab == nbSentLabs) 
        sentiLab = nbSentLabs - 1;

    // Add estimated 'z' and 'l' to count variables
    nd[m]++;
    ndz[m][topic]++;
    nz[topic]++;
    nzlw[topic][sentiLab][w]++;
    nzlt[topic][sentiLab][t]++;
    nzl[topic][sentiLab]++;
    return 0;
}

// Update the non-symmetric alpha parameter by the method of 
int model::updateAlphaML (bool verbose) {

    int ** data; // Temp valuable for exporting 3-dimentional array to 2-dimentional
    double * alpha_temp;
    data = new int*[nbTopics];
    for (int k = 0; k < nbTopics; k++) {
        data[k] = new int[nbDocs];
        for (int m = 0; m < nbDocs; m++)
            data[k][m] = 0;
    }

    alpha_temp = new double[nbTopics];
    for (int k = 0; k < nbTopics; k++)
        alpha_temp[k] = 0.0;

    // Update alpha
    for (int k = 0; k < nbTopics; k++)
        for (int m = 0; m < nbDocs; m++)
            data[k][m] = ndz[m][k]; // ntldsum[j][k][m];

    for (int k = 0; k < nbTopics; k++)
        alpha_temp[k] =  alpha_z[k]; // alpha[j][k];

    polya_fit_simple(data, alpha_temp, nbTopics, nbDocs, verbose);
    
    // Update alpha
    alphaSum = 0.0;
    for (int k = 0; k < nbTopics; k++) {
        //printf("Old alpha[%d] = %f\n", k, alpha_z[k]);
        alpha_z[k] = alpha_temp[k];
        //printf("New alpha[%d] = %f\n", k, alpha_temp[k]);
        alphaSum += alpha_z[k];
    }
    return 0;
}

// Update the non-symmetric alpha parameter by the method of 
int model::updateGammaML (bool verbose) {
    
    int ** data; // Temp valuable for exporting 3-dimentional array to 2-dimentional
    double * gamma_temp;
    data = new int*[nbSentLabs];
    for (int l = 0; l < nbSentLabs; l++) {
        data[l] = new int[nbTopics];
        for (int k = 0; k < nbTopics; k++)
            data[l][k] = 0;
    }

    gamma_temp = new double[nbSentLabs];
    for (int l = 0; l < nbSentLabs; l++)
        gamma_temp[l] = 0.0;

    // Update alpha
    for (int l = 0; l < nbSentLabs; l++)
        for (int m = 0; m < nbTopics; m++)
            data[l][m] = nzl[m][l]; // ntldsum[j][k][m];

    for (int l = 0; l < nbSentLabs; l++)
        gamma_temp[l] =  gamma_l[l]; // alpha[j][k];

    polya_fit_simple(data, gamma_temp, nbSentLabs, nbTopics, verbose);
    
    // Update alpha
    gammaSum = 0.0;
    for (int l = 0; l < nbSentLabs; l++) {
        //printf("Old alpha[%d] = %f\n", k, alpha_z[k]);
        gamma_l[l] = gamma_temp[l];
        //printf("New alpha[%d] = %f\n", k, alpha_temp[k]);
        gammaSum += gamma_l[l];
    }
    return 0;
}

// Update the non-symmetric gamma parameter by the method of "Ground-Truth-based"
int model::updateGammaGT (bool verbose) {
    // Verify criterion 
    double crit = this->computeGammaFromGT (verbose, _gt);
    
    // Update Gamma
    if ( this->gammaPace > 0 )    // Increase gamma Neg
    {
        for (int z = 0; z < this->nbTopics; z++)
            this->gamma_l[0] = this->gammaPace;
        for (int z = 0; z < this->nbTopics; z++)
            this->gamma_l[1] = 0.01;
    }
    else if ( this->gammaPace < 0 )       // Increase Gamma Pos
    {
        for (int z = 0; z < this->nbTopics; z++)
            this->gamma_l[0] = 0.01;
        for (int z = 0; z < this->nbTopics; z++)
            this->gamma_l[1] = - this->gammaPace;
    }
    // Update gammaSum
    this->gammaSum = 0.0;
    for (int l = 0; l < nbSentLabs; l++)
        this->gammaSum += this->gamma_l[l];
    return 0;
}

// Compute Gamma Criterion (used for Gamma Neg Update)
double model::computeGammaFromGT (bool verbose, double negPortion) {
    // Calculate pos/neg proportions
    double cptPos = 0.1;
    double cptNeg = 0.1;
    for (int z = 0; z < nbTopics; z++) {
        cptNeg += (double) nzl[z][0];//pi_zl[z][0];// * topicDocs[z].size();      // neg
        cptPos += (double) nzl[z][1];//pi_zl[z][1];// * topicDocs[z].size();      // pos
    }
    if (verbose)
        printf("\n  #words (-) = %.2f, #words (+) = %.2f", cptNeg, cptPos);
    
    double estNpPortion = (double)cptNeg / (double)(cptPos+cptNeg);
    if (verbose)
        printf ("\n  estNegPortion = %.2f", estNpPortion);
    
    // Compute gammaPace (how many pseudo counts to be added in order to approach realNpRatio?)
    double realNbNeg = (cptPos + cptNeg) * negPortion;
    double realNbPos = (cptPos + cptNeg) - realNbNeg;
    
    double diff = realNbNeg - realNbPos;
    if (verbose)
        printf("\n  diff = %.2f, but should be = %.2f", cptNeg - cptPos, diff);
    
    // Update Gamma Pace 
    this->gammaPace = (diff-(cptNeg - cptPos)) / this->nbTopics;
    return estNpPortion;        // Just for info, never used.
}

// Deduce global topics (regdless of sentiment label))
vector<vector<stringProba> > model::processTopicWrds (int nbWords) {
    vector<vector<stringProba> > topicWords;
    topicWords.resize (nbTopics);
    
    if (nbWords > vocabSize)
        nbWords = vocabSize; // Print out entire vocab list    
    mapid2word::iterator it;
    stringProba s;
    vector<pair<int, double> > words_probs;
    
    for (int k = 0; k < nbTopics; k++) {
        // Build word probas vector
        words_probs.clear();
        for (int w = 0; w < vocabSize; w++) {
            pair<int, double> word_prob;
            word_prob.first = w; // w: word id
            word_prob.second = 0.0;     // Init value
            for (int l = 0; l < nbSentLabs; l++) {
                word_prob.second += phi_zlw[k][l][w] * pi_zl[k][l]; // topic-word probability
            }
            words_probs.push_back(word_prob);
        }
        // Sort word probs vector
        std::sort( words_probs.begin(), words_probs.end(), pairSort() );
        // Return only first "nbWords"
        for (int i = 0; i < nbWords; i++) {
            it = id2word->find(words_probs[i].first);
            if (it != id2word->end()) {
                s.value = it->second;
                s.proba = words_probs[i].second;
                topicWords[k].push_back(s);
            }
        }
    }
    return topicWords;
}

// Deduce topic documents (assign each document to the most probable topic) -- probas not sorted !!
vector<vector< pair<int, double> > > model::processTopicDocs() {
    vector<vector< pair<int, double> > > topicDocs;
    topicDocs.resize(nbTopics);
    // Loop over docs
    for (int m = 0; m < nbDocs; m++) {
        // For doc "m", calculate the most probable topic
        int topic = 0;
        for (int z = 1; z < nbTopics; z++) {
            if (theta_dz[m][z] > theta_dz[m][topic])
                topic = z;
        }
        topicDocs[topic].push_back( pair<int, double>(m, theta_dz[m][topic]) );
    }
    return topicDocs;
}

// Return topic word distribution (sorted by probability))
vector<vector<vector<stringProba> > > model::getTopicSentWrds (int nbWords) {
    vector<vector<vector<stringProba> > > topicWords;
    topicWords.resize(nbTopics);
    
    if (nbWords > vocabSize)
        nbWords = vocabSize; // Print out entire vocab list
    
    mapid2word::iterator it;
    stringProba s;

    for (int k = 0; k < nbTopics; k++) {
        topicWords[k].resize(nbSentLabs);
        for (int l = 0; l < nbSentLabs; l++) {
            // Build word probas vector
            vector<pair<int, double> > words_probs;
            pair<int, double> word_prob;
            for (int w = 0; w < vocabSize; w++) {
                word_prob.first = w; // w: word id
                word_prob.second = phi_zlw[k][l][w]; // topic-word probability
                words_probs.push_back(word_prob);
            }
            // Sort word probs vector
            std::sort(words_probs.begin(), words_probs.end(), pairSort());
            
            for (int i = 0; i < nbWords; i++) {
                it = id2word->find(words_probs[i].first);
                if (it != id2word->end()) {
                    s.value = it->second;
                    s.proba = words_probs[i].second;
                    topicWords[k][l].push_back(s);
                }
            }
        }
    }    
    return topicWords;
}

// Return topic titles (sorted by probability)
vector<vector<stringProba> > model::processTopicTitles (string* titles, int nbTitles) {
    vector<vector<stringProba> > topicTitles;
    topicTitles.resize (nbTopics);
    
    // Get topic-docs ids
    vector<vector< pair<int, double> > > topicDocs = processTopicDocs();
    
    // Sort topic-docs ids
    for (int z=0; z<nbTopics; z++) {
        std::sort(topicDocs[z].begin(), topicDocs[z].end(), pairSort());
    }
    
    // Take the most probable doc titles
    for (int z = 0; z < nbTopics; z++) {
        int realNbTits = (int) topicDocs[z].size();       // Real nb titles = min (nbTitles, effective nb docs in topic)
        if (realNbTits > nbTitles)
            realNbTits = nbTitles;
        for (int i = 0; i < realNbTits; i++) {
            stringProba tit;
            tit.value = titles[ topicDocs[z][i].first ];
            tit.proba = topicDocs[z][i].second;
            topicTitles[z].push_back(tit);
        }
    }    
    return topicTitles;
}

// Return topic sentiments (sorted by probability). Sentiments in the same order as lexicon
vector<vector<double> > model::getTopicSents () {
    vector<vector<double> > topicSents;
    if ( handleSent )
        topicSents = pi_zl;
    return topicSents;
}

// Return document's most probable topic
vector<vector<double> > model::getDocTopics () {
    return theta_dz;
}

// Return topic-sentiment evolution over time (nb of docs at each timestamp)
vector<vector<vector<int> > > model::getTopicSentEvol () {
    vector<vector<vector<int> > > topicSentEvol;
    if (handleTime) {
        // Calculate topic sizes
        vector<vector< pair<int, double> > > topicDocs = processTopicDocs();

        topicSentEvol.resize(nbTopics);
        for (int z = 0; z < nbTopics; z++) {
            topicSentEvol[z].resize(nbSentLabs);
            for (int l = 0; l < nbSentLabs; l++) {
                topicSentEvol[z][l].resize(nbStamps);
                for (int r = 0; r < nbStamps; r++)
                    topicSentEvol [z] [l] [r] = psi_zlt[z][l][r] * pi_zl[z][l] * topicDocs[z].size(); // # of docs of topic 'i' and sent 'l' at time 'r'
            }
        }
    }
    return topicSentEvol;
}

// Load lexicon according to "nbSeniLabs" and inject it into "word2atr"
int model::loadLexicon(string lexiconPath, mapword2atr &word2atr, int nbSents, bool verbose) {
    char buff [BUFF_SIZE_SHORT];
    string line;
    vector<double> wordPrior;    
    int pol;
    double tmp, val;
    FILE * fin = fopen(lexiconPath.c_str(), "r");
    if (!fin) {
        printf("Cannot read file %s!\n", lexiconPath.c_str());
        return 1;
    }    
    fgets(buff, BUFF_SIZE_SHORT - 1, fin);  // Ignore first line
    while (fgets(buff, BUFF_SIZE_SHORT - 1, fin) != NULL) {
        line = buff;
        tokenizer strtok(line, "\t\r\n");
        // Put here word preprocessing (stemmiong, etc) in order to match words from "word2atr"
        if (strtok.nbTokens() < 4)  {
            if (verbose)
                printf("Warning! The word [%s] in the lexicon has only %d scores !\n", line.c_str(), nbSents);
            continue;
        }
        else if ( word2atr.find(strtok.tokenAt(0).c_str()) == word2atr.end() ) {    // Word in lexicon but not in vocabulary => (ignore it))
            continue;
        }            
        else {  // Word in lexicon and in vocabulary, update corresponding probabilities in word2atr
            // Tailor sentiment labels w.r.t. nbSentiLabs (consider neutral or not))
            if (nbSents == 2) {     // Delete neutral score
                strtok.deleteAt(2); // "2" is the index of neutral sore in lexicon
            }
            tmp = 0.0;      // To deduce polarity (max val))
            pol = 0;
            wordPrior.clear();
            for (int k = 0; k < nbSents; k++) {
                val = atof ( strtok.tokenAt(k+1).c_str() );
                if (tmp < val) {
                    tmp = val;
                    pol = k;
                }
                wordPrior.push_back(val);
            }
            // Update polarity
            word2atr.at(strtok.tokenAt(0)).polarity = pol;
        }
    }
    fclose(fin);
    return 0;
}

// Read model parameters from file (provisoire))
int model::read_params_from_file (string filename) {
    char buff[BUFF_SIZE_SHORT];
    string line;
    FILE * fin = fopen(filename.c_str(), "r");
    if (!fin) {
        printf("Cannot read file %s\n", filename.c_str());
        return 1;
    }
    while (fgets(buff, BUFF_SIZE_SHORT - 1, fin)) {
        line = buff;
        tokenizer strtok(line, "= \t\r\n");
        string optstr = strtok.tokenAt(0);
        string optval = strtok.tokenAt(1);
        
        if(optstr == "nbSentLabs")
                this->nbSentLabs = atoi(optval.c_str());        
        else if(optstr == "nbTopics") 
                this->nbTopics = atoi(optval.c_str());	
        else if(optstr == "maxIters") 
                this->maxIters = atoi(optval.c_str());	
        else if(optstr == "updatParamStep")
                this->updatParamStep = atoi(optval.c_str());				
        else if(optstr == "updatGamma")
                this->updatGamma = atoi(optval.c_str());				
        else if(optstr == "nbTopWords") 
                this->nbTopWords = atoi(optval.c_str());				
        else if (optstr == "handleSent") 
                this->handleSent = atoi(optval.c_str());
        else if(optstr == "handleTime") 
                this->handleTime = atoi(optval.c_str());	
        else if(optstr == "modelName") 
                this->name = optval;	
        else if(optstr == "sentLexFile") 
                this->sentLexFile = optval;	
        else if(optstr == "datasetFile") 
                this->datasetFile = optval;	
        else if(optstr == "outputFolder") 
                this->outputFolder = optval;	
        else if (optstr == "_alpha")
                this->_alpha = atof(optval.c_str());
        else if (optstr == "_beta")    
                this->_beta = atof(optval.c_str());
        else if (optstr == "_gammaNeg")    
                this->_gammaNeg = atof(optval.c_str());
        else if (optstr == "_gammaPos")    
                this->_gammaPos = atof(optval.c_str());
        else if (optstr == "_gt")    
                this->_gt = atof(optval.c_str());
        else if (optstr == "_mu")    
                this->_mu = atof(optval.c_str());
    }
    fclose(fin);
    return 0;
}

    