#include "polya_fit_simple.h"
#include <math.h>
#include <stdio.h>

using namespace std;

// Compute Digamma function
double digama ( double x, int *ifault ) 
{
    double c = 8.5;
    double d1 = -0.5772156649;
    double r;
    double s = 0.00001;
    double s3 = 0.08333333333;
    double s4 = 0.0083333333333;
    double s5 = 0.003968253968;
    double value;
    double y;
    //  Check the input.
    if ( x <= 0.0 )
    {
        value = 0.0;
        *ifault = 1;
        return value;
    }
    //  Initialize.
    *ifault = 0;
    y = x;
    value = 0.0;
    //  Use approximation if argument <= S.
    if ( y <= s )
    {
        value = d1 - 1.0 / y;
        return value;
    }
    //  Reduce to DIGAMA(X + N) where (X + N) >= C.
    while ( y < c )
    {
        value = value - 1.0 / y;
        y = y + 1.0;
    }
    //  Use Stirling's (actually de Moivre's) expansion if argument > C.
    r = 1.0 / y;
    value = value + log ( y ) - 0.5 * r;
    r = r * r;
    value = value - r * ( s3 - r * ( s4 - r * s5 ) );

    return value;
}

// Optimize alpha values with ML
int polya_fit_simple (int ** data, double * alpha, int _K, int _nSample, bool verbose)
{
    int K = _K;                 // hyperparameter dimension
    int nSample = _nSample;     // total number of samples, i.e.documents
    int polya_iter = 100000;    // maximum number of fixed point iterations
    int ifault1, ifault2;
    double sum_alpha_old;
    double * old_alpha = NULL;
    double sum_g = 0; // sum_g = sum_digama(data[i][k] + old_alpha[k]),
    double sum_h = 0; // sum_h + sum_digama(data[i] + sum_alpha_old) , where data[i] = sum_data[i][k] for all k,
    double * data_row_sum = NULL; // the sum of the counts of each data sample P = {P_1, P_2,...,P_k}
    bool sat_state = false;
    int i, k, j;  
    old_alpha = new double[K];

    for (k = 0; k < K; k++)
        old_alpha[k] = 0;
    
    data_row_sum = new double[nSample];
    for (i = 0; i < nSample; i++)
        data_row_sum[i] = 0;
    
    // data_row_sum
    for (i = 0; i < nSample; i++)
        for (k = 0; k < K; k++)
            data_row_sum[i] += *(*(data+k)+i) ;
    
    // simple fix point iteration
    for (i = 0; i < polya_iter; i++) {  // reset sum_alpha_old
        sum_alpha_old = 0;
        // update old_alpha after each iteration
        for (j = 0; j < K; j++)
            old_alpha[j] = *(alpha+j);
        // calculate sum_alpha_old
        for (j = 0; j < K; j++) {
            sum_alpha_old += old_alpha[j];
        }
        for (k = 0; k < K; k++) {
            sum_g = 0;
            sum_h = 0;
            // calculate sum_g[k]
            for (j = 0; j < nSample; j++)
                sum_g += digama( *(*(data+k)+j) + old_alpha[k], &ifault1);
            // calculate sum_h
            for (j = 0; j < nSample; j++)
                sum_h += digama(data_row_sum[j] + sum_alpha_old, &ifault1);
            
            // update alpha (new)
            *(alpha+k) = old_alpha[k] * (sum_g - nSample * digama(old_alpha[k], &ifault1)) / (sum_h - nSample * digama(sum_alpha_old, &ifault2));
            
            //printf("\n *** new gamma [%d] = %.4f", k, *(alpha+k));
            
        }
        // terminate iteration ONLY if each dimension of {alpha_1, alpha_2, ... alpha_k} satisfy the termination criteria,
        for (j = 0; j < K; j++) {
            if (fabs( *(alpha+j) - old_alpha[j]) > 0.000001) 
                break;
            if ( j == K-1) {
                sat_state = true;
            }
        }
        // check whether to terminate the whole iteration
        if(sat_state) {
            if (verbose)
                printf("  Terminated at iteration # %d\n", i );
            break;
        }
        else if (i == polya_iter-1)
            if (verbose)
                printf("  Not converged. Terminated at iteration # %d\n", i+1 );
    }
    return 0;
}
