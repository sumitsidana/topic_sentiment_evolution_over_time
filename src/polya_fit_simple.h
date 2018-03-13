#ifndef	_POLYA_FIT_SIMPLE_H
#define	_POLYA_FIT_SIMPLE_H

double digama ( double x, int *ifault );
int polya_fit_simple(int ** data, double * alhpa, int _K, int _nSample, bool verbose);

#endif
