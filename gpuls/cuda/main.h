#ifndef GPULS_MAIN_H
#define GPULS_MAIN_H

void gpuls(unsigned int * objectId, float * timeX, float * magY, float * magDY, unsigned int sizeData, double minFreq, double maxFreq, unsigned int freqToTest, float ** pgram);

#endif
