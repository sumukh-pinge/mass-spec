#ifndef __CSV_H__
#define __CSV_H__

#include <stdio.h>
#include <vector>
#include <set>
#include <cstdlib>
#include <string>
#include <cstring>
#include <fstream>
#include <iostream>

void readFCPSTrainDat(char* filename, int& nFeatures, int& nClasses,
                    std::vector<std::vector<float>>& X,
                    std::vector<int>& y);

#endif
