#include "host.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <vector>
#include <time.h>
#include <chrono>

using namespace std;

#define N_CENTER		26	//number of centers. (e.g., isolet: 26,)
#define N_FEAT			617	//feature per input (e.g., isolet: 617)
#define Dhv				2048  //hypervectors length
int EPOCH = 5;
bool shuffled = true;
string X_data_path = "./isolet_trainX.bin";
string y_data_path = "./isolet_trainY.bin";

