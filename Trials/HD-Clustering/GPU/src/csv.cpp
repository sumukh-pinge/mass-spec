#include "../include/csv.hpp"


const char * DELIMS = ",";
const int MAX_LINE_LENGTH = 10000; // Choose this large enough for your need.
                                  // Or use a dynamic buffer together with 
                                  // std::string and getline(istream, string&).

void readFCPSTrainDat(char* filename, int& nFeatures, int& nClasses,
                    std::vector<std::vector<float>>& X,
                    std::vector<int>& y){

    std::fstream fin(filename);
    fin.ignore(MAX_LINE_LENGTH,'\n');  //skip first line (header)

    // Prepare a C-string buffer to be used when reading lines.
    char buffer[MAX_LINE_LENGTH] = {};

    std::set<int> class_set;
    int data_idx = 0;

    // Read one line at a time.
    while ( fin.getline(buffer, MAX_LINE_LENGTH) ) {
        const char* class_buf = strtok(buffer, DELIMS);
        if(class_buf == NULL)
            break;
        int class_num = atoi(class_buf) - 1;
        y.push_back(class_num);  // class start from zero
        class_set.insert(class_num);

        // new line
        X.push_back(std::vector<float>());
        for (;;) {
            char* feature_buf = strtok(NULL, DELIMS);
            if (feature_buf == NULL)
                break;
            else
                X[data_idx].push_back(atof(feature_buf));
        }
        
        data_idx++;
    }
    nFeatures = X[0].size();
    nClasses = class_set.size();

    // Cleanup
    fin.close();
    printf("Dataset loading Done \n");
}
