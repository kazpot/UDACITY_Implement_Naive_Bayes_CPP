#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <map>
#include <assert.h>
#include <algorithm>
#include "classifier.h"
using namespace std;
#define M_PI 3.14159265359
/**
 * Initializes GNB
 */
GNB::GNB() {

}

GNB::~GNB() {}

void GNB::train(vector<vector<double>> data, vector<string> labels)
{  
    /*
        Trains the classifier with N data points and labels.

        INPUTS
        data - array of N observations
          - Each observation is a tuple with 4 values: s, d, 
            s_dot and d_dot.
          - Example : [
                [3.5, 0.1, 5.9, -0.02],
                [8.0, -0.3, 3.0, 2.2],
                ...
            ]

        labels - array of N labels
          - Each label is one of "left", "keep", or "right".
    */
    assert(data.size()==labels.size());

    map <string, vector<vector<double>>> lfm;
    map <string, int> class_count;
    int data_size = data.size();
    
    //label list (keep, right, left)
    labels_list_ = labels;
    std::sort(labels_list_.begin(), labels_list_.end());
    auto newEnd = std::unique(labels_list_.begin(), labels_list_.end());
    labels_list_.erase(newEnd, labels_list_.end());
   
    features_count_ = data[0].size();
 
    //init
    for (auto l : labels_list_){
        class_count[l] = 0;
        vector <double> temp (data[0].size(), 0.0); //init vector [0.0, 0.0, 0.0, 0.0]
        f_stats_[l].push_back(temp); //for sum of each feature, later mean
        f_stats_[l].push_back(temp); //for stddev of each feature
    }
 
    //calc sum per label
    for (auto i = 0; i < data_size; i++) {
        lfm[labels[i]].push_back(data[i]); // data collecton per class
        class_count[labels[i]] += 1; // data count per class
        for (auto j = 0; j < features_count_; j++){
            f_stats_[labels[i]][0][j] += data[i][j]; // calc sum per feature
        }
    }

    // transform sum to mean
    for (auto l : labels_list_){
        for (auto j = 0; j < features_count_; j++){
            f_stats_[l][0][j] /= class_count[l];
        }
    }

    // calc stddev for each class -> sqrt(sum(pow(x - mean, 2))/(N-1))
    for (auto l : labels_list_){
        for (auto j = 0; j < features_count_; j++){
            double numerator = 0;
            for (auto i = 0; i < lfm[l].size(); i++){
                numerator += pow(lfm[l][i][j] - f_stats_[l][0][j], 2);
            }
            f_stats_[l][1][j] = sqrt(numerator/(class_count[l]-1));
        }
    }
}

string GNB::predict(vector <double> vec)
{
    /*
        Once trained, this method is called and expected to return 
        a predicted behavior for the given observation.

        INPUTS

        observation - a 4 tuple with s, d, s_dot, d_dot.
          - Example: [3.5, 0.1, 8.5, -0.2]

        OUTPUT

        A label representing the best guess of the classifier. Can
        be one of "left", "keep" or "right".
        """
        # TODO - complete this
    */
    
    assert (features_count_ == vec.size());
    
    double max = 0;
    double mean = 0;
    double stddev = 0;
    map <string, double> p;
    string result;

    for (auto l : labels_list_) {
        for (auto i = 0; i < features_count_; i++){
            mean = f_stats_[l][0][i];
            stddev = f_stats_[l][1][i];
            double exponent = exp(-pow(vec[i] - mean, 2) / (2 * pow(stddev,2)));
            p[l] += 1.0 /sqrt(2 * M_PI * stddev) * exponent;
        }

        if (max < p[l]) {
            max = p[l];
            result = l;
        }
    }
    return result;
}