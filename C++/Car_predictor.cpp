#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

using namespace std;

// Structure to represent a node in the decision tree
struct Node {
    int feature_index;  // Index of the feature used for the split
    double threshold;   // Threshold value for the split
    Node* left;         // Pointer to the left subtree
    Node* right;        // Pointer to the right subtree
    int class_label;    // The class label at leaf nodes
    bool is_leaf;       // Boolean to check if the node is a leaf

    Node() : feature_index(-1), threshold(0), left(NULL), right(NULL), class_label(-1), is_leaf(false) {}
};

// Function to calculate Gini Index for splitting
double giniIndex(vector<vector<double>>& dataset, vector<int>& labels, int feature_index, double threshold) {
    vector<int> left_count(2, 0);
    vector<int> right_count(2, 0);

    for (int i = 0; i < dataset.size(); i++) {
        if (dataset[i][feature_index] < threshold)
            left_count[labels[i]]++;
        else
            right_count[labels[i]]++;
    }

    int left_size = left_count[0] + left_count[1];
    int right_size = right_count[0] + right_count[1];
    int total_size = left_size + right_size;

    double left_gini = 1.0;
    if (left_size > 0)
        left_gini -= pow((double)left_count[0] / left_size, 2) + pow((double)left_count[1] / left_size, 2);

    double right_gini = 1.0;
    if (right_size > 0)
        right_gini -= pow((double)right_count[0] / right_size, 2) + pow((double)right_count[1] / right_size, 2);

    return ((double)left_size / total_size) * left_gini + ((double)right_size / total_size) * right_gini;
}

// Function to find the best split
pair<int, double> getBestSplit(vector<vector<double>>& dataset, vector<int>& labels) {
    int best_feature = -1;
    double best_threshold = 0;
    double best_gini = numeric_limits<double>::max();

    for (int feature_index = 0; feature_index < dataset[0].size(); feature_index++) {
        for (int i = 0; i < dataset.size(); i++) {
            double threshold = dataset[i][feature_index];
            double gini = giniIndex(dataset, labels, feature_index, threshold);

            if (gini < best_gini) {
                best_gini = gini;
                best_feature = feature_index;
                best_threshold = threshold;
            }
        }
    }

    return {best_feature, best_threshold};
}

// Recursive function to build the decision tree
Node* buildTree(vector<vector<double>>& dataset, vector<int>& labels, int depth, int max_depth) {
    // Check if all labels are the same
    int class_0 = count(labels.begin(), labels.end(), 0);
    int class_1 = labels.size() - class_0;

    // Stop if max depth is reached or dataset is pure (all 0s or all 1s)
    if (class_0 == 0 || class_1 == 0 || depth >= max_depth) {
        Node* leaf = new Node();
        leaf->is_leaf = true;
        leaf->class_label = (class_0 > class_1) ? 0 : 1;
        return leaf;
    }

    // Find the best split
    pair<int, double> best_split = getBestSplit(dataset, labels);

    // Split the dataset
    vector<vector<double>> left_data, right_data;
    vector<int> left_labels, right_labels;

    for (int i = 0; i < dataset.size(); i++) {
        if (dataset[i][best_split.first] < best_split.second) {
            left_data.push_back(dataset[i]);
            left_labels.push_back(labels[i]);
        } else {
            right_data.push_back(dataset[i]);
            right_labels.push_back(labels[i]);
        }
    }

    // Create the current node
    Node* node = new Node();
    node->feature_index = best_split.first;
    node->threshold = best_split.second;

    // Recursively build the left and right children
    node->left = buildTree(left_data, left_labels, depth + 1, max_depth);
    node->right = buildTree(right_data, right_labels, depth + 1, max_depth);

    return node;
}

// Function to make predictions using the decision tree
int predict(Node* tree, vector<double>& sample) {
    if (tree->is_leaf)
        return tree->class_label;

    if (sample[tree->feature_index] < tree->threshold)
        return predict(tree->left, sample);
    else
        return predict(tree->right, sample);
}

int main() {
    // Dataset: Age, Income, Credit Score
    vector<vector<double>> dataset = {
        {25, 50000, 650},
        {40, 100000, 720},
        {35, 85000, 680},
        {22, 45000, 600},
        {50, 120000, 800}
    };

    // Labels: 0 = Will Not Buy, 1 = Will Buy
    vector<int> labels = {0, 1, 1, 0, 1};

    // Set the maximum depth for the tree
    int max_depth = 3;

    // Build the decision tree
    Node* tree = buildTree(dataset, labels, 0, max_depth);

    // Test the decision tree with a new sample
    vector<double> test_sample = {30, 60000, 700};  // Age, Income, Credit Score
    int predicted_class = predict(tree, test_sample);

    cout << "Predicted class: " << (predicted_class == 1 ? "Will Buy" : "Will Not Buy") << endl;

    return 0;
}
