import numpy as np
from classes import NN_Classifier
from classes import Validator



def feature_search_demo(data):
    # Load the dataset from the file
    # data = np.loadtxt(data_file)
    
    num_instances, num_features = data.shape
    num_features -= 1  # Deduct one to exclude the class label column

    # Initialize feature set for both forward selection and backward elimination
    forward_features = []  
    backward_features = list(range(1, num_features + 1))  # 1-based index for features

    print(f"Your dataset has {num_features} features with {num_instances} instances.")
    print("Type the number of the algorithm you want to run.")
    print("1. Forward Selection\n2. Backward Elimination")
    
    choice = int(input())

    if choice == 1:
        print("Running Forward Selection")
        classifier = Validator(data, forward_features)
        features = forward_features
        print(f"Using no features (default rate) and training the classifier, I get an accuracy of {classifier.validate() * 100}%")
    elif choice == 2:
        print("Running Backward Elimination")
        classifier = Validator(data, backward_features)
        features = backward_features
        print(f"Using all features and training the classifer, I get an accuracy of {classifier.validate() * 100}%")
    else:
        print("Invalid choice. Please select either 1 or 2.")
        return
    print("Beginning search...")
    search_algorithm(data, features, choice, classifier)