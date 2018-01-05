Proj5 README
___________________________________________________________


Files edited/added by me.
========================
* get_positive_features.m - code to get positive features for facial recognition for the 6713 images provided in the data folder
* get_random_negative_features.m - code to get negative features. This is the first iternation of this function
* get_random_negative_features2.m - optimized code to get negative features. This is the second iternation. This is the default choice right now in proj5.m
* proj5.m - the main project code which also has the code for training the SVM, negative hard mining and retraining the SVM. For getting positive and negative examples, I have added code to save matrices for debugging. They can be ignored in the submitted code. **To turn on negative hard mining please set variable on line 147 to true.**
* run_detector.m - The detector code with sliding window at different scales. I have modified the code to accept the size of feature vector (1116) as well. 
* run_hard_negatives_detector.m - The detector for mining hard negatives. Set threshold on line number 150 in proj5.m and number of samples to mine in line number 149 in proj5.m