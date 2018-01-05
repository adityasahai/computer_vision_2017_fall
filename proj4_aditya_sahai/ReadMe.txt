# README for Proj4 - Scene recognition with bag of words
## CS 6476 - Introduction to Computer Vision

##### This readme includes descriptions of files modified or added by me. 

Files & Folders - Their Application
-----
- **LMGist.m** - Function to calculate GIST features by Aude Oliva, Antonio Torralba
- **imresizecrop.m** - Function accopanying LMGist mentioned above to help with calculations.
- **get_gist.m** - Function to calculate and normalize GIST features for this project. It calculates the param fields and calculates gist features for image paths passed to it. Returns normalized matrix of features for each image.
- **get_bags_of_sifts.m** - uses vl_dsift, vl_kdtreebuild, vl_kdtreequery and histc to calculate bags of sift features for each image in image paths.
- **svm_classify.m** - Linear SVM classification code using vl_svmtrain()
- **nearest_neighbor_classify.m** - Simple 1 NN classifier
- **build_vocabulary.m** - Function to build visual word vocabulary. uses vl_kmeans()
- **get_tiny_images.m** - Code to create a feature vector using 16x16 tiny images. 
- **primal_svm_classify.m** - Uses Olivier Chapelle's SVM training function to train non-linear SVM
- **primal_svm_classify_linear.m** - Uses Olivier Chapelle's SVM training function to train a linear SVM
- **primal_svm.m** - Olivier Chapelle's training function
- **proj4.m** - Project file. I have modified this to include the following Feature extractions
  * tiny image.
  * bag of sift
  * gist
  * bags of sifts + gist
  * gist
  
  
  And the following Classifiers,
  * nearest neighbour
  * support vector machine (linear)
  * primal_svm (quadratic)
  * primal_svm (linear)
  * libsvm (quadratic)

- **vocab_size_test_gist.txt** - testing results for vocab size with gist + bags of sift features and svm classifier
- **vocab_size_test.txt** - testing results for vocab size with bags of sift features and svm classifier
- **svmpredict.mexmaci64** - compiled svmpredict function for libsvm
- **svmtrain.mexmaci64** - compiled svmtrain function for libsvm
- **proj4_vocab_size_perf_test.m** - vocabulary size performance testing script
- **80.fig** - confusion matrix figure for 80% accuracy
- **vlfleat-0.9.20 2** - VLFeat compiled for OSX. Please feel free to modify the 'run' command in proj4.m and proj4_vocab_size_perf_test.m to run your own binary

#### The code submitted is primed to run with the following matrices. I have saved copies of these. Without loading these, the code in proj4 takes a long time to run for Bags of SIFTs + GIST feature extraction. SVM training time is minimal in comparison. 
- **test_image_gist_feats.mat** - Saved Matlab Matrix with GIST features for test images. If this file is present, proj4.m will load this file and use the data instead of recalculating all the GIST features again. This helps with the running time.
- **train_image_gist_feats.mat** - Same file as above but for training images
- **train_image_feats_fast_step_3_2000.mat** - Saved Matlab matrix with Bags of SIFTs features, extracted by using vl_dsift() at step size 3 and 2000 vocabulary size.
- **test_image_feats_fast_step_3_2000.mat** - Same as above for test images.
- **vocab_2000.mat** - Vocab.mat with 2000 size vocabulary.


