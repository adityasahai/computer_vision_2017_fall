run('vlfeat-0.9.20 2/toolbox/vl_setup.m');
% run('vlfeat-0.9.20-bin\vlfeat-0.9.20\toolbox\vl_setup.m')

data_path = '../data/';

categories = {'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office', ...
       'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street', ...
       'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest'};

abbr_categories = {'Kit', 'Sto', 'Bed', 'Liv', 'Off', 'Ind', 'Sub', ...
    'Cty', 'Bld', 'St', 'HW', 'OC', 'Cst', 'Mnt', 'For'};

num_train_per_cat = 100; 

fprintf('Getting paths and labels for all train and test data\n')
[train_image_paths, test_image_paths, train_labels, test_labels] = ...
    get_image_paths(data_path, categories, num_train_per_cat);

fprintf('Using bag of sift + gist representation for images\n');

  for vocab_size = [1 10 100 200 400 500 1000 2000 5000 10000]
        fprintf('Using Vocab Size - %d\n', vocab_size);
        vocab = build_vocabulary(train_image_paths, vocab_size);
        save('vocab.mat', 'vocab')
        % Bag of SIFT
        train_image_feats = get_bags_of_sifts(train_image_paths);
        test_image_feats  = get_bags_of_sifts(test_image_paths);
        % GIST
        train_image_gist_feats = get_gist(train_image_paths);
        test_image_gist_feats = get_gist(test_image_paths);
        % Concat
        train_image_feats = [train_image_feats train_image_gist_feats];
        test_image_feats = [test_image_feats test_image_gist_feats];

        fprintf('Using support vector machine classifier to predict test set categories\n');

        predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats);

        create_results_webpage( train_image_paths, ...
                        test_image_paths, ...
                        train_labels, ...
                        test_labels, ...
                        categories, ...
                        abbr_categories, ...
                        predicted_categories)
  end
  
  fprintf('Using bag of sift representation for images\n');
  
  for vocab_size = [1 10 100 200 400 500 1000 2000 5000 10000]
        fprintf('Using Vocab Size - %d\n', vocab_size);
        vocab = build_vocabulary(train_image_paths, vocab_size);
        save('vocab.mat', 'vocab')
        % Bag of SIFT
        train_image_feats = get_bags_of_sifts(train_image_paths);
        test_image_feats  = get_bags_of_sifts(test_image_paths);
        % GIST
%         train_image_gist_feats = get_gist(train_image_paths);
%         test_image_gist_feats = get_gist(test_image_paths);
%         % Concat
%         train_image_feats = [train_image_feats train_image_gist_feats];
%         test_image_feats = [test_image_feats test_image_gist_feats];

        fprintf('Using support vector machine classifier to predict test set categories\n');

        predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats);

        create_results_webpage( train_image_paths, ...
                        test_image_paths, ...
                        train_labels, ...
                        test_labels, ...
                        categories, ...
                        abbr_categories, ...
                        predicted_categories)
  end