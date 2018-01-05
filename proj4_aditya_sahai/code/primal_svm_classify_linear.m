function predicted_categories = primal_svm_classify(train_image_feats, train_labels, test_image_feats)
	global X
	categories = unique(train_labels);
	num_categories = length(categories);
    W = zeros(size(train_image_feats,2), num_categories);
    B = zeros(1, num_categories);
  	lambda = 0.001;
    X = train_image_feats;
	for i = 1:num_categories
	  Y = strcmp(categories{i, 1}, train_labels);
	  y = zeros(size(train_image_feats, 1), 1);
	  y(Y == 1) = 1;
	  y(Y == 0) = -1;
	  [W(:, i), B(:, i)] = primal_svm(1, y, lambda);
  end
  scores = W' * test_image_feats' + repmat(B', 1, size(test_image_feats, 1));
  [~, indices] = max(scores);
  predicted_categories = categories(indices);
end