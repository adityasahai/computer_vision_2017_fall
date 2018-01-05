function predicted_categories = primal_svm_classify(train_image_feats, train_labels, test_image_feats)
	global K X
	categories = unique(train_labels);
	num_categories = length(categories);
    W = zeros(size(train_image_feats,1), num_categories);
    B = zeros(1, num_categories);
  	lambda = 1;
    opt.cg = 1;
    X = train_image_feats;
    K = (1 + train_image_feats * train_image_feats')^2; % Quadratic Kernel
	for i = 1:num_categories
	  Y = strcmp(categories{i, 1}, train_labels);
	  y = zeros(size(train_image_feats, 1), 1);
	  y(Y == 1) = 1;
	  y(Y == 0) = -1;
	  [W(:, i), B(:, i)] = primal_svm(0, y, lambda, opt);
  end
  test_kernel =  (1 + train_image_feats * test_image_feats')^2;
	scores = test_kernel*W + repmat(B', 1, size(test_image_feats, 1))';
	[~, indices] = max(scores');
	predicted_categories = categories(indices);
end