function predicted_categories = libsvm_classify(train_image_feats, train_labels, test_image_feats)
	categories = unique(train_labels);
	num_categories = length(categories);
  W = zeros(size(train_image_feats, 2), num_categories);
  B = zeros(1, num_categories);
	for i = 1:num_categories
	  Y = strcmp(categories{i, 1}, train_labels);
	  y = zeros(size(train_image_feats, 1), 1);
	  y(Y == 1) = 1;
	  y(Y == 0) = -1;
    model = svmtrain(y, train_image_feats, '-s 0 -t 1 -g 0.001');
    w = model.SVs' * model.sv_coef;
    b = -model.rho;
    if (model.Label(1) == -1)
      w = -w; 
      b = -b;
    end
    W(:, i) = w;
    B(:, i) = b;
	end
	scores = W' * test_image_feats' + repmat(B', 1, size(test_image_feats, 1));
	[~, indices] = max(scores);
	predicted_categories = categories(indices);
end