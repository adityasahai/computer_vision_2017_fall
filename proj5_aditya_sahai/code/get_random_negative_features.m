% Starter code prepared by James Hays for CS 4476, Georgia Tech
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale because the positive training data is only available in
% grayscale. For best performance, you should sample random negative
% examples at multiple scales.

function features_neg = get_random_negative_features(non_face_scn_path,...
                            feature_params, num_samples)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (default 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.

% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray
	image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));
	num_images = length(image_files);
  feature_size = (feature_params.template_size /...
                    feature_params.hog_cell_size)^2 * 31;
	features_neg = zeros(num_samples, feature_size);
	scales = [1.5, 1, 0.9, 0.8, 1.2, 0.7, 0.6, 1.3];
  num_scales = size(scales, 2);
  template_size = feature_params.template_size;
	samples_collected = 0;
  window = feature_params.template_size / ...
            feature_params.hog_cell_size;
	while samples_collected < num_samples
		% pick random image
		image_id = randi([1, num_images]);
		img = imread(fullfile(non_face_scn_path, image_files(image_id).name));
		img = single(img)/255;
		if (size(img, 3) > 1)
			img = rgb2gray(img);
		end
		% pick random scale
    img_prime = img;
    scale = scales(randi([1, num_scales]));
		if scale ~= 1
			img_prime = imresize(img, scale);
    end
    [n, m] = size(img_prime);
    if n > template_size && m > template_size
      hog = vl_hog(img_prime, feature_params.hog_cell_size);
      rows = size(hog, 1);
      cols = size(hog, 2);
      if rows <= window || cols <= window
        continue;
      end
      n_w = randi([1, rows-window]);
      m_w = randi([1, cols-window]);
      features = hog(n_w:n_w + window -1, m_w:m_w + window - 1, :);
      samples_collected = samples_collected + 1;
      features_neg(samples_collected, :) = reshape(features,...
                                                  [feature_size, 1]);
    end
	end
end