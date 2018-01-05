function features_neg = get_random_negative_features2(non_face_scn_path,...
                            feature_params, num_samples)
	image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));
	num_images = length(image_files);
  feature_size = (feature_params.template_size /...
                    feature_params.hog_cell_size)^2 * 31;
	features_neg = zeros(num_samples, feature_size);
	scales = 1.2:-0.04:0.5;
  num_scales = size(scales, 2);
  % Number of samples per image per scale
  num_samples_pips = ceil(num_samples / (num_scales * num_images));
	samples_collected = 0;
  w = feature_params.template_size / ...
            feature_params.hog_cell_size;
  for i = 1:num_images
    if samples_collected >= num_samples
      break;
    end
    img = imread(fullfile(non_face_scn_path, image_files(i).name));
    img = single(img)/255;
    if (size(img, 3) > 1)
      img = rgb2gray(img);
    end
    for j = 1:num_scales
      scale = scales(j);
      if scale ~= 1
        img_prime = imresize(img, scale);
      else
        img_prime = img;
      end
      [n, m] = size(img_prime);
      if n > feature_params.template_size &&...
            m > feature_params.template_size
        % Calculate hog fetures for this image at this scale
        hog = vl_hog(img_prime, feature_params.hog_cell_size);
        rows = size(hog, 1);
        cols = size(hog, 2);
        for k = 1:num_samples_pips
          if rows <= w || cols <= w
            continue;
          end
          n_w = randi([1, rows-w]);
          m_w = randi([1, cols-w]);
          features = hog(n_w:n_w+w-1, m_w:m_w+w-1, :);
          samples_collected = samples_collected + 1;
          features_neg(samples_collected, :) = reshape(features, ...
                                                      [feature_size, 1]);
        end
      end
    end
  end
  features_neg = features_neg(1:num_samples, :);
end