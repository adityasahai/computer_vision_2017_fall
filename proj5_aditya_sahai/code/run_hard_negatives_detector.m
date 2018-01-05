function hard_negative_features = .... 
    run_hard_negatives_detector(test_scn_path, w, b, feature_params,...
                                  feature_size, num_negatives, threshold)

  test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

  scale_factor = 0.9;
  
  hard_negative_features = zeros(0, feature_size);
  
  for i = 1:length(test_scenes)
      img = imread( fullfile( test_scn_path, test_scenes(i).name ));
      img = single(img)/255;
      if(size(img,3) > 1)
          img = rgb2gray(img);
      end
      img_prime = img;
      s = 1;
      while size(img_prime, 1) > feature_params.template_size &&...
          size(img_prime, 2) > feature_params.template_size
        if s ~= 1
          img_prime = imresize(img_prime, s);
        end
        window = feature_params.template_size / ...
                    feature_params.hog_cell_size;
        hog = vl_hog(img_prime, feature_params.hog_cell_size);
        rows = size(hog, 1) - window;
        cols = size(hog, 2) - window;
        for k = 1:rows 
          for j = 1:cols
            feature_set = hog(k:k+window-1, j:j+window-1,:);
            features = feature_set(:);
            confidence = w'*features + b;
            if (confidence > threshold)
              hard_negative_features = [hard_negative_features;...
                                          features'];
            end
          end
        end
        s = s * scale_factor;
      end
  end
  rows = size(hard_negative_features, 1);
  if rows > num_negatives
    hard_negative_features = hard_negative_features(1:num_negatives, :);
  end
end




