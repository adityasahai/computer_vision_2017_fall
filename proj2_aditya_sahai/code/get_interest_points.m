% Local Feature Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% Returns a set of interest points for the input image

% 'image' can be grayscale or color, your choice.
% 'feature_width', in pixels, is the local feature width. It might be
%   useful in this function in order to (a) suppress boundary interest
%   points (where a feature wouldn't fit entirely in the image, anyway)
%   or (b) scale the image filters being used. Or you can ignore it.

% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
% 'confidence' is an nx1 vector indicating the strength of the interest
%   point. You might use this later or not.
% 'scale' and 'orientation' are nx1 vectors indicating the scale and
%   orientation of each interest point. These are OPTIONAL. By default you
%   do not need to make scale and orientation invariant local features.
function [x, y, orientation, confidence, scale] = get_interest_points(image, feature_width, image_orig)

% Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
% You can create additional interest point detector functions (e.g. MSER)
% for extra credit.

% If you're finding spurious interest point detections near the boundaries,
% it is safe to simply suppress the gradients / corners near the edges of
% the image.

% The lecture slides and textbook are a bit vague on how to do the
% non-maximum suppression once you've thresholded the cornerness score.
% You are free to experiment. Here are some helpful functions:
%  BWLABEL and the newer BWCONNCOMP will find connected components in 
% thresholded binary image. You could, for instance, take the maximum value
% within each component.
%  COLFILT can be used to run a max() operator on each sliding window. You
% could use this to ensure that every interest point is at a local maximum
% of cornerness.

    % apply multiple gaussians
	image_size = size(image);
	scale_factor = 0.5;
	gaussian_size = [15 15];
	image_cells = {3, 5};
    %begining with double sized image
	resized_image = image_orig;
    feature_width_bt = feature_width/2;
    sigma = 0.5;

	for i = 1:3
		for j = 1:5
%             if j == 1
%                 image_cells{i, j} = resized_image;
%             else
%                 % calculate filter
                gaussian = fspecial('gaussian', gaussian_size, sigma);
                gaussed_image = imfilter(resized_image, gaussian);
                image_cells{i, j} = gaussed_image;
                sigma = sigma * 2;
%             end
		end
		% resize image
		resized_image = imresize(resized_image, scale_factor, 'bilinear');
	end

    % Calculate Difference of Gaussians
    dog_cells = {3, 4};
    for i = 1:size(image_cells, 1)
        for j = 1:size(image_cells, 2)-1
        dog_cells{i, j} = image_cells{i, j} - image_cells{i, j+1};
        end
    end
    
    % Calculate local extremas in the target scale
    extrema_cells = {1, 4};
    threshold_low = -0.000009;
    threshold_high = 0.000009;
    for i = 1:4
        check_image = dog_cells{2, i};
        points = zeros(image_size);
        for j = (feature_width_bt + 1):(size(check_image, 1) - (feature_width_bt + 1))
            for k = (feature_width_bt + 1):(size(check_image, 2) - (feature_width_bt + 1))
                % check if pixel has zero value
                pixel = check_image(j, k);
                if pixel == 0 || (pixel > threshold_low && pixel < threshold_high)
                    continue;
                end
                
                % check max/min at same level
                window_val = check_image(j-1:j+1, k-1:k+1);
                if ~isExtrema(pixel, window_val, 0)
                    continue;
                end

%                 % Get upper scale matrix
%                 j_scale_up = ceil(j/scale_factor);
%                 k_scale_up = ceil(k/scale_factor);
%                 upper_scale_image = dog_cells{1, i};
%                 try
%                     window_val = upper_scale_image(j_scale_up-1:j_scale_up+1, ...
%                             k_scale_up-1:k_scale_up+1);
%                 catch E
%                     disp(E);
%                 end
%                 if ~isExtrema(pixel, window_val, -1)
%                     continue;
%                 end
                
                % Get lower scale matrix
                j_scale_down = ceil(j * scale_factor);
                k_scale_down = ceil(k * scale_factor);
                lower_scale_image = dog_cells{3, i};
                try
                    window_val = lower_scale_image(j_scale_down-1:j_scale_down+1, ...
                            k_scale_down-1:k_scale_down+1);
                catch E
                    disp(E);
                end
                    
                if ~isExtrema(pixel, window_val, -1)
                    continue
                end
                
                % pixel is extrema
                points(j, k) = 1;
            end
        end
        extrema_cells{1, i} = points;
    end
    
                
    keypoints = ones(image_size);
    for i = 1:4
        keypoints = keypoints & extrema_cells{1, i};
    end
    
    [y, x] = find(keypoints);
%     size(y)
    
    % find orientations for each keypoints
    orientation = zeros(size(y, 1));
    [image_grad_mag, image_grad] = imgradient(image);
    for i = 1:size(y, 1)
        orientation(i) = findOrientation(y(i), x(i), feature_width_bt, ...
                                image_grad, image_grad_mag);
    end
end

function output = isExtrema(val, window_val, level)
    min_output = true;
    max_output = true;
    % check for max
    for i = 1:size(window_val,1)
        if max_output == false
              break;
        end
        for j = 1:size(window_val, 2)
            if i == 2 && j == 2 && level == 0
               % Don't compare the point with it self on same level
                continue;
            end
            if val <= window_val(i, j)
                max_output = false;
                break;
            end
        end
    end
    % check for min
    for i = 1:size(window_val,1)
        if min_output == false
            break;
        end
        for j = 1:size(window_val, 2)
            if i == 2 && j == 2 && level == 0
                continue;
            end
            if val >= window_val(i, j)
                min_output = false;
                break;
            end
        end
    end
    output = min_output | max_output;
end

function orientation = findOrientation(y, x, feature_width_bt, image_gradient, image_grad_mag)
    % Image grid aroud the keypoint
    grid_grad = image_gradient(y - feature_width_bt:y+feature_width_bt, ...
                  x - feature_width_bt:x + feature_width_bt);
    grid_grad_mag = image_grad_mag(y - feature_width_bt:y+feature_width_bt, ...
                  x - feature_width_bt:x + feature_width_bt);              
    
    % Calculate dominant gradient in the grid
    histogram = zeros(1, 36);
    for i = 1:size(grid_grad, 1)
        for j = 1:size(grid_grad, 2)
            angle = grid_grad(i, j);
            % convert angle into 360
            angle = mod(angle+360, 360);
            index = floor(angle/360 * 36) + 1;
            if index > 36
                index = 36;
            end
            histogram(1, index) = histogram(1, index) + grid_grad_mag(i, j);
        end
    end
    [~, ind] = max(histogram(1, :));
    orientation = ind*10;
end


