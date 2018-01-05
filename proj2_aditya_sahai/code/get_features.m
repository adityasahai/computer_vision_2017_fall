% Local Feature Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% Returns a set of feature descriptors for a given set of interest points. 

% 'image' can be grayscale or color, your choice.
% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
%   The local features should be centered at x and y.
% 'feature_width', in pixels, is the local feature width. You can assume
%   that feature_width will be a multiple of 4 (i.e. every cell of your
%   local SIFT-like feature will have an integer width and height).
% If you want to detect and describe features at multiple scales or
% particular orientations you can add input arguments.

% 'features' is the array of computed features. It should have the
%   following size: [length(x) x feature dimensionality] (e.g. 128 for
%   standard SIFT)

function [features] = get_features(image, x, y, feature_width, orientation)

% To start with, you might want to simply use normalized patches as your
% local feature. This is very simple to code and works OK. However, to get
% full credit you will need to implement the more effective SIFT descriptor
% (See Szeliski 4.1.2 or the original publications at
% http://www.cs.ubc.ca/~lowe/keypoints/)

% Your implementation does not need to exactly match the SIFT reference.
% Here are the key properties your (baseline) descriptor should have:
%  (1) a 4x4 grid of cells, each feature_width/4. 'cell' in this context
%    nothing to do with the Matlab data structue of cell(). It is simply
%    the terminology used in the feature literature to describe the spatial
%    bins where gradient distributions will be described.
%  (2) each cell should have a histogram of the local distribution of
%    gradients in 8 orientations. Appending these histograms together will
%    give you 4x4 x 8 = 128 dimensions.
%  (3) Each feature vector should be normalized to unit length
%
% You do not need to perform the interpolation in which each gradient
% measurement contributes to multiple orientation bins in multiple cells
% As described in Szeliski, a single gradient measurement creates a
% weighted contribution to the 4 nearest cells and the 2 nearest
% orientation bins within each cell, for 8 total contributions. This type
% of interpolation probably will help, though.

% You do not have to explicitly compute the gradient orientation at each
% pixel (although you are free to do so). You can instead filter with
% oriented filters (e.g. a filter that responds to edges with a specific
% orientation). All of your SIFT-like feature can be constructed entirely
% from filtering fairly quickly in this way.

% You do not need to do the normalize -> threshold -> normalize again
% operation as detailed in Szeliski and the SIFT paper. It can help, though.

% Another simple trick which can help is to raise each element of the final
% feature vector to some power that is less than one.

% Placeholder that you can delete. Empty features.
    features = zeros(size(x,1), 128);
    
    fwbt = feature_width/2;
    
    % Calculate Gradient of the image
    [grad_x, grad_y] = imgradientxy(image);
    [grad_mag, ~] = imgradient(image);
    theta = atan2(grad_y, grad_x);
    
    for i = 1:size(x, 1)
            % Rotate Image as per orientation
            angle = orientation(i);
            tform = affine2d([cosd(angle) -sind(angle) 0; sind(angle) cosd(angle) 0; 0 0 1]);
            [rotated_image, ref] = imwarp(image, tform);
            % Coordinates of keypoints after rotation
            % Coordinate manipulation code referred from
            % https://stackoverflow.com/questions/33127640/how-to-find-a-point-on-image-after-rotation
            [x1, y1] = transformPointsForward(tform, x(i), y(i));
            x1 = x1 - ref.XWorldLimits(1);
            y1 = y1 - ref.YWorldLimits(1);
            x_index_low = ceil(x1) - floor(fwbt);
            x_index_high = ceil(x1) + floor(fwbt) - 1;
            y_index_low = ceil(y1) - floor(fwbt);
            y_index_high = ceil(y1) + floor(fwbt)- 1;
            
            % Coordinate window for gradient and gradient magnitude
            % matrices
            x_index_low_theta = floor(x(i)) - floor(fwbt);
            x_index_high_theta = floor(x(i)) + floor(fwbt) - 1;
            y_index_low_theta = floor(y(i)) - floor(fwbt);
            y_index_high_theta = floor(y(i)) + floor(fwbt)- 1;
            
            try
                feature_box = rotated_image(y_index_low:y_index_high, x_index_low:x_index_high);
            catch E
                disp(E.message);
                disp(x_index_low);
                disp(x_index_high);
                disp(y_index_low);
                disp(y_index_high);
                continue;
            end
            features(i, :) = extractCustomFeatures(feature_box, ...
                theta(y_index_low_theta:y_index_high_theta, ...
                            x_index_low_theta:x_index_high_theta), ...
                  grad_mag(y_index_low_theta:y_index_high_theta, ...
                  x_index_low_theta:x_index_high_theta));
    end    
end
        
function output = extractCustomFeatures(feature_box, theta, mag)
    feature = zeros(1, 128);
    image_cell = mat2cell(feature_box, [4, 4, 4, 4], [4, 4, 4, 4]);
    theta_cell = mat2cell(theta, [4, 4, 4, 4], [4, 4, 4, 4]);
    mag_cell = mat2cell(mag, [4, 4, 4, 4], [4, 4, 4, 4]);
    
    k = 1;
    for i = 1:size(image_cell(:))
       angles = theta_cell(i);
       angles = angles{:};
       mags = mag_cell(i);
       mags = mags{:};
       histogram = zeros(1, 8);
       for j = 1:size(angles(:))
           % Angles are in radians
           norm_angle = mod(angles(j) + 2*pi, 2*pi);
           index = floor(norm_angle/(2*pi) * 8) + 1;
           if index > 8
               index = 8;
           end
           histogram(index) = histogram(index) + mags(j);
       end
       feature(k:k+7) = histogram(:);
       k = k + 8;
    end
    output = feature;
end




% [features, ~] = extractFeatures(image, [x y]);

%     features = zeros(size(x,1), 128);
%     num_points = size(x, 1);
%     for i = 1:num_points
%         x_index_low = floor(x(i)) - 8;
%         x_index_high = floor(x(i)) + 7;
%         y_index_low = floor(y(i)) - 8;
%         y_index_high = floor(y(i)) + 7;
%         try
%             feature_box = image(y_index_low:y_index_high, x_index_low:x_index_high);
%         catch E
%             disp(E.message);
%             continue;
%         end
%         [r, c] = size(feature_box);
%         % Apply Gaussian to feature box
%         gaussian = fspecial('gaussian', [3 3], 2);
%         fb_gaussed = imfilter(feature_box, gaussian);
%         weighted_gradient_image = gradientweight(fb_gaussed);
%         [~, gradient] = imgradient(fb_gaussed);
%         histogram = features(i, :);
%         bin_id = 1;
%         for j = 1:feature_width/4:r
%             for k = 1:feature_width/4:c
%                 small_box = weighted_gradient_image(j:j+3, k:k+3);
%                 small_gradient_box = gradient(j:j+3, k:k+3);
%                 histogram(bin_id:bin_id+7) = putinbin(small_box(:), small_gradient_box(:), histogram(bin_id:bin_id+7));
%                 bin_id = bin_id + 8;
%             end
%         end
%         features(i, :) = histogram;
%     end
% end
% 
% function output = putinbin(small_box, gradient, histogram)
%     % The histogram is [8 1] vector
%     gradient = mod(gradient+360, 360);
%     for i = 1:size(gradient, 1)
%         index = floor(gradient(i)/360 * 8) + 1;
%         if (index > 8)
%             index = 8;
%         end
%         histogram(index) = histogram(index) + small_box(i);
%     end
%     output = histogram;
% end







