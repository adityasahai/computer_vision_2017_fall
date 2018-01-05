% RANSAC Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by Henry Hu

% Find the best fundamental matrix using RANSAC on potentially matching
% points

% 'matches_a' and 'matches_b' are the Nx2 coordinates of the possibly
% matching points from pic_a and pic_b. Each row is a correspondence (e.g.
% row 42 of matches_a is a point that corresponds to row 42 of matches_b.

% 'Best_Fmatrix' is the 3x3 fundamental matrix
% 'inliers_a' and 'inliers_b' are the Mx2 corresponding points (some subset
% of 'matches_a' and 'matches_b') that are inliers with respect to
% Best_Fmatrix.

% For this section, use RANSAC to find the best fundamental matrix by
% randomly sample interest points. You would reuse
% estimate_fundamental_matrix() from part 2 of this assignment.

% If you are trying to produce an uncluttered visualization of epipolar
% lines, you may want to return no more than 30 points for either left or
% right images.

% % From Wikipedia: https://en.wikipedia.org/wiki/Random_sample_consensus
% % Given:
% %     data – a set of observed data points
% %     model – a model that can be fitted to data points
% %     n – the minimum number of data values required to fit the model
% %     k – the maximum number of iterations allowed in the algorithm
% %     t – a threshold value for determining when a data point fits a model
% %     d – the number of close data values required to assert that a model fits well to data
% % 
% % Return:
% %     bestfit – model parameters which best fit the data (or nul if no good model is found)
% % 
% % iterations = 0
% % bestfit = nul
% % besterr = something really large
% % while iterations < k {
% %     maybeinliers = n randomly selected values from data
% %     maybemodel = model parameters fitted to maybeinliers
% %     alsoinliers = empty set
% %     for every point in data not in maybeinliers {
% %         if point fits maybemodel with an error smaller than t
% %              add point to alsoinliers
% %     }
% %     if the number of elements in alsoinliers is > d {
% %         % this implies that we may have found a good model
% %         % now test how good it is
% %         bettermodel = model parameters fitted to all points in maybeinliers and alsoinliers
% %         thiserr = a measure of how well model fits these points
% %         if thiserr < besterr {
% %             bestfit = bettermodel
% %             besterr = thiserr
% %         }
% %     }
% %     increment iterations
% % }
% % return bestfit

function [ Best_Fmatrix, inliers_a, inliers_b] = ransac_fundamental_matrix(matches_a, matches_b)


%%%%%%%%%%%%%%%%
% Your code here
%%%%%%%%%%%%%%%%

num_interations = 5000;
F_best_fit = zeros(3);
sample_range = size(matches_b, 1);
threshold = 0.0003;
inliers = zeros([size(matches_a, 1) 1]);
fitting_threshold = 0.5 * size(matches_a, 1); % Matching atleast half the points
for i = 1:num_interations
    % select 8 random points
    sample_idx = randsample(sample_range, 8);
    sample_a = matches_a(sample_idx, :);
    sample_b = matches_b(sample_idx, :);
    % find fundamental matrix for sample points
    F_found = estimate_fundamental_matrix(sample_a, sample_b);
    % Check other points in matches which fit the model
    for j = 1:size(matches_a, 1)
        h_a = [matches_a(j, :) 1];
        h_b = [matches_b(j, :) 1];
        error = h_a * F_found * h_b';
        if abs(error) < threshold
            inliers(j) = 1;
        end
        count_inliers = sum(inliers);
        if count_inliers >= fitting_threshold
            % We have found a good model
            fitting_threshold = count_inliers;
            F_best_fit = F_found;
            % If all points are inliers, then stop running
            if fitting_threshold == size(matches_a, 1)
              break;
            end
        end
    end
end
% disp(fitting_threshold);

Best_Fmatrix = F_best_fit;
inliers_a = matches_a(find(inliers), :);
inliers_b = matches_b(find(inliers), :);
end

