% Fundamental Matrix Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by Henry Hu

% Returns the camera center matrix for a given projection matrix

% 'Points_a' is nx2 matrix of 2D coordinate of points on Image A
% 'Points_b' is nx2 matrix of 2D coordinate of points on Image B
% 'F_matrix' is 3x3 fundamental matrix

% Try to implement this function as efficiently as possible. It will be
% called repeatly for part III of the project

function [ F_matrix ] = estimate_fundamental_matrix(Points_a,Points_b)

%%%%%%%%%%%%%%%%
% Your code here
%%%%%%%%%%%%%%%%

sample_a = Points_a(1:8, :);
sample_b = Points_b(1:8, :);

% Normalize coordinates based on Hartley's method
[sample_a_norm, T_a] = normalize_coordinates(sample_a);
[sample_b_norm, T_b] = normalize_coordinates(sample_b);

a = zeros([8, 9]);

for i = 1:8
    x = sample_a_norm(i, 1);
    y = sample_a_norm(i, 2);
    x_prime = sample_b_norm(i, 1);
    y_prime = sample_b_norm(i, 2);
    
    a(i,1) = x*x_prime;
    a(i,2) = y*x_prime;
    a(i,3) = x_prime;
    a(i,4) = x*y_prime;
    a(i,5) = y*y_prime;
    a(i,6) = y_prime;
    a(i,7) = x;
    a(i,8) = y;
    a(i,9) = 1;
end

% Applying Single value decomposition
[~, ~, V] = svd(a);
F = V(:, end);
F_norm = reshape(F, [3 3])';

F_matrix = T_b'*F_norm*T_a;

% Applying Singularity Constraint
[U1, S1, V1] = svd(F_matrix);
S1(3, 3) = 0;
F_matrix = U1 * S1 * V1';

        
end

function [ norm_mat, transformation_mat ] = normalize_coordinates(points)
  num_points = size(points, 1);
  x_centroid = sum(points(:, 1)) / num_points;
  y_centroid = sum(points(:, 2)) / num_points;
  
  s = 0;
  for i = 1:num_points
    s = s + ((points(i, 1) - x_centroid)^2 + (points(i, 2) - y_centroid)^2);
  end
  s = (s/(2*num_points))^(0.5);
  norm_mat = zeros(size(points));
  norm_mat(:, 1) = (points(:, 1) - x_centroid) / s;
  norm_mat(:, 2) = (points(:, 2) - y_centroid) / s;
  
  transformation_mat = zeros(3);
  transformation_mat(1,1) = 1/s;
  transformation_mat(2,2) = 1/s;
  transformation_mat(3,3) = 1;
  transformation_mat(1,3) = -1 * x_centroid / s;
  transformation_mat(2,3) = -1 * y_centroid / s;
end
