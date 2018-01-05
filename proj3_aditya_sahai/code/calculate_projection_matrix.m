% Projection Matrix Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by Henry Hu, Grady Williams, James Hays

% Returns the projection matrix for a given set of corresponding 2D and
% 3D points. 

% 'Points_2D' is nx2 matrix of 2D coordinate of points on the image
% 'Points_3D' is nx3 matrix of 3D coordinate of points in the world

% 'M' is the 3x4 projection matrix


function M = calculate_projection_matrix( Points_2D, Points_3D )

% To solve for the projection matrix. You need to set up a system of
% equations using the corresponding 2D and 3D points:

%                                                     [M11       [ u1
%                                                      M12         v1
%                                                      M13         .
%                                                      M14         .
%[ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1          M21         .
%  0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1          M22         .
%  .  .  .  . .  .  .  .    .     .      .          *  M23   =     .
%  Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn          M24         .
%  0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]        M31         .
%                                                      M32         un
%                                                      M33         vn ]

% Then you can solve this using least squares with the '\' operator or SVD.
% Notice you obtain 2 equations for each corresponding 2D and 3D point
% pair. To solve this, you need at least 6 point pairs.

%%%%%%%%%%%%%%%%
% Your code here
%%%%%%%%%%%%%%%%
% Lets form the A matrix
% we need 9 points from the points_3d matrix and 9 points from the
% points_2d matrix
sample3 = Points_3D(1:8, :);
sample2 = Points_2D(1:8, :);

a = zeros([16, 12]);
j = 1;
for i = 1:8
	% First Row
    a(j, 1:3) = sample3(i, 1:3);
    a(j, 4:8) = [1, 0, 0, 0, 0];
    a(j, 9:11) = -1 * sample2(i, 1) * sample3(i, 1:3);
    a(j, end) = -1 * sample2(i, 1);
    % Second Row
    a(j+1, 1:4) = zeros([1, 4]);
    a(j+1, 5:7) = sample3(i, 1:3);
    a(j+1, 8) = 1;
    a(j+1, 9:11) = -1 * sample2(i, 2) * sample3(i, 1:3);
    a(j+1, end) = -1 * sample2(i, 2);
    j = j + 2;
end

% Applying single value decomposition on A
[~, ~, V] = svd(a);
M = V(:, end);
M = reshape(M, [], 3)';

%Your total residual should be less than 1.

end

