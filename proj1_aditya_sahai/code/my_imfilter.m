% Improved version of the my_imfilter(). This particular implementation has
% been inspired by the following source 
% https://blogs.mathworks.com/steve/2006/10/04/separable-convolution/
% http://blogs.mathworks.com/steve/2006/11/28/separable-convolution-part-2/
% Read more about this in the HTML report
function output = my_imfilter(image, filter)
    if (rank(filter) ~= 1)
        output = priv_filter(image, filter);
        % Comment out the other lines in this function
        % except the one above to use the base level of the function
    else
        [U,S,V] = svd(filter);
        v = U(:, 1) * sqrt(S(1,1));
        h = V(:, 1)' * sqrt(S(1,1));
        mat1 = priv_filter(image, v);
        output = priv_filter(mat1, h);
    end

% Baseline implementation of my_imfilter()
function output = priv_filter(image, filter)
% Check filter
if (~ismatrix(filter))
    error('my_imfilter: The filter should be a valid matrix');
end

% Check image
[~, ~, img_channels] = size(image);
if (img_channels ~= 3 && img_channels ~= 1)
   error('my_imfilter: The image should be a valid image'); 
end

% save dimensions of the filter
[filter_rows, filter_cols] = size(filter);

% check filter dimensions, should be odd
if (mod(filter_rows, 2) == 0 || mod(filter_cols, 2) == 0)
    error('my_imfilter: The rows and columns of the filter should be odd');
end
    
k_row = (filter_rows - 1)/2;
k_col = (filter_cols - 1)/2;

% pad image with zeros 
padded_matrix = padarray(image, [k_row k_col]);
% save padded image size
[rows, cols, channels] = size(padded_matrix);

new_image_matrix = zeros(size(padded_matrix));
for c = 1:channels
    for i = k_row+1:rows-(k_row)
        for j = k_col+1:cols-(k_col)
            new_image_matrix(i, j, c) = sum(dot(padded_matrix(i - k_row:i + k_row, j - k_col:j + k_col, c), filter));
        end
    end
end
output = new_image_matrix(k_row+1:end-k_row, k_col+1:end-k_col, :);




