function gist = get_gist(image_paths)
%{
clear param
param.imageSize = [256 256]; % set a normalized image size
param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale (from HF to LF)
param.numberBlocks = 4;
param.fc_prefilt = 4;

% Pre-allocate gist:
Nfeatures = sum(param.orientationsPerScale)*param.numberBlocks^2;
gist = zeros([Nimages Nfeatures]); 

% Load first image and compute gist:
img = imread(file{1});
[gist(1, :), param] = LMgist(img, '', param); % first call
% Loop:
for i = 2:Nimages
   img = imread(file{i});
   gist(i, :) = LMgist(img, '', param); % the next calls will be faster
end
%}
  fprintf('Calculating gist features');
	clear param
	param.imageSize = [256 256];
	param.orientationsPerScale = [8 8 8 8];
	param.numberBlocks = 4;
	param.fc_prefilt = 4;

	Nimages = size(image_paths, 1);

	% Pre Allocate Gist
	Nfeatures = sum(param.orientationsPerScale) * param.numberBlocks^2;
	gist = zeros([Nimages Nfeatures]);

	%Load first image and compute gist
	img = imread(image_paths{1, 1});
	img = im2single(img); % Don't know if this is necessary
	[gist(1, :), param] = LMgist(img, '', param);

	for i = 2:Nimages
		img = imread(image_paths{i});
		gist(i, :) = LMgist(img, '', param);
	end
	% Normalize features
	gist = gist ./ norm(gist);
end