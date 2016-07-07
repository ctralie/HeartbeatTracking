%Programmer: Chris Tralie
%Purpose: To make a synthetic video of a patch undergoing affine motions
%and to test out affine RANSAC and warping
W = 400;

rng(100);
X = makeRandomWalkCurve(400, 10, 3);
Y = smoothCurve(X, 20);
Y = Y(20:end-20, :);
Y = bsxfun(@minus, Y, min(Y, [], 1));
Y = bsxfun(@times, Y, 1./max(Y, [], 1));
scale = Y(:, 3)/2 + 0.5;
Y = Y(:, 1:2) - 0.5;
Y = W/2 + Y*W/3;

P = imread('pic.jpg');
P = imresize(P, [100, 100]);
P = double(P)/255.0;

%Figure out rotation angle
dY = Y(2:end, :) - Y(1:end-1, :);
thetas = atan2(dY(:, 2), dY(:, 1))*180/pi;

frames = cell(size(dY, 1), 1);
for ii = 1:size(dY, 1)
    I = zeros(500, 500, 3);
    PP = imrotate(P, thetas(ii));
    PP = imresize(PP, scale(ii));
    dim = round(size(PP, 1)/2);
    ub = int32(Y(ii, 1) - dim);
    ul = int32(Y(ii, 2) - dim);
    I(ub:ub+size(PP, 1)-1, ul:ul+size(PP, 2)-1, :) = PP;
    frames{ii} = I;
end

refFrame = frames{1};
initialPoints = detectMinEigenFeatures(rgb2gray(refFrame));%, 'ROI', roi);
initialPoints = initialPoints.Location;
tracker = vision.PointTracker('MaxBidirectionalError', 1);
initialize(tracker, initialPoints, refFrame);

outputView = imref2d(size(refFrame));
for ii = 1:length(frames)
    frame = frames{ii};
    [points, validity] = step(tracker, frame);
    [tform, inlierPtsDistored, inlierPtsOriginal] = estimateGeometricTransform(points, initialPoints, 'affine');
    J = imwarp(frame, tform, 'OutputView', outputView);
    subplot(221);
    imagesc(frame);
    title('Initial Frame');
    subplot(222);
    imagesc(J);
    title('Warped Frame');
    subplot(223);
    showMatchedFeatures(refFrame, frame, initialPoints, points);
    print('-dpng', '-r150', sprintf('%i.png', ii));
end