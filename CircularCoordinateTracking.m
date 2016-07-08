% load image data
%path = 'GroundTruth/1';
%filename = sprintf('%s/00001.jpg', path);
path = 'MeColorRealSense'; NFrames = 400;
filename = sprintf('%s/1.png', path);
refFrame = imread(filename);

lm = getKeypointsDlib(filename);
regions = getDlibFaceRegions(lm);

patches = getRegionPatchIndices(regions, size(refFrame), 20);

% for ii = 1:size(patches, 1)
%     refFrame(patches(ii, :, 1)) = 0;
% end
% imagesc(refFrame);

I = zeros(NFrames, numel(refFrame));
for ii = 1:NFrames
    fprintf(1, 'Loading frame %i...\n', ii);
    %filename = sprintf('%s/%.5i.jpg', path, ii);
    filename = sprintf('%s/%i.png', path, ii);
    frame = imread(filename);
    I(ii, :) = frame(:);
end

%Process in blocks
BlockLen = 150;
BlockHop = 10;
W = 30;
Kappa = 0.2;

scoresImage = refFrame;%zeros(size(refFrame));

for pp = 1:size(patches, 1)
    fprintf(1, 'Doing patch %i...\n', pp);
    tic;
    J = I(1:BlockLen, patches(pp, :, :));
    J = reshape(J, [size(J, 1), size(patches, 2), size(patches, 3)]);
    %J = J(:, :, 2);%Get green channel only
    J = squeeze(mean(J, 2));
    J = getSmoothedDerivative(J, 20);
    D = pdist2(J, J);

    %TODO: Test normalization?
    %Convolve down diagonals
    N = size(D, 1);
    ND = N-W+1;
    DD = zeros(ND, ND);
    for ii = 1:N-W+1
        b = diag(D, ii-1);
        b2 = cumsum(b);
        b2 = b2(W:end) - [0; cumsum(b(1:end-W))];
        DD(ii + (0:length(b2)-1)*(ND+1)) = b2;
    end
    DD = DD + DD';
    DD(1:ND+1:end) = 0.5*DD(1:ND+1:end);%Main diagonal was counted twice
    
    %Get graph laplacian
    A = groundTruthKNN(DD, round(Kappa*size(DD, 2)));
    A = A.*groundTruthKNN(DD', round(Kappa*size(DD, 1)))';
    A = A - speye(size(A, 1));
    L = spdiags(sum(A, 2), 0, size(A, 1), size(A, 2)) - A;
    [V, lambda] = eigs(L, 3, 0);
    %Center and RMS Normalize eigenvectors
    V = V(:, 1:2);
    V = bsxfun(@minus, mean(V, 1), V);
    VSum = sum(V(:).^2);
    if VSum == 0
        VSum = 1;
    end
    V = V*sqrt(size(V, 1)/VSum);
    %Compute mean distance from circle
    ds = sqrt(sum(V.^2, 2));
    score = sum(abs(ds-1))/size(V, 1);
    %Get unwrapped circular coordinates
    theta = unwrap(atan2(V(:, 2), V(:, 1)));
    toc;
    
    scoresImage(patches(pp, :, 1)) = 255*exp(-score/0.2);
    scoresImage(patches(pp, :, 2)) = 0;
    scoresImage(patches(pp, :, 3)) = 0;
    
%     clf;
%     subplot(231);
%     imagesc(D);
%     subplot(232);
%     imagesc(DD);
%     subplot(233);
%     imagesc(A);
%     subplot(234);
%     plot(V(:, 1), V(:, 2), '.');
%     title('Eigenvectors');
%     subplot(235);
%     plot(mod(theta, 2*pi));
%     title(sprintf('Circ Coords Score %g', score));
%     subplot(236);
%     F = refFrame;
%     F(patches(pp, :, 1)) = 0;
%     imagesc(F);
%     print('-dpng', '-r100', sprintf('%i.png', pp));
end

imagesc(scoresImage);