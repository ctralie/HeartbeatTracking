% load image data
% Fs = 30;
% path = 'GroundTruth/1';
% NFrames = 400;

% Fs = 30;
% path = 'GroundTruth/4';
% NFrames = 900;
% filename = sprintf('%s/0891.jpg', path);
% startFrame = 890;
% groundTruthMean = 71.2081;

Fs = 30;
path = 'GroundTruth/3';
NFrames = 900;
filename = sprintf('%s/1211.jpg', path);
startFrame = 1210;
groundTruthMean = 61.4943;

% Fs = 40; %Sample rate
% path = 'MeColorRealSense'; 
% NFrames = 400;
% filename = sprintf('%s/1.png', path);
% startFrame = 0;


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
    filename = sprintf('%s/%.4i.jpg', path, ii+startFrame);
    %filename = sprintf('%s/%i.png', path, ii);
    frame = imread(filename);
    I(ii, :) = frame(:);
end

%Process in blocks
BlockLen = 180;
%BlockLen= 180;
BlockHop = 40;
W = 15;
NBlocks = floor((NFrames-BlockLen)/BlockHop+1);

Kappa = 0.3;

circScoresImage = zeros(size(refFrame));
gtScoresImage = zeros(size(refFrame));

rates = linspace(30, 200, 1000);
scores = 0*rates;
AllRates = [];

for kk = 1:NBlocks
    fprintf(1, 'Doing Block %i of %i...\n', kk, NBlocks);
    tic;
    for pp = 1:size(patches, 1)
        J = I(BlockHop*(kk-1)+(1:BlockLen), patches(pp, :, :));
        J = reshape(J, [size(J, 1), size(patches, 2), size(patches, 3)]);
        %J = J(:, :, 2);%Get green channel only
        J = squeeze(mean(J, 2));
        J = getSmoothedDerivative(J, 20);
        J = getSmoothedDerivative(J, 20);
        JOrig = J;
        D = pdist2(J, J);
        
        J = getDelayEmbedding(J, W);
        J = bsxfun(@minus, mean(J, 1), J);
        J = bsxfun(@times, 1./sqrt(sum(J.^2, 2)), J);
        DD = pdist2(J, J);
        
        
        
%         %TODO: Test normalization?
%         %Convolve down diagonals
%         
%         N = size(D, 1);
%         ND = N-W+1;
%         DD = zeros(ND, ND);
%         for ii = 1:N-W+1
%             b = diag(D, ii-1);
%             b2 = cumsum(b);
%             b2 = b2(W:end) - [0; cumsum(b(1:end-W))];
%             DD(ii + (0:length(b2)-1)*(ND+1)) = b2;
%         end
%         DD = DD + DD';
%         DD(1:ND+1:end) = 0.5*DD(1:ND+1:end);%Main diagonal was counted twice

        %Get graph laplacian
        A = groundTruthKNN(DD, round(Kappa*size(DD, 2)));
        A = A.*groundTruthKNN(DD', round(Kappa*size(DD, 1)))';
        A = A - speye(size(A, 1));
        L = spdiags(sum(A, 2), 0, size(A, 1), size(A, 2)) - A;
        
        %Get eigenvectors
        V = zeros(size(L, 1), 3);
        try
            [V, lambda] = eigs(L, 3, 0);
        catch ME
            continue
        end
        
        %Center and RMS Normalize eigenvectors
        V = V(:, 1:2);
        V = bsxfun(@minus, median(V, 1), V);
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
        deriv = abs(getSmoothedDerivative((Fs*60/(2*pi))*theta(:), 20));

        scoreExp = exp(-score/0.1);
        circScoresImage(patches(pp, :, 1)) = circScoresImage(patches(pp, :, 1)) + scoreExp;
        circScoresImage(patches(pp, :, 2)) = 0;
        circScoresImage(patches(pp, :, 3)) = 0;
        
        gtScore = exp(-(groundTruthMean-rate)^2/(2^2));
        gtScoresImage(patches(pp, :, 1)) = gtScoresImage(patches(pp, :, 1)) + gtScore;
        gtScoresImage(patches(pp, :, 2)) = 0;
        gtScoresImage(patches(pp, :, 3)) = 0;
        
        rate = abs((theta(end)-theta(1))/(length(theta)-1));
        rate = rate*Fs*60/(2*pi);
        AllRates(end+1) = median(deriv);
        scores = scores + scoreExp*exp(-(rate-rates).^2/(2*2^2));

        clf;
        
        subplot(331);
        F = refFrame;
        F(patches(pp, :, 1)) = 0;
        imagesc(F);
        
        t = (1:size(JOrig, 1))/Fs;
        subplot(332);
        plot(t, JOrig(:, 1), 'r');
        hold on;
        plot(t, JOrig(:, 2), 'g');
        plot(t, JOrig(:, 3), 'b');
        
        subplot(334);
        imagesc(D);
        subplot(335);
        imagesc(DD);
        subplot(336);
        imagesc(A);
        subplot(337);
        plot(V(:, 1), V(:, 2), '.');
        title('Eigenvectors');
        subplot(338);
        plot(mod(theta, 2*pi));
        title(sprintf('Score %g', score));
        subplot(3, 3, 9);
        %Plot local slopes
        
        histogram(deriv, 1:2:100);
        ylabel('Beats Per Minute');
        
        title(sprintf('Rate = %g\bMedian=%g', rate, median(abs(deriv))));
        print('-dpng', '-r100', sprintf('%i.png', pp));
    end
    toc;
end


circScoresImage = circScoresImage-min(circScoresImage(circScoresImage > 0));
circScoresImage = 255*squeeze(circScoresImage(:, :, 1))/max(circScoresImage(:));
circScoresImage = repmat(circScoresImage, [1, 1, 3]);
scoresImageDisp = refFrame;
scoresImageDisp(circScoresImage > 0) = circScoresImage(circScoresImage > 0);
figure(1);
imagesc(scoresImageDisp);


gtScoresImage = gtScoresImage-min(gtScoresImage(gtScoresImage > 0));
gtScoresImage = 255*squeeze(gtScoresImage(:, :, 1))/max(gtScoresImage(:));
gtScoresImage = repmat(gtScoresImage, [1, 1, 3]);
scoresImageDisp = refFrame;
scoresImageDisp(gtScoresImage > 0) = gtScoresImage(gtScoresImage > 0);
figure(2);
imagesc(scoresImageDisp);

figure(3);
plot(rates, scores);
xlabel('Rate (beats per minute');
ylabel('Score');