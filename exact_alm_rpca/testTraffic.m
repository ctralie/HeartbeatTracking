%First do PCA
NFrames = 475;
frame = imread('traffic/1.png');
X = zeros(length(frame(:)), NFrames);
X(:, 1) = frame(:);
K = 10;
IDims = size(frame);
for ii = 2:NFrames
    ii
    F = imread(sprintf('traffic/%i.png', ii));
    X(:, ii) = F(:);
end
XTX = X'*X;
[V, S] = eig(XTX);
S = diag(S);
US = X*V;
US = US(:, end-K:end);
V = V(:, end-K:end);
XPCA = US*V';
imagesc(reshape(XPCA(:, 1), IDims)/255.0);

for ii = 1:NFrames
    clf;
    subplot(131);
    imagesc(reshape(X(:, ii), IDims)/255.0);
    title('Original');
    axis off;
    subplot(132);
    imagesc(reshape(XPCA(:, ii), IDims)/255.0);
    title(sprintf('Rank %i PCA', K));
    axis off;
    subplot(133);
    imagesc(reshape(X(:, ii)-XPCA(:, ii), IDims)/255.0);
    title('Difference');
    fig = gcf;
    fig.PaperUnits = 'inches';
    fig.PaperPosition = [0 0 20 6];
    print('-dpng', '-r100', sprintf('%i.png', ii));
end

%Now do RPCA
[XRPCA, Outlier, iter] = inexact_alm_rpca(X, 5e-2);

for ii = 1:NFrames
    clf;
    subplot(131);
    imagesc(reshape(X(:, ii), IDims)/255.0);
    title('Original');
    axis off;
    subplot(132);
    imagesc(reshape(XRPCA(:, ii), IDims)/255.0);
    title(sprintf('Rank %i PCA', K));
    axis off;
    subplot(133);
    imagesc(reshape(Outlier(:, ii), IDims)/255.0);
    title('Difference');
    fig = gcf;
    fig.PaperUnits = 'inches';
    fig.PaperPosition = [0 0 20 6];
    print('-dpng', '-r100', sprintf('%i.png', ii));
end
