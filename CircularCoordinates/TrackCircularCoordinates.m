%W: Window size
%Kappa: Nearest neighbor percentage
function [theta1, theta2] = TrackCircularCoordinates(X, Fs, W, Kappa, DebugStr, hopOffset)
    if nargin < 5
        DebugStr = -1;
    end
    NPatches = size(X, 1);
    NFrames = size(X, 2);
    M = NFrames-W+1;
    AllDs = zeros(NPatches, M^2);
    
    %Step 1: Compute all normalized delay SSMs
    for pp = 1:NPatches
        J = X(pp, :);
        JOrig = J;
        %D = pdist2(J(:), J(:));

        J = getDelayEmbedding(J(:), W);
        J = bsxfun(@minus, mean(J, 1), J);
        JNorm = sqrt(sum(J.^2, 2));
        JNorm(JNorm == 0) = 1;
        J = bsxfun(@times, 1./JNorm, J);
        DD = pdist2(J, J);
        AllDs(pp, :) = DD(:);
    end
    
    %Step 2: Spectral cluster SSMs
    AllDs = AllDs(sum(AllDs, 2) > 0, :);
    [V, ~] = SpectralEmbed(AllDs);
    V = V(:, 2); %Fiedler vector for spectral clustering
    
    Ds1 = AllDs(V > 0, :);
    d1 = reshape(mean(Ds1, 1), M, M);
    [V1, A1, e1] = SpectralEmbedUnweighted(d1, Kappa);
    theta1 = atan2(V1(:, 3), V1(:, 2));
    
    Ds2 = AllDs(V <= 0, :);
    d2 = reshape(mean(Ds2, 1), M, M);
    [V2, A2, e1] = SpectralEmbedUnweighted(d2, Kappa);
    theta2 = atan2(V2(:, 3), V2(:, 2));
    
    
    if DebugStr ~= -1
        clf;
        subplot(331);
        t = (1:length(JOrig))/Fs + hopOffset/Fs;
        plot(t, mean(X, 1));
        xlim([t(1), t(end)]);
        title('Mean time series');
        
        subplot(332);
        imagesc(d1);
        title('Cluster 0');
        
        subplot(333);
        imagesc(d2);
        title('Cluster 1');
        
        subplot(338);
        scatter(V1(:, 2), V1(:, 3), 10, 1:size(V1, 1), 'fill');
        axis equal;
        title('Eigs D1');
        
        subplot(335);
        imagesc(A1);
        title('Adjacency 1');
        
        subplot(339);
        scatter(V2(:, 2), V2(:, 3), 10, 1:size(V2, 1), 'fill');
        axis equal;
        title('Eigs D2');
        
        subplot(336);
        imagesc(A2);
        title('Adjacency 2');
        
        subplot(334);
        plot(theta1, 'r');
        title('Theta 1');
        
        subplot(337);
        plot(theta2, 'b');
        title('Theta 2');

        print('-dpng', '-r100', sprintf('%sCircCoords.png', DebugStr));
    end
    
    %Unwrap circular coordinates and make sure they start from 0
    theta1 = unwrap(theta1);
    if theta1(end) - theta1(1) < 0
        theta1 = -theta1;
    end
    theta1 = theta1 - theta1(1);
    
    theta2 = unwrap(theta2);
    if theta2(end) - theta2(end) < 0
        theta2 = -theta2;
    end
    theta2 = theta2 - theta2(1);
end