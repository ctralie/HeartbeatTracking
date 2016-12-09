%W: Window size
%Kappa: Nearest neighbor percentage
function [bpmFinal, rates, scores, AllDs] = TrackCircularCoordinates(X, Fs, W, Kappa, refFrame, patches, DebugStr, DebugPatchesStr, hopOffset)
    if nargin < 7
        DebugStr = -1;
    end
    if nargin < 8
        DebugPatchesStr = -1;
    end
    ratesRange = linspace(30, 200, 1000); %Heartbeat estimation points
    NPatches = size(X, 1);
    NFrames = size(X, 2);
    rates = zeros(NPatches, 1);
    ratesSmoothed = zeros(NPatches, length(ratesRange));
    scores = zeros(NPatches, 1);
    AllDs = zeros(NPatches, (NFrames-W+1)^2);
    
    
    
    for pp = 1:NPatches
        J = X(pp, :);
        JOrig = J;
        D = pdist2(J(:), J(:));

        J = getDelayEmbedding(J(:), W);
        J = bsxfun(@minus, mean(J, 1), J);
        J = bsxfun(@times, 1./sqrt(sum(J.^2, 2)), J);
        DD = pdist2(J, J);
        AllDs(pp, :) = DD(:);

        %Get graph laplacian
        A = groundTruthKNN(DD, round(Kappa*size(DD, 2)));
        A = A.*groundTruthKNN(DD', round(Kappa*size(DD, 1)))';
        A = A - speye(size(A, 1));
        L = spdiags(sum(A, 2), 0, size(A, 1), size(A, 2)) - A;
        

        %Get eigenvectors
        V = zeros(size(L, 1), 3);
        try
            [V, ~] = eigs(L, 3, 0);
        catch ME
            disp('Error');
            continue;
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

        scores(pp) = exp(-score/0.1);

        avgRate = abs((theta(end)-theta(1))/(length(theta)-1));
        avgRate = avgRate*Fs*60/(2*pi);
        rates(pp) = median(deriv); %Store median rate
        ratesSmoothed(pp, :) = exp(-(rates(pp)-ratesRange).^2/(2*2^2));

        if DebugPatchesStr ~= -1
            clf;
            subplot(331);
            F = refFrame;
            pidx = mod(pp, size(patches, 1));
            channel = floor(pp/size(patches, 1)) + 1;
            if pidx == 0
                pidx = size(patches, 1);
                channel = channel - 1;
            end
            F(patches(pidx, :, :)) = 0;
            F(patches(pidx, :, channel)) = 255;
            imagesc(F);

            t = (1:length(JOrig))/Fs + hopOffset/Fs;
            subplot(332);
            plot(t, JOrig);
            xlim([t(1), t(end)]);
            title('Time Series Patch');

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

            title(sprintf('Rate = %g\bMedian=%g', avgRate, median(abs(deriv))));
            print('-dpng', '-r100', sprintf('%s_Patch%i.png', DebugStr, pp));
        end
    end
    
    %Determine final heartreat and visualize beat estimates
    bpmFinal = bsxfun(@times, scores(:), ratesSmoothed);
    bpmFinal = mean(bpmFinal, 1);
    if DebugStr ~= -1
        clf;
        subplot(211);
        t = (1:length(JOrig))/Fs + hopOffset/Fs;
        plot(t, mean(X, 1));
        xlim([t(1), t(end)]);
        xlabel('Time (Seconds)');
        title('Patch / Channel Average Time Series');
        subplot(212);
        plot(ratesRange, bpmFinal);
        xlabel('Heartrate (bpm)');
        title('Heartrate Estimate');
        print('-dpng', '-r100', sprintf('%sSmoothedHeartrate.png', DebugStr));
    end
    [~, idx] = max(bpmFinal);
    bpmFinal = ratesRange(idx);
    
    
    %Step 4: Visualize Goodness of Fit of Patches
    if DebugStr ~= -1
        IMScores = refFrame;
        G = uint8(255*scores/max(scores(:)));
        for pp = 1:size(patches, 1)
            for aa = 1:3
                IMScores(patches(pp, :, aa)) = G(pp);
            end
        end
        imwrite(IMScores, sprintf('GoodnessFit%s.png', DebugStr));
    end
end