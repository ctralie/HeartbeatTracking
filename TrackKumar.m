%Inputs: X

%Returns:
%bpmFinal: Estimate of heartrate in beats per minute in this block
%freq, PFinal: Spectrum of estimates
function [bpmFinal, freq, PFinal] = TrackKumar(X, Fs, Ath, bWin, refFrame, t1, fl, fh, DebugStr, patches)
    if nargin < 7
        DebugStr = -1;
    end
    
    %Step 1: Compute coarse estimate of pulse rate by averaging (Section 4.3)
    XRange = max(X, [], 2) - min(X, [], 2);
    SCoarse = squeeze(sum(X(XRange < Ath, :), 1));
    N = length(SCoarse);
    PCoarse = fft(SCoarse);
    PCoarse = PCoarse(1:N/2+1);
    PCoarse = (1/(Fs*N)) * abs(PCoarse).^2;
    PCoarse(:, 1) = 0; %Just in case X wasn't mean-centered
    freq = 0:Fs/N:Fs/2;
    [~, idxPR] = max(PCoarse);
    bpmCoarse = freq(idxPR)*60;
    
    if DebugStr ~= -1
        clf;
        subplot(211);
        plot(t1, SCoarse);
        title('Initial Coarse Time Series');
        subplot(212);
        plot(freq*60, abs(PCoarse));
        %xlim([fl, fh]*60);
        xlabel('Beats Per Minute');
        title(sprintf('Initial Coarse Power Spectrum (Max %g bmp)', bpmCoarse));
        print('-dsvg', sprintf('KumarCoarseEstimate%s.svg', DebugStr));
    end
    
    %Step 2: Estimage goodness of regions
    %Figure out frequency index range of pulse window
    PRL = floor(1 + ((bpmCoarse - bWin)/60)*(N/Fs));
    PRH = ceil(1 + ((bpmCoarse + bWin)/60)*(N/Fs));
    %Figure out frequency index range of passband
    BPL = floor(1 + fl*N/Fs);
    BPH = floor(1 + fh*N/Fs);
    
    %Compute goodness of fit for each region
    PSD = fft(X, N, 2);
    PSD = PSD(:, 1:N/2+1);
    PSD = (1/(Fs*N)) * abs(PSD).^2;
    a = sum(PSD(:, PRL:PRH), 2);
    b = sum(PSD(:, BPL:BPH), 2);
    G = a./(a+b);
    G = G.*(XRange < Ath);
    
    %Step 3: Compute Final Estimate Based on Updated Goodness of Fit
    %Regions
    SFinal = squeeze(sum(bsxfun(@times, G(:), X), 1));
    PFinal = fft(SFinal);
    PFinal = PFinal(1:N/2+1);
    PFinal = (1/(Fs*N)) * abs(PFinal).^2;
    [~, idxFinal] = max(PFinal);
    bpmFinal = freq(idxFinal)*60;
    
    if DebugStr ~= -1
        clf;
        subplot(211);
        plot(t1, SFinal);
        title('Final Filtered Time Series');
        subplot(212);
        plot(freq*60, abs(PFinal));
        xlim([fl, fh]*60);
        xlabel('Beats Per Minute');
        title(sprintf('Final Filtered Power Spectrum (Max %g bmp)', bpmFinal));
        print('-dsvg', sprintf('FinalEstimate%s.svg', DebugStr));

        clf;
        hist(G);
        title('Goodness Ratios');
        ylabel('Count');
        xlabel('Ratios');
        print('-dsvg', sprintf('GoodnessHist%s.svg', DebugStr));
    end
    
    %Step 4: Visualize Goodness of Fit of Patches
    if DebugStr ~= -1
        IMScores = refFrame;
        G = uint8(255*G/max(G(:)));
        for pp = 1:size(patches, 1)
            for aa = 1:3
                IMScores(patches(pp, :, aa)) = G(aa+(pp-1)*3);
            end
        end
        imwrite(IMScores, sprintf('GoodnessFit%s.png', DebugStr));
    end
end