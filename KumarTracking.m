%% Load Video
%load image data
% Fs = 30;
% path = 'GroundTruth/2';
% NFrames = 400;
% startFrame = 0;
% filename = sprintf('%s/00001.jpg', path);

% Fs = 30;
% path = 'GroundTruth/4';
% NFrames = 900;
% filename = sprintf('%s/0891.jpg', path);
% startFrame = 890;
% groundTruthMean = 71.2081;

% Fs = 30;
% path = 'GroundTruth/3';
% NFrames = 900;
% filename = sprintf('%s/1211.jpg', path);
% startFrame = 1210;
% groundTruthMean = 61.4943;

% Fs = 40; %Sample rate
% path = 'MeColorRealSense'; 
% NFrames = 400;
% filename = sprintf('%s/1.png', path);
% startFrame = 0;

Fs = 30;
path = 'GeorgeData/4_30_mauricio_ir_depth_30/ir';
NFrames = 400;
filename = sprintf('%s/ir_0000.png', path);
startFrame = 0;


refFrame = imread(filename);
lm = getKeypointsDlib(filename);
regions = getDlibFaceRegions(lm);

patches = getRegionPatchIndices(regions, size(refFrame), 20);

% for ii = 1:size(patches, 1)
%     refFrame(patches(ii, :, 1)) = 0;
% end
% imagesc(refFrame);

I = zeros(NFrames, numel(refFrame));
%V = VideoWriter('video.avi');
%open(V);
for ii = 1:NFrames
    fprintf(1, 'Loading frame %i...\n', ii);
    %filename = sprintf('%s/%.4i.jpg', path, ii+startFrame);
    %filename = sprintf('%s/%.5i.jpg', path, ii+startFrame);
    %filename = sprintf('%s/%i.png', path, ii);
    filename = sprintf('%s/ir_%.4i.png', path, ii+startFrame);
    frame = imread(filename);
    %writeVideo(V, frame);
    I(ii, :) = frame(:);
end
%close(V);

%% Setup Parameters
%Kumar Parameters
BlockLen = Fs*10; %Process in 10 second blocks
BlockHop = BlockLen;
NBlocks = floor((NFrames-BlockLen)/BlockHop+1);
Ath = 8; %Maximum range cutoff for regions
bWin = 5; %b in Equation 7

Kappa = 0.3;

circScoresImage = zeros(size(refFrame));
gtScoresImage = zeros(size(refFrame));

rates = linspace(30, 200, 1000);
scores = 0*rates;
AllRates = [];

%Bandpass Filter design
fl = 0.7;
fh = 4;
minfq = fl*2/Fs;
maxfq = fh*2/Fs;
bpfilter = fir1(floor(BlockLen/3.1),[minfq maxfq]);
%s2f = filtfilt(bpfilter,1,s2);

%% Core Algorithm

for kk = 1%1:NBlocks %Go through each "epoch" as Kumar calls it
    fprintf(1, 'Doing Block %i of %i...\n', kk, NBlocks);
    X = zeros(size(patches, 1)*size(patches, 3), BlockLen); %Allocate space for the averaged/filtered patches
    
    %Step 1: Average and bandpass filter regions of interest
    for pp = 1:size(patches, 1)
        J = I(BlockHop*(kk-1)+(1:BlockLen), patches(pp, :, :));
        J = reshape(J, [size(J, 1), size(patches, 2), size(patches, 3)]);
        J = squeeze(mean(J, 2));
        JFilt = filtfilt(bpfilter, 1, J);
        JFilt2 = getSmoothedDerivative(J, 20);
        
        for aa = 1:3
            X((pp-1)*3+aa, :) = JFilt(:, aa);
        end
        t1 = (1:size(J, 1))/Fs;
        t2 = (1:size(JFilt2, 1))/Fs;
        
%         clf;
%         
%         subplot(221);
%         F = refFrame;
%         F(patches(pp, :, 1)) = 0;
%         imagesc(F);
%         
%         
%         subplot(222);
%         J = bsxfun(@minus, mean(J, 1), J);
%         %plot(t1, J(:, 1), 'r');
%         hold on;
%         %plot(t1, J(:, 2), 'g');
%         plot(t1, J(:, 3), 'b');
%         title('Original');
% 
%         subplot(223);
%         %plot(t1, JFilt(:, 1), 'r');
%         hold on;
%         %plot(t1, JFilt(:, 2), 'g');
%         plot(t1, JFilt(:, 3), 'b');
%         title('Zero Phase Bandpass Filter');
%         
%         subplot(224);
%         
%         %plot(t2, JFilt2(:, 1), 'r');
%         hold on;
%         %plot(t2, JFilt2(:, 2), 'g');
%         plot(t2, JFilt2(:, 3), 'b');
%         title('Smoothed Derivative Filter');
%         
%         print('-dpng', '-r100', sprintf('%i.png', pp));
        
    end

    %Step 2: Compute coarse estimate of pulse rate (Section 4.3)
    XRange = max(X, [], 2) - min(X, [], 2);
    SCoarse = sum(X(XRange < Ath, :), 1);
    N = length(SCoarse);
    PCoarse = fft(SCoarse);
    PCoarse = PCoarse(1:N/2+1);
    PCoarse = (1/(Fs*N)) * abs(PCoarse).^2;
    freq = 0:Fs/N:Fs/2;
    [~, idxPR] = max(PCoarse);
    bpmCoarse = freq(idxPR)*60;
    
    clf;
    subplot(211);
    plot(t1, SCoarse);
    title('Initial Coarse Time Series');
    subplot(212);
    plot(freq*60, abs(PCoarse));
    xlim([fl, fh]*60);
    xlabel('Beats Per Minute');
    title(sprintf('Initial Coarse Power Spectrum (Max %g bmp)', bpmCoarse));
    print('-dsvg', sprintf('CoarseEstimate%i.svg', kk));
    
    %Step 3: Estimage goodness of regions
    
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
    
    %Step 4: Compute Final Estimate Based on Updated Goodness of Fit
    %Regions
    SFinal = sum(bsxfun(@times, G(:), X), 1);
    PFinal = fft(SFinal);
    PFinal = PFinal(1:N/2+1);
    PFinal = (1/(Fs*N)) * abs(PFinal).^2;
    [~, idxFinal] = max(PFinal);
    bpmFinal = freq(idxFinal)*60;
    clf;
    subplot(211);
    plot(t1, SFinal);
    title('Final Filtered Time Series');
    subplot(212);
    plot(freq*60, abs(PFinal));
    xlim([fl, fh]*60);
    xlabel('Beats Per Minute');
    title(sprintf('Final Filtered Power Spectrum (Max %g bmp)', bpmFinal));
    print('-dsvg', sprintf('FinalEstimate%i.svg', kk));
    
    clf;
    hist(G);
    title('Goodness Ratios');
    ylabel('Count');
    xlabel('Ratios');
    print('-dsvg', sprintf('GoodnessHist%i.svg', kk));
    
    %Step 5: Visualize Goodness of Fit of Patches
    %Visualize goodness of fit
    IMScores = refFrame;
    G = uint8(255*G/max(G(:)));
    for pp = 1:size(patches, 1)
        for aa = 1:3
            IMScores(patches(pp, :, aa)) = G(aa+(pp-1)*3);
        end
    end
    imwrite(IMScores, sprintf('GoodnessFit%i.png', kk));
end
