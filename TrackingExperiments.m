addpath('CVPR2014Data');
addpath('GeorgeData');
addpath('OtherData');
addpath(genpath('exact_alm_rpca')); %RPCA Code

%% Main Parameters
%Output Options
DOWARPPLOT = 0;
PLOTPATCHES = 1;
PLOTBANDPASSFILTERS = 0;

%Video parameters
PatchSize = 20;
DOAFFINEWARP = 0;

%Processing Parameters
BlockLenSec = 5;
BlockHopSec = 1;




%% Video loading / Keypoint Detection / Warping
%Step 1: Load in video and ground truth data
[I, files, refFrame, Fs, groundTruthMean] = getCVPR2014Video(4, 400);
NFrames = size(I, 1);

%Step 2: Get keypoints, face regions, and patches within those regions
if DOAFFINEWARP
    %This is the bottleneck step and is only needed if an affine warp
    %is being done
    Keypoints = getKeypointsDlib(files); 
else
    %Otherwise, only the first frame is needed to determine regions
    Keypoints = getKeypointsDlib(files(1));
end
regions = getDlibFaceRegions(Keypoints{1});
patches = getRegionPatchIndices(regions, size(refFrame), PatchSize);
if PLOTPATCHES
    IM = refFrame;
    for ii = 1:size(patches, 1)
        IM(patches(ii, :, 1)) = 0;
    end
    imagesc(IM);
    title('Patches');
end

%Step 3: If enabled, affine warp all frames to the first frame
if DOAFFINEWARP
    I = doAffineWarpVideo(I, refFrame, Keypoints, DOWARPPLOT);
end




%% Setup Block Window Parameters
BlockLen = round(Fs*BlockLenSec); %Number of video frames per block
BlockHop = round(BlockHopSec); %Number of video frames to hop to next block
NBlocks = floor((NFrames-BlockLen)/BlockHop+1);

%Kumar Parameters
Ath = 8; %Maximum range cutoff for regions
bWin = 5; %b in Equation 7
Kappa = 0.3;

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





%% Loop Through Blocks And Do Heartrate Estimates
for kk = 1:NBlocks
    fprintf(1, 'Doing Block %i of %i...\n', kk, NBlocks);
    X = zeros(size(patches, 1)*size(patches, 3), BlockLen); %Allocate space for the averaged/filtered patches
    
    %Step 1: Average and bandpass filter regions of interest
    for pp = 1:size(patches, 1)
        J = I(BlockHop*(kk-1)+(1:BlockLen), patches(pp, :, :));
        J = reshape(J, [size(J, 1), size(patches, 2), size(patches, 3)]);
        J = squeeze(mean(J, 2)); %Spatial average within patch
        JFilt = filtfilt(bpfilter, 1, J); %Bandpass Filter
        JFilt2 = getSmoothedDerivative(J, 20); %Smoothed Derivative Filter
        
        for aa = 1:3
            X((pp-1)*3+aa, :) = JFilt(:, aa);
        end
        t1 = (1:size(J, 1))/Fs;
        t2 = (1:size(JFilt2, 1))/Fs;
        
        if PLOTBANDPASSFILTERS
            clf;
            subplot(221);
            F = refFrame;
            F(patches(pp, :, 1)) = 0;
            imagesc(F);
            subplot(222);
            J = bsxfun(@minus, mean(J, 1), J);
            %plot(t1, J(:, 1), 'r');
            hold on;
            %plot(t1, J(:, 2), 'g');
            plot(t1, J(:, 3), 'b');
            title('Original');
            subplot(223);
            %plot(t1, JFilt(:, 1), 'r');
            hold on;
            %plot(t1, JFilt(:, 2), 'g');
            plot(t1, JFilt(:, 3), 'b');
            title('Zero Phase Bandpass Filter');
            subplot(224);
            %plot(t2, JFilt2(:, 1), 'r');
            hold on;
            %plot(t2, JFilt2(:, 2), 'g');
            plot(t2, JFilt2(:, 3), 'b');
            title('Smoothed Derivative Filter');
            print('-dpng', '-r100', sprintf('BandpassPatch%i.png', pp));
        end
    end %End patch averaging / filtering loop
    %Subtract off mean of each patch
    X = bsxfun(@minus, X, mean(X, 2));
    
    %Step 2: Apply different tracking techniques to each block
    [bpmFinal, freq, PFinal] = TrackKumar(X, Fs, Ath, bWin, refFrame, t1, fl, fh, sprintf('Kumar%i', kk), patches);
    

end %End block loop
