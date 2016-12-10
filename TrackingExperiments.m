%Programmer: Chris Tralie
%Purpose: To make it easy to run a bunch of different heartrate detection
%algorithms in overlapping blocks.  Only the frames needed to process the
%current block are stored in memory.  It's easy to mix and match different
%algorithms and parameters (e.g. affine warping, low rank stuff)

addpath('BUData');
addpath('CVPR2014Data');
addpath('GeorgeData');
addpath('OtherData');
addpath('CircularCoordinates');
addpath(genpath('exact_alm_rpca')); %RPCA Code

%% Main Parameters
%Output Options
DOWARPPLOT = 0;
PLOTPATCHES = 1;
PLOTBANDPASSFILTERS = 0;
DEBUGKUMAR = 1;
DEBUGCIRCULAR = 0;
DEBUGCIRCULARPATCHES = 0;

%Video parameters
PatchSize = 20;
DOAFFINEWARP = 1;

%Processing Parameters
BlockLenSec = 5;
BlockHopSec = 0.1;
lowranklambda = 0; %Do a low rank approximation of the patch time series?




%% Video loading / Keypoint Detection / Setting Up Patch Regions
%Step 1: Configure video to be loaded
%VideoLoader = @(i1, i2) getCVPR2014Video(4, i1, i2);
VideoLoader = @(i1, i2) getBUVideo(0, 'F013/T1', i1, i2);

%Load the first frame just to get the sample rate and the total number of
%frames
ret = VideoLoader(1, 1);
Fs = ret.Fs;
NFrames = ret.NFrames;
refFrame = ret.refFrame; %Reference frame
GTPR = ret.GTPR; %Ground truth pulse rate

%Setup Block Window Parameters
BlockLen = round(Fs*BlockLenSec); %Number of video frames per block
BlockHop = round(Fs*BlockHopSec); %Number of video frames to hop to next block
NBlocks = floor((NFrames-BlockLen)/BlockHop+1);

%Now load in the first block
ret = VideoLoader(1, BlockLen);
files = ret.files;
I = ret.I;

%Step 2: Get keypoints, face regions, and patches within those regions
if DOAFFINEWARP
    %This is the bottleneck step and is only needed if an affine warp
    %is being done
    Keypoints = getKeypointsDlib(files); 
else
    %Otherwise, only the first frame is needed to determine regions
    Keypoints = getKeypointsDlib(files(1));
end
lm = Keypoints{1};%The first keypoints

regions = getDlibFaceRegions(Keypoints{1});
patches = getRegionPatchIndices(regions, size(refFrame), PatchSize);
if PLOTPATCHES
    IM = refFrame;
    for ii = 1:size(patches, 1)
        IM(patches(ii, :, 1)) = 0;
    end
    imagesc(IM);
    title('Patches');
    print('-dpng', '-r100', 'Patches.png');
end

VWarpedWriter = VideoWriter('WarpedVideo.avi');
open(VWarpedWriter);

%Step 3: If enabled, affine warp all frames to the first frame
if DOAFFINEWARP
    I = doAffineWarpVideo(I, refFrame, lm, Keypoints, VWarpedWriter, DOWARPPLOT, 0);
end


%% Setup Algorithm Parameters
%Kumar Parameters
Ath = 8; %Maximum range cutoff for regions
bWin = 5; %b in Equation 7

%Circular coordinate parameters
W = 15;
Kappa = 0.3;

%Bandpass Filter design
fl = 0.7;
fh = 4;
minfq = fl*2/Fs;
maxfq = fh*2/Fs;
bpfilter = fir1(floor(BlockLen/3.1),[minfq maxfq]);
%s2f = filtfilt(bpfilter,1,s2);



%% Loop Through Blocks And Do Heartrate Estimates
KumarRates = zeros(1, NBlocks);
lastIdx = BlockLen;

for kk = 1:NBlocks
    fprintf(1, 'Doing Block %i of %i...\n', kk, NBlocks);
    hopOffset = BlockHop*(kk-1);
    if kk > 1
        %Load in new frames and discard oldest frames
        I = I(BlockHop+1:end, :);
        files = files(BlockHop+1:end);
        
        hopOffset = BlockHop*(kk-1);
        ret = VideoLoader(lastIdx+1, lastIdx+BlockHop);
        if DOAFFINEWARP
            Keypoints = getKeypointsDlib(ret.files);
            ret.I = doAffineWarpVideo(ret.I, refFrame, lm, Keypoints, VWarpedWriter, DOWARPPLOT, lastIdx+1);
        end
        I = [I; ret.I];
        lastIdx = lastIdx + BlockHop;
    end
    
    X = zeros(size(patches, 1)*size(patches, 3), BlockLen); %Allocate space for the averaged/filtered patches
    NPatches = size(patches, 1);
    
    %Step 1: Average and bandpass filter regions of interest
    for pp = 1:NPatches
        J = I(:, patches(pp, :, :));
        J = reshape(J, [size(J, 1), size(patches, 2), size(patches, 3)]);
        J = squeeze(mean(J, 2)); %Spatial average within patch
        JFilt = filtfilt(bpfilter, 1, J); %Bandpass Filter
        JFilt2 = getSmoothedDerivative(J, 20); %Smoothed Derivative Filter
        
        %Stack all of the channels separately.  Put all R first, all G
        %second, all B third
        for aa = 1:3
            X((aa-1)*NPatches + pp, :) = JFilt(:, aa);
        end
        if lowranklambda > 0
            [X, Outlier, iter] = inexact_alm_rpca(X, lowranklambda);
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
    %Kumar Technique
    if DEBUGKUMAR
        [bpmFinal, freq, PFinal] = TrackKumar(X, Fs, Ath, bWin, refFrame, t1, fl, fh, sprintf('Kumar%i', kk), patches, hopOffset);
    else
        [bpmFinal, freq, PFinal] = TrackKumar(X, Fs, Ath, bWin, refFrame, t1, fl, fh);
    end
    KumarRates(kk) = bpmFinal;
    
    %Circular coordinates technique;
	%[bpmFinal, rates, scores, AllDs] = TrackCircularCoordinates(X, Fs, W, Kappa, refFrame, patches, sprintf('Circular%i', kk), sprintf('Circular%i', kk), hopOffset);

end %End block loop

%Plot performance against ground truth
clf;
ts = (1:length(KumarRates))*BlockHop/Fs;
plot(ts, KumarRates, 'b');
hold on;
ts = (1:length(GTPR))/1000.0;
plot(ts, GTPR, 'r');
xlabel('Time');
ylabel('HeartRate');
legend({'Kumar Estimate', 'Ground Truth'});
title('Performance');

close(VWarpedWriter);