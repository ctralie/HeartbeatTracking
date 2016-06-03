% 
% %% Process video to get data points
% % path = 'E:\Research\Pulse_Detection\Matlab\data\face.mp4';
% path = 'G:\PulseData\Database\GroundTruth\4\gt4.avi';
% videoFileReader = vision.VideoFileReader(path);
% videoPlayer = vision.VideoPlayer('Position', [100, 100, 680, 520]);
% 
% objectFrame = step(videoFileReader);
% 
% faceDetector = vision.CascadeObjectDetector;
% 
% objectRegion = step(faceDetector, objectFrame);
% objectRegion1(1) = objectRegion(1) + round(0.25 * objectRegion(3));
% objectRegion1(2) = objectRegion(2) + round(0.05 * objectRegion(4));
% objectRegion1(3) = round(0.5 * objectRegion(3));
% objectRegion1(4) = round(0.15 * objectRegion(4));
% 
% objectRegion2(1) = objectRegion(1) + round(0.25 * objectRegion(3));
% objectRegion2(2) = objectRegion(2) + round(0.45 * objectRegion(4));
% objectRegion2(3) = round(0.5 * objectRegion(3));
% objectRegion2(4) = round(0.35 * objectRegion(4));
% 
% objectImage = insertShape(objectFrame, 'Rectangle', objectRegion1,'Color', 'red');
% objectImage = insertShape(objectImage, 'Rectangle', objectRegion2,'Color', 'red');
% figure; imshow(objectImage); title('Red box shows object region');
% 
% points1 = detectMinEigenFeatures(rgb2gray(objectFrame), 'ROI', objectRegion1);
% points2 = detectMinEigenFeatures(rgb2gray(objectFrame), 'ROI', objectRegion2);
% points = [points1; points2];
% pointImage = insertMarker(objectFrame, points.Location, '+', 'Color', 'white');
% figure, imshow(pointImage), title('Detected interest points');
% 
% tracker = vision.PointTracker('MaxBidirectionalError', 1);
% initialize(tracker, points.Location, objectFrame);
% 
% data = [];
% 
% % while ~isDone(videoFileReader)
% %       frame = step(videoFileReader);
% %       [points, validity] = step(tracker, frame);
% %       out = insertMarker(frame, points(validity, :), '+');
% %       ypoint = points(:,2)';
% %       data = cat(1, data, ypoint);
% %       step(videoPlayer, out);
% % end
% 
% for i = 1:400
%       frame = step(videoFileReader);
%       [points, validity] = step(tracker, frame);
%       out = insertMarker(frame, points(validity, :), '+');
%       ypoint = points(:,2)';
%       data = cat(1, data, ypoint);
%       step(videoPlayer, out);
% end
% 
% release(videoPlayer);
% release(videoFileReader);

%% Discard unstable points

dist = data(2:end, :) - data(1:end-1, :);
maxDist = round(max(dist), 1);
distMode = mode(maxDist);
idx = maxDist <= distMode;
stableData = data(:, idx);

%% Interpolate data
frameRate = 30;
interpolateFreq = 30;
[n, d] = size(stableData);
x = 1:n;
interCoeff = frameRate / interpolateFreq; % interpolation coefficient
yy = 1:interCoeff:n;
interpolatedData = [];
for i = 1:d
    interpolatedData = cat(2, interpolatedData, spline(x, stableData(:, i), yy)'); 
end

%% Band filter data
% Zero-Pole-Gain design
% [z,p,k] = butter(3, [0.75, 5] / (interpolateFreq / 2), 'bandpass');
% sos = zp2sos(z,p,k);
% filteredData = sosfilt(sos, interpolatedData, 2);
% figure, plot(1:size(filteredData, 1), filteredData(:,200))

fl = 0.75;
fh = 5;
minfq = fl*2/interpolateFreq;
maxfq = fh*2/interpolateFreq;
bpfilter = fir1(128,[minfq maxfq]);
filteredData = filtfilt(bpfilter,1,double(interpolatedData));

%% PCA decomposition
m = mean(filteredData);
n_interpolate = size(filteredData, 1);
d = filteredData - repmat(m, n_interpolate, 1);
filteredDataNorms = sum(d .* d, 2);
[sortedValues, sortIndex] = sort(filteredDataNorms, 'descend');
alpha = 0.25; % discard points with largest l2-norms
idx = sortIndex(round(length(sortIndex) * alpha):end);
filteredDataSelect = filteredData(idx, :);

filteredDataSelectCentered = filteredDataSelect - repmat(mean(filteredDataSelect, 1), length(idx), 1);
pcaCoeff = pca(filteredDataSelectCentered);
filteredDataCentered = filteredData - repmat(mean(filteredData, 1), n_interpolate, 1);
dataAfterPca = filteredDataCentered * pcaCoeff;

%% Signal Selection
fs = interpolateFreq;                           % Sample frequency
m = size(dataAfterPca, 1);                      % Window length
n = pow2(nextpow2(m));                          % Transform length
powerRatio = nan(5, 1);
freq = nan(5, 1);
for i = 1:5
    y = fft(dataAfterPca(:, i), n);             % DFT
    f = (0 : n - 1) * (fs / n);                 % Frequency range
    power = y .* conj(y) / n;                   % Power of the DFT
    fullSpecPower = sum(power(1:floor(length(power) / 2)));
    [freqPower, freq(i)] = max(power(1:floor(length(power) / 2)));
    freqPower = freqPower + power(2 * freq(i));
    powerRatio(i) = freqPower / fullSpecPower;

    % visualize the 0-Centered Periodogram
    y0 = fftshift(y);                           % Rearrange y values
    f0 = (-n / 2 : n / 2 - 1) * (fs / n);       % 0-centered frequency range
    power0 = y0 .* conj(y0) / n;                % 0-centered power
    figure, plot(f0, power0)
    xlabel('Frequency (Hz)')
    ylabel('Power')
    title('{\bf 0-Centered Periodogram}')
end

pulseFreq = (freq - 1) * (fs / n);
% [~, signalIdx] = max(powerRatio);
% [~, freqPulse] = max(power);
