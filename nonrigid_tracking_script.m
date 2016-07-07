% load image data
%path = 'GroundTruth/1';
%filename = sprintf('%s/00001.jpg', path);
path = 'MeColorRealSense';
filename = sprintf('%s/1.png', path);
refFrame = imread(filename);

% read landmarks acquired using dlib
%load('lm') 
lm = getKeypointsDlib(filename);

%% define face regions

% define forehead upper boundary using landmarks
forehead_y = min(lm(:, 2)) - (lm(29, 2) - min(lm(:, 2)));

regions = cell(1, 7);
colormap = {'y', 'm', 'c', 'r', 'g', 'b', 'w'};

% define region 1: left forehead
x = [lm(19:22, 1); lm(22, 1); lm(19, 1)];
y = [lm(19:22, 2); forehead_y; forehead_y];
regions{1} = [x, y];

% define region 2: middle forehad
x = [lm([22, 28, 23], 1); lm(23, 1); lm(22, 1)];
y = [lm([22, 28, 23], 2); forehead_y; forehead_y];
regions{2} = [x, y];

% define region 3: right forehead 
x = [lm(23:26, 1); lm(26, 1); lm(23, 1)];
y = [lm(23:26, 2); forehead_y; forehead_y];
regions{3} = [x, y];

% define region 4: left cheek
x = lm([1, 7, 49, 40:42, 37], 1);
y = lm([1, 7, 49, 40:42, 37], 2);
regions{4} = [x, y];

% define region 5 right cheek
x = lm([11, 17, 46:48, 43, 55], 1);
y = lm([11, 17, 46:48, 43, 55], 2);
regions{5} = [x, y];

% define region 6 left chin
x = lm([7:9, 58:60, 49], 1);
y = lm([7:9, 58:60, 49], 2);
regions{6} = [x, y];

% define region 7 right chin
x = lm([9:11, 55:58], 1);
y = lm([9:11, 55:58], 2);
regions{7} = [x, y];
NRegions = length(regions);

figure, imshow(refFrame), hold on

for ii = 1:NRegions
    x = regions{ii}(:, 1);
    y = regions{ii}(:, 2);
    h = fill(x, y, colormap{ii});
    set(h, 'FaceAlpha', 0.3)
end

%% detect feature points in first frame

% specify head bounding box
x1 = min(lm(:, 1));
y1 = forehead_y;
x2 = max(lm(:, 1));
y2 = max(lm(:, 2));
roi = [x1, y1, x2 - x1, y2 - y1];

% detect feature points using uses minimum eigenvalue algorithm
points = detectMinEigenFeatures(rgb2gray(refFrame), 'ROI', roi);

% select points within each region
initialPoints = cell(1, NRegions);
for ii = 1:NRegions
    in = inpolygon(points.Location(:,1), points.Location(:, 2), regions{ii}(:, 1), regions{ii}(:, 2));
    initialPoints{ii} = points.Location(in, :);
end

% show tracked points
figure, imshow(refFrame), title('Detected interest points'), hold on;
for ii = 1:7
    scatter(initialPoints{ii}(:, 1), initialPoints{ii}(:, 2), colormap{ii})
end
print('-dpng', '-r100', 'TrackedPoints.png');

%% track points

% initialize tracker for each region
trackers = cell(1, NRegions);

for ii = 1:7
    trackers{ii} = vision.PointTracker('MaxBidirectionalError', 1);
    initialize(trackers{ii}, initialPoints{ii}, refFrame);
end

NFrames = 160;
trackedPoints = cell(NFrames, NRegions);
figure;
% loop through frames
outputView = imref2d(size(refFrame));
for ii = 1:NFrames
      %filename = sprintf('%s/%.5i.jpg', path, ii);
      filename = sprintf('%s/%i.png', path, ii);
      frame = imread(filename);
      %figure, imshow(frame), hold on
      for jj = 1:7
          [points, validity] = step(trackers{jj}, frame);
          size(points)
          size(initialPoints{jj})
          if jj == 5
            %Documentation here: http://www.mathworks.com/help/vision/ref/estimategeometrictransform.html?requestedDomain=www.mathworks.com
            [tform, inlierPtsDistored, inlierPtsOriginal] = estimateGeometricTransform(points, initialPoints{jj}, 'affine');
            clf;
            %TODO: Make sure this transformation is going in the right direction
            J = imwarp(frame, tform, 'OutputView', outputView);
            subplot(221);
            imshow(frame);
            subplot(222);
            imshow(J);
            subplot(223);
            showMatchedFeatures(refFrame, frame, initialPoints{jj}, points);
            P = initialPoints{jj};
            minX = min(P(:, 1)); maxX = max(P(:, 1));
            minY = min(P(:, 2)); maxY = max(P(:, 2));
            xlim([minX, maxX]);
            ylim([minY, maxY]);
            subplot(224);
%             fig = gcf;
%             fig.PaperUnits = 'inches';
%             fig.PaperPosition = [0 0 20 10];
            print('-dpng', '-r100', sprintf('%i.png', ii));
          end
          
          trackedPoints{ii, jj} = points;
          %scatter(points(:, 1), points(:, 2), colormap{jj})
      end
end
