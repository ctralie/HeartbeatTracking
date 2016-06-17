% load image data
path = '.\data\';
refFrame = imread(strcat(path, '1211.jpg'));

% read landmarks acquired using dlib
load('lm') 

%% define face regions

% define forehead upper boundary using landmarks
forehead_y = min(lm(:, 2)) - (lm(29, 2) - min(lm(:, 2)));

region = cell(1, 7);
colormap = {'y', 'm', 'c', 'r', 'g', 'b', 'w'};

% define region 1: left forehead
x = [lm(19:22, 1); lm(22, 1); lm(19, 1)];
y = [lm(19:22, 2); forehead_y; forehead_y];
region{1} = [x, y];

% define region 2: middle forehad
x = [lm([22, 28, 23], 1); lm(23, 1); lm(22, 1)];
y = [lm([22, 28, 23], 2); forehead_y; forehead_y];
region{2} = [x, y];

% define region 3: right forehead 
x = [lm(23:26, 1); lm(26, 1); lm(23, 1)];
y = [lm(23:26, 2); forehead_y; forehead_y];
region{3} = [x, y];

% define region 4: left cheek
x = lm([1, 7, 49, 40:42, 37], 1);
y = lm([1, 7, 49, 40:42, 37], 2);
region{4} = [x, y];

% define region 5 right cheek
x = lm([11, 17, 46:48, 43, 55], 1);
y = lm([11, 17, 46:48, 43, 55], 2);
region{5} = [x, y];

% define region 6 left chin
x = lm([7:9, 58:60, 49], 1);
y = lm([7:9, 58:60, 49], 2);
region{6} = [x, y];

% define region 7 right chin
x = lm([9:11, 55:58], 1);
y = lm([9:11, 55:58], 2);
region{7} = [x, y];

figure, imshow(refFrame), hold on

for i = 1:7
    x = region{i}(:, 1);
    y = region{i}(:, 2);
    h = fill(x, y, colormap{i});
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
initialPoints = cell(1, 7);
for i = 1:7
    in = inpolygon(points.Location(:,1), points.Location(:, 2), region{i}(:, 1), region{i}(:, 2));
    initialPoints{i} = points.Location(in, :);
end

% show tracked points
figure, imshow(refFrame), title('Detected interest points'), hold on;
for i = 1:7
    scatter(initialPoints{i}(:, 1), initialPoints{i}(:, 2), colormap{i})
end

%% track points

% initialize tracker for each region
trackers = cell(1, 7);

for i = 1:7
    trackers{i} = vision.PointTracker('MaxBidirectionalError', 1);
    initialize(trackers{i}, initialPoints{i}, refFrame);
end

trackedPointsX = cell(1, 7); % X coordinates
trackedPointsY = cell(1, 7); % Y coordinates
% loop through frames
for i = 1:5
      frame = imread(strcat(path, num2str(1211 + i), '.jpg'));
      figure, imshow(frame), hold on
      for j = 1:7
      [points, validity] = step(trackers{j}, frame);
      trackedPointsX{j} = cat(1, trackedPointsX{j}, points(:, 1)');
      trackedPointsY{j} = cat(1, trackedPointsY{j}, points(:, 2)');
      scatter(points(:, 1), points(:, 2), colormap{j})
      end
end