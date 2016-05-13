subject = '4_15_george_ir_depth_30';
foldername = ['GeorgeData/', subject, '/ir'];
NFrames = 300;

%Load IR video
V = {};
for ii = 1:NFrames
    ii
    V = rgb2gray(imread(sprintf('%s/ir_%.4d.png', foldername, 0)));
    V{end+1} = V;
end
V = cell2mat(V);
V = reshape(V, [480, 640, 300]);

%Process blocks
res = 20;