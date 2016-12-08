function [ I, files, refFrame, Fs, groundTruthMean ] = getCVPR2014Video(subjectNum, MaxFrames)
    % Load Video
    Fs = 30;
    refFrame = 0;
    groundTruthMean = 0;
    path = sprintf('CVPR2014Data/%i', subjectNum);
    
    if subjectNum == 1
        NFrames = 3200;
        startFrame = 0;
        groundTruthMean = 74.2129;
    elseif subjectNum == 2
        NFrames = 3600;
        startFrame = 0;
        groundTruthMean = 74.3426;
    elseif subjectNum == 3
        NFrames = 900;
        startFrame = 1210;
        groundTruthMean = 61.4943;
    elseif subjectNum == 4
        NFrames = 900;
        startFrame = 890;
        groundTruthMean = 71.2081;
    end
    if nargin > 1
        NFrames = min(MaxFrames, NFrames);
    end

    files = cell(1, NFrames);
    for ii = 1:NFrames
        if subjectNum < 3
            filename = sprintf('%s/%.5i.jpg', path, ii+startFrame);
        else
            filename = sprintf('%s/%.4i.jpg', path, ii+startFrame);
        end
        fprintf(1, 'Loading frame %i...\n', ii);
        frame = imread(filename);
        files{ii} = filename;
        if ii == 1
            refFrame = frame;
            I = zeros(NFrames, numel(refFrame));
        end
        I(ii, :) = frame(:);
    end
end

