function [ ret ] = getCVPR2014Video(subjectNum, i1, i2)
    % Load Video
    Fs = 30;
    refFrame = 0;
    GTPR = 0;
    path = sprintf('CVPR2014Data/%i', subjectNum);
    
    if subjectNum == 1
        NFrames = 3200;
        startFrame = 0;
        GTPR = 74.2129;
    elseif subjectNum == 2
        NFrames = 3600;
        startFrame = 0;
        GTPR = 74.3426;
    elseif subjectNum == 3
        NFrames = 900;
        startFrame = 1210;
        GTPR = 61.4943;
    elseif subjectNum == 4
        NFrames = 900;
        startFrame = 890;
        GTPR = 71.2081;
    end
    ret = struct();
    ret.NFrames = NFrames;
    i2 = min(i2, NFrames);
    
    NFrames = i2-i1+1;
    files = cell(1, NFrames);
    for ii = 1:NFrames
        fNum = i1+startFrame+ii-1;
        if subjectNum < 3
            filename = sprintf('%s/%.5i.jpg', path, fNum);
        else
            filename = sprintf('%s/%.4i.jpg', path, fNum);
        end
        fprintf(1, 'Loading frame %i...\n', fNum);
        frame = imread(filename);
        files{ii} = filename;
        if ii == 1
            refFrame = frame;
            I = zeros(NFrames, numel(refFrame));
        end
        I(ii, :) = frame(:);
    end
    
    ret.I = I;
    ret.files = files;
    ret.refFrame = refFrame;
    ret.Fs = Fs;
    %The best we can do for this dataset is to make a flat time series with 
    %the average pulse rate.  Make 1000 samples per second like the BU
    %dataset
    ret.GTPR = GTPR*ones(1, ceil(1000*NFrames/Fs));
end

