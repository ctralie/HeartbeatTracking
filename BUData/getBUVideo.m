%Purpose: Load BU Data
%First10Or30: If 0, first 10 subjects, if 1, next 30 subjects
%subjectDir: The subdirectory path within the root directory
%i1:i2: Frame range

%Returns: All of the usual stuff, plus "GTPR" (the ground truth pulse rate
%sampled at 1000 samples / second)
function [ ret ] = getBUVideo(First10Or30, subjectDir, i1, i2)
    if First10Or30 == 0
        imgPrefix = 'BUData/first10subjects_2D';
        gtPrefix = 'BUData/first10subjects_Phydatareleased/Phydata';
    else
        imgPrefix = 'BUData/T10_T11_30Subjects';
        gtPrefix = 'BUData/T10_T11_30PhyBPHRData';
    end
    %Count the number of frames actually in this video
    filesDir = dir(imgpath);
    NFrames = 0; %Figure out number of files
    for ii = 1:length(filesDir)
        [~, ~, fext] = fileparts(filesDir(ii).name);
        if strcmp(fext, '.jpg') == 1
            NFrames = NFrames + 1;
        end
    end
    ret = struct();
    ret.NFrames = NFrames;
    i2 = min(i2, NFrames);
    
    %Load ground truth pulse rate
    gtFilename = sprintf('%s/%s/Pulse Rate_BPM.txt', gtPrefix, subjectDir);
    GTPR = load(gtFilename);
    % Load Video
    Fs = 25; %According to docs, framerate for RGB is 25/s, framerate for heartrate is 1000/s
    refFrame = 0;
    imgpath = sprintf('%s/%s', imgPrefix, subjectDir);
    NFrames = i2-i1+1;

    %Load in video frames
    files = cell(1, NFrames);
    for ii = 1:NFrames
        filename = sprintf('%s/%.3i.jpg', imgpath, i1+ii-1);
        fprintf(1, 'Loading frame %i...\n', i1+ii-1);
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
    ret.GTPR = GTPR;
end

