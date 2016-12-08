function [ I, files, refFrame, Fs, groundTruthMean ] = getGeorgeIRData( dirName, NFrames )
    Fs = 25; %This can vary on RealSense so this is an approximation
    groundTruthMean = load(sprintf('%s/heartRate.txt', dirName));
    files = cell(1, NFrames);
    for ii = 1:NFrames
        fprintf(1, 'Loading frame %i...\n', ii);
        filename = sprintf('%s/ir/ir_%.4i.png', dirName, ii-1);
        files{ii} = filename;
        frame = imread(filename);
        if ii == 1
            refFrame = frame;
            I = zeros(NFrames, numel(refFrame));
        end
        I(ii, :) = frame(:);
    end
end

