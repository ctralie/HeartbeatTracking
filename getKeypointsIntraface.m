%Use the intraface library
function [I, paramsout] = getKeypointsIntraface(filename, params)
    % read image from input file
    im=imread(filename);
    dims = size(im);
    
    if nargin < 2
        [DM,TM,option] = xx_initialize;
        paramsout.DM = DM;
        paramsout.TM = TM;
        paramsout.option = option;
        pred = 0;
    else
        paramsout = params;
        DM = params.DM;
        TM = params.TM;
        option = params.option;
        pred = params.pred;
    end
    
    output = xx_track_detect(DM,TM,im,pred,option);
    paramsout.pred = output.pred;
    I = zeros(size(output.pred, 1), 2);
    if size(I, 1) > 0
        I(:, 1) = dims(1) - output.pred(:, 2);
        I(:, 2) = output.pred(:, 1);
    end
end