%thetas1, thetas2: Two different NBlocks x SamplesPerBlock matrices of
%circular coordinates in each block
%BlockHop: The hop size between blocks in samples
%Fs: Sample rate
%fl, fh: Heartrate band limits in hz
function [thetaFinal, BlocksAligned, rs] = mergeCircularCoordinates( thetas1, thetas2, BlockHop, Fs, fl, fh, doPlot )
    if nargin < 7
        doPlot = 0;
    end
    NBlocks = size(thetas1, 1);
    SamplesPerBlock = size(thetas1, 2);
    TotalSamples = BlockHop*(NBlocks-1) + SamplesPerBlock;
    
    thetas = zeros(size(thetas1)); %Final estimate of thetas
    
    %Array to hold aligned blocks, which makes it easy to take the medians
    BlocksAligned = NaN*ones(NBlocks, TotalSamples);
    
    rs = zeros(1, NBlocks); %Linear regression coefficients
    
    if doPlot
        hold on;
    end
    
    %Choose which cluster has a better correlation coefficient, and
    %put that cluster in the proper spot
    for ii = 1:NBlocks
        idx = (1:SamplesPerBlock) + (ii-1)*BlockHop;
        [slope1, RSqr1] = linreg(thetas1(ii, :), Fs);
        [slope2, RSqr2] = linreg(thetas2(ii, :), Fs);
        if slope1 < fl || slope1 > fh
            RSqr1 = 0;
        end
        if slope2 < fl || slope2 > fh
            RSqr2 = 0;
        end
        if RSqr1 > RSqr2
            theta = thetas1(ii, :);
            rs(ii) = RSqr1;
        else
            theta = thetas2(ii, :);
            rs(ii) = RSqr2;
        end
        if rs(ii) > 0
            %If at least one of the blocks has a slope within the desired
            %range, then median align it to the previous block
            if ii > 1
                kk = ii - 1;
                %Find the first valid block that overlaps
                while kk >= 1
                    if rs(kk) > 0
                        t1 = thetas(kk, BlockHop*(ii-kk)+(1:SamplesPerBlock-BlockHop*(ii-kk)));
                        t2 = theta(1:SamplesPerBlock-BlockHop*(ii-kk));
                        med = median(median(t1 - t2));
                        theta = theta + med;
                        break;
                    end
                    kk = kk - 1;
                end
            end
            %Put the block into its place
            thetas(ii, :) = theta;
            BlocksAligned(ii, idx) = theta;
        end
        if doPlot
            if rs(ii) > 0
                plot(idx, theta);
            end
        end
    end
    thetaFinal = nanmedian(BlocksAligned, 1);
end

