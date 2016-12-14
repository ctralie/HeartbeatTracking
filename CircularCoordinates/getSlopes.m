function [slopes] = getSlopes( thetas, sWin )
    if nargin < 2
        sWin = 10;
    end
    N = length(thetas);
    slopes = zeros(1, N);
    deriv = zeros(1, sWin*2);
    deriv(1:sWin) = ones(1, sWin);
    deriv(sWin+1:end) = -ones(1, sWin);
    slopes(sWin:end-sWin) = conv(thetas, deriv, 'valid')/(sWin^2);
    slopes(1:sWin-1) = slopes(sWin);
    slopes(end-sWin+1:end) = slopes(end-sWin);
end

