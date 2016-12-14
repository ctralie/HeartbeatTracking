%Do linear regression on the time series y
%Return slope in cycles per second
function [ slope, RSqr ] = linreg( y, Fs, doPlot )
    if nargin < 3
        doPlot = 0;
    end
    M = [ones(length(y), 1) (1:length(y))'];
    b = M \ y(:);
    ycalc = M*b;
    slope = Fs*b(2)/(2*pi);
    RSqr = 1 - sum((y(:)-ycalc(:)).^2)/sum((y(:)-mean(y)).^2);
    if doPlot
        plot(y);
        hold on;
        plot(ycalc);
        title(sprintf('RSqr = %g', RSqr));
    end
end

