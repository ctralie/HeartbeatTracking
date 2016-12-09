function [ I ] = doAffineWarpVideo( I, refFrame, Keypoints, V, DOWARPPLOT, warpoffset )
    if nargin < 5
        DOWARPPLOT = 0;
    end
    lm = Keypoints{1};
    xr = [min(lm(:, 1)), max(lm(:, 1))];
    yr = [min(lm(:, 2)), max(lm(:, 2))];
    %Do affine warping of all frames
    %First calculate the triangle indices and barycentric coordinates
    %of pixel locations in the original frame
    D1 = delaunayTriangulation(Keypoints{1});
    [X, Y] = meshgrid(1:size(refFrame, 2), 1:size(refFrame, 1));
    P = [X(:) Y(:)];
    triIdx = D1.pointLocation(P);
    refFrameIdx = 1:size(P, 1);
    P = P(~isnan(triIdx), :);
    refFrameIdx = refFrameIdx(~isnan(triIdx));
    triIdx = triIdx(~isnan(triIdx));
    bary = D1.cartesianToBarycentric(triIdx, P);

    for ii = 1:length(Keypoints)
        FOrig = reshape(I(ii, :), size(refFrame));
        F = AffineWarp(D1, triIdx, bary, refFrameIdx, FOrig, Keypoints{ii});
        writeVideo(V, uint8(F(yr(1):yr(2), xr(1):xr(2), :)));
        I(ii, :) = F(:); %Copy back warped frame
        if DOWARPPLOT
            clf;
            subplot(121);
            imagesc(uint8(FOrig));
            hold on;
            lm = Keypoints{ii};
            plot(lm(:, 1), lm(:, 2), '.');
            axis off;
            subplot(122);
            imagesc(uint8(F));
            xlim(xr);
            ylim(yr);
            axis off;
            fig = gcf;
            fig.PaperUnits = 'inches';
            fig.PaperPosition = [0 0 12 6];
            print('-dpng', '-r100', sprintf('%i.png', ii + warpoffset));
        end
    end
    

end

