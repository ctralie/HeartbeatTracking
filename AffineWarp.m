%D1: Delaunay object for the first frame, to which this is being warped
%IM: Image
%X: Keypoints
function [J] = AffineWarp(D1, triIdx, bar, refFrameIdx, I, X)
    D2 = triangulation(D1.ConnectivityList, X);
    PNew = D2.barycentricToCartesian(triIdx, bar);
    
    %Do warping to the output image by pulling back map from original frame
    %to this frame
    J = 0*I;
    for kk = 1:3
        Ikk = I(:, :, kk);
        val = interp2(Ikk, PNew(:, 1), PNew(:, 2));
        Jkk = zeros(size(Ikk));
        Jkk(refFrameIdx) = val;
        J(:, :, kk) = Jkk;
    end
end
