%Do a spectral embedding with a normalized symmetric Laplacian which comes
%from an exponential similarity matrix with an autotuned distance threshold
%based on nearest neighbors
function [V, e] = SpectralEmbed( X )
    %First, compute all pariwise L2 distances using fast matlab trick
    XSqr = sum(X.^2, 2);
    DSqr = bsxfun(@plus, XSqr(:), XSqr(:)') - 2*(X*X');
    DSqr(DSqr < 0) = 0;
    D = sqrt(DSqr);
    
    %Compute average neighborhood distances and compute exponential
    %embedding
    Sigma = mean(D, 2)/3;
    W = exp(bsxfun(@times, -D.^2, 1./(2*Sigma.^2)));
    Diag = sum(W, 2);
    DSqrt = diag(1./sqrt(Diag));
    L = eye(size(DSqrt, 1)) - DSqrt*W*DSqrt;
    
    %Do spectral embedding
    [V, e] = eigs(L, 3, 'sm');
    e = diag(e);
    [~, idx] = sort(e);
    %Sort from smallest to largest
    V = V(:, idx);
end

