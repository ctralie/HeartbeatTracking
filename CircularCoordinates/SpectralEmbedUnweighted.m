%Do the unweighted Laplacian based on a threshold Kappa and mutual nearest
%neighbors within that threshold
function [V, A, e] = SpectralEmbedUnweighted( D, Kappa )
    %Get graph laplacian
    A = groundTruthKNN(D, round(Kappa*size(D, 2)));
    A = A.*groundTruthKNN(D', round(Kappa*size(D, 1)))';
    A = A - speye(size(A, 1));
    L = spdiags(sum(A, 2), 0, size(A, 1), size(A, 2)) - A;


    %Get eigenvectors
    V = zeros(size(L, 1), 3);
    e = zeros(1, 3);
    try
        [V, e] = eigs(L, 3, 'sm');
        e = diag(e);
        [~, idx] = sort(e);
        %Sort from smallest to largest
        V = V(:, idx);
    catch ME
        disp('Error');
        ME
    end
end

