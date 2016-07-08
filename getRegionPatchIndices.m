function [ patches ] = getRegionPatchIndices( regions, dims, res )
    %dims: Dimensions of image
    %res: Resolution of patches
    idx = 1:prod(dims);
    idx = reshape(idx, dims);
    I = zeros(dims);
    for ii = 1:length(regions)
        r = regions{ii};
        I = I + repmat(poly2mask(r(:, 1), r(:, 2), dims(1), dims(2)), [1, 1, 3]);
    end
    patches = {};
    for ii = 1:res:dims(1)
        if ii + res > dims(1)
            continue;
        end
        for jj = 1:res:dims(2)
            if jj + res > dims(2)
                continue;
            end
            if sum(I(ii:ii+res-1, jj:jj+res-1, :)) > 0
                idxs = idx(ii:ii+res-1, jj:jj+res-1, :);
                idxs = reshape(idxs, [res*res, dims(3)]);
                patches{end+1} = idxs;
            end
        end
    end
    patches = reshape(patches, [1, 1, length(patches)]);
    patches = cell2mat(patches);
    patches = shiftdim(patches, 2);
end

