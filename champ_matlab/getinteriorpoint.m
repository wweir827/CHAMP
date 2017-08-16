function interiorPoint = getinteriorpoint(normals, offsets)
    % Find an interior point of the halfspaces given by normals, offsets.
    zMax = -inf;
    for i = 1:size(normals, 1)
        % If the halfspace is non-vertical.
        if abs(normals(i, end)) > 1e-15
            zMax = max(zMax, -offsets(i) / normals(i, end));
        end
    end

    dim = size(normals, 2);
    % Take a very small step above the highest plane at 0,0.
    interiorPoint = [zeros(1, dim - 1), zMax] + 1e-6 * ones(1, dim);
end
