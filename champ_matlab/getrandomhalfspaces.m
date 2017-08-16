function halfspaces = getrandomhalfspaces(n, dim)
    % Generate n random halfspaces of dimension dim for testing.
    if dim == 3
        % Random halfspaces with normals [x,y,1] with -0.5 < x,y < 0,5
        % and random offset such that the plane restricted to [0,1]x[0,1]
        % has z-values within [0,1].
        normals = [rand(n, 1) - 0.5, rand(n, 1) - 0.5, -ones(n, 1)];
        normals = normals ./ sqrt(sum(normals .^ 2, 2));
        offsets = zeros(n, 1);
        for i = 1:n
            minOffset = -min([0, ...
                              normals(i, 1), ...
                              normals(i, 2), ...
                              normals(i, 1) + normals(i, 2)]);
            maxOffset = -max([normals(i, 3), ...
                              normals(i, 3) + normals(i, 1), ...
                              normals(i, 3) + normals(i, 2), ...
                              normals(i, 3) + normals(i, 2) + normals(i, 1)]);
            % 0.75 and 0.25 factors force more intersections. Random offset between
            % 0.75*minOffset+0.25*maxOffset and 0.25*minOffset+0.75*maxOffset
            offsets(i) = 0.5 * (maxOffset - minOffset) * rand() + (0.75 * minOffset + 0.25 * maxOffset);
        end
        halfspaces = [normals(:, 1:2), -1 * offsets ./ normals(:, 3)];
    elseif dim == 2
        % Random lines with 1/5 < slope < 5 and 0 < intercept < 2
        slopes = (5 - 1/5) * rand(n, 1) + 1/5;
        intercepts = 2 * rand(n, 1);
        halfspaces = [intercepts, slopes];
    else
        error('Only 2D and 3D halfspaces implemented.');
    end
end