function [normals, offsets, vertices, facets] = halfspaceintersection(normals, offsets, interiorPoint)
    % Calculates intersection of the halfspaces in normal-offset form and
    % interiorPoint inside this intersection.
    %
    %   normals is a matrix where each row is a halfspace's normal vector.
    %   offsets is a matrix where each row is a halfspace's offset.
    %   interiorPoint is a point inside the halfspace intersection.
    %
    %   The rows of normals and offsets correspond; that is, if the ith row
    %   of normals is [a,b,c] and the ith row of offsets is d, one of our
    %   halfspaces is ax+by+cz+d <= 0.

    % Verify interior point is actually inside the intersection.
    for i = 1:size(normals, 1)
        assert(dot(interiorPoint, normals(i, :)) + offsets(i) < 0, ...
        'Impossible interior point.')
    end

    % Normalize halfspace normals.
    norm_offsets = offsets ./ sqrt(sum(normals .^ 2, 2));
    norm_normals = normals ./ sqrt(sum(normals .^ 2, 2));

    % Translate interior point to origin.
    norm_offsets = norm_offsets + sum(norm_normals .* interiorPoint, 2);

    % Dualize halfspaces (e.g. ax+by+cz+d <= 0 becomes (a/d, b/d, c/d)).
    dual_points = norm_normals ./ norm_offsets;

    % Convex Hull of dual points and get corresponding normals, offsets.
    hull = convhulln(dual_points);
    hull_indices = unique(hull);

    % Get the points at halfspace intersections.
    vertices = zeros(size(hull));
    for i = 1:size(hull, 1)
        % Get vertices of the dual facet.
        vs = dual_points(hull(i, :), :);
        % Get corresponding intersection and translate back.
        vertices(i, :) = dualizeintersection(vs) + interiorPoint;
    end
    vertices = unique(vertices, 'rows');

    % Facets will map the halfspace index to all vertices in the
    % intersection that lie on the halfspace boundary.
    facets = containers.Map('KeyType', 'int64', 'ValueType', 'any');
    for i = 1:size(hull_indices)
        h = hull_indices(i);
        for j = 1:size(vertices, 1)
            v = vertices(j, :);
            % Adding vertex to facet if it lies on this halfspace.
            % This can definitely be done in the previous loop (for adding
            % vertices), but I wasn't getting consistent results.
            if abs(dot(normals(h, :), v) + offsets(h)) < 1e-10
                if isKey(facets, h)
                    facets(h) = [facets(h); v];
                else
                    facets(h) = v;
                end
            end
        end
    end

    normals = normals(hull_indices(:, 1), :);
    offsets = offsets(hull_indices(:, 1), :);
end
