function ind2domain = getintersection(coefArray)
    % Calculates intersection of halfspaces described by coefficient array.
    %   Returns a map between the indices of the halfspaces
    %   contributing to the upper envelope and the vertex intersections
    %   of each halfspace.
    %
    %   That is, if the halfspace represented by the first row of the
    %   passed array is part of the upper envelope, then
    %   ind2domain(1) = [x_1, y_1, z_1; x_2, y_2, z_2; ...]
    %   where (x_i, y_i, z_i) are the vertices of the upper envelope lying
    %   on this first halfspace.
    [normals, offsets] = createhalfspacesfromarray(coefArray);
    interiorPoint = getinteriorpoint(normals, offsets);
    numHalfspaces = size(coefArray, 1);
    singleLayer = size(coefArray, 2) == 2;

    % Add some boundary halfspaces.
    if ~singleLayer
        % x > 0 and y > 0. No limits on z.
        boundaryNormals = [0, -1, 0; -1, 0, 0];
        boundaryOffsets = [0; 0];
    else
        % x > 0 and y > 0
        boundaryNormals = [-1, 0; 0, -1];
        boundaryOffsets = [0; 0];
    end

    % Imposing the x > 0 and y > 0 limits.
    normalsWithBoundary = [normals; boundaryNormals];
    offsetsWithBoundary = [offsets; boundaryOffsets];

    [~, ~, vertices, ~] = halfspaceintersection(normalsWithBoundary, offsetsWithBoundary, interiorPoint);

    nonInfiniteVertices = vertices(~any(isinf(vertices), 2), :);
    maxVertex = max(nonInfiniteVertices);

    % Checks on maximum limits of intersections.
    if abs(maxVertex(1)) < 1e-15 && abs(maxVertex(2)) < 1e-15
        error('Max intersection detected at (0,0). Invalid input set.')
    end
    if abs(maxVertex(2)) < 1e-15
        maxVertex(2) = maxVertex(1);
    end
    if abs(maxVertex(1)) < 1e-15
        maxVertex(1) = maxVertex(2);
    end

    if ~singleLayer
        % Imposing maximums x < x_max and y < y_max to deal with infinite
        % vertices in these two directions.
        normalsWithBoundary = [normalsWithBoundary; 1, 0, 0; 0, 1, 0];
        offsetsWithBoundary = [offsetsWithBoundary; -maxVertex(1); -maxVertex(2)];
        [~, ~, ~, facets] = halfspaceintersection(normalsWithBoundary, offsetsWithBoundary, interiorPoint);
    end

    ind2domain = containers.Map('KeyType', 'int64', 'ValueType', 'any');
    for i = cell2mat(keys(facets))
        % Don't include boundary halfspaces
        if i < numHalfspaces
            vs = facets(i);
            for j = 1:size(vs, 1)
                % Remove infinite vertices
                if ~any(isinf(vs(j, :)))
                    if isKey(ind2domain, i)
                        ind2domain(i) = [ind2domain(i); vs(j, :)];
                    else
                        ind2domain(i) = vs(j, :);
                    end
                end
            end
        end
    end
end
