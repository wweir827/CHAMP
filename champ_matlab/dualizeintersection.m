function v = dualizeintersection(vs)
    % Dualizes polytope vertices to halfspaces.

    % Get normal of dual facet.
    if size(vs, 2) == 2 % 2D
        dx = vs(1, 1) - vs(2, 1);
        dy = vs(1, 2) - vs(2, 2);
        normal = [-dy, dx];
    elseif size(vs, 2) == 3 % 3D
        U = vs(2, :) - vs(1, :);
        V = vs(3, :) - vs(1, :);
        normal = -cross(U, V);
    end

    % Ensure the normal is outward.
    if dot(normal, -vs(1, :)) > 0
        normal = -normal;
    end

    % Get offset and normalize outward normal of dual facet.
    normal = normal / norm(normal);
    offset = norm(normal * dot(normal, vs(1, :)));

    % Dualize dual facet to corresponding halfspace intersection point.
    v = normal / -offset;
end
