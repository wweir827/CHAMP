function [normals, offsets] = createhalfspacesfromarray(coefArray)
    % Returns halfspaces described by coefficient array in normal-offset
    % form.
    singleLayer = size(coefArray, 2) == 2;
    numHalfspaces = size(coefArray, 1);

    cConsts = coefArray(:, 1);
    cGammas = coefArray(:, 2);
    if ~singleLayer
        cOmegas = coefArray(:, 3);
    end

    if singleLayer
        normals = [cGammas, ones(numHalfspaces, 1)];
        pts = [zeros(numHalfspaces, 1), cConsts];
    else
        normals = [cGammas, -cOmegas, ones(numHalfspaces, 1)];
        pts = [zeros(numHalfspaces, 1), zeros(numHalfspaces, 1), cConsts];
    end

    normals = normals ./ sqrt(sum(normals .^ 2, 2));
    offsets = sum(normals .* pts, 2);
    normals = -normals;
end
