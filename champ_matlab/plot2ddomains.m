function plot2ddomains(ind2domain)
    % Plots the projection of the domains onto the (gamma, omega) plane,
    % using the ind2domain map returned by getintersection.
    hold on;
    for i = cell2mat(keys(ind2domain))
        vs = ind2domain(i);
        x = vs(:, 1);
        y = vs(:, 2);
        % Sort polygon vertices by angle in order to plot.
        [~, j] = sort(angle(complex(x - mean(x), y - mean(y))));
        x = x(j);
        y = y(j);
        % Random polygon colors.
        p = patch(x, y, rand());
        p.LineWidth = 2;
    end
end