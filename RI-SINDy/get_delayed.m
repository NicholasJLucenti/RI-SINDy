function y_delayed = get_delayed(t, y, tau)
    y_delayed = interp1(t, y, max(t - tau, 0), 'linear');
end