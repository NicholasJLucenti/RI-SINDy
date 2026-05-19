function dxdt = smooth_derivative(x, dt, sg_p, sg_f)
    dxdt = sgolayfilt(gradient(x, dt), sg_p, sg_f);
end
