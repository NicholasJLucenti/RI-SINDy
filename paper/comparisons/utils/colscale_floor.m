function colscale = colscale_floor(colscale, floorFraction)
% COLSCALE_FLOOR  Floor near-zero library column norms to prevent SR3's
% Xi update (which divides by column norm to recover physical scale)
% from arithmetically exploding when a Hill/regulatory column collapses
% to near-zero norm during hyperparameter search.
%
%   colscale = COLSCALE_FLOOR(colscale)
%   colscale = COLSCALE_FLOOR(colscale, floorFraction)
%
% floorFraction (optional, default 1e-3) sets the floor at
% floorFraction * median(nonzero colscale). Any column below that floor
% is raised to the floor value; well-conditioned columns are untouched
% since the floor sits orders of magnitude below a normal column norm.
% Used identically across the Hes1, Goodwin, and NF-kB SR3 fits.
    if nargin < 2, floorFraction = 1e-3; end
    floorScale = floorFraction * median(colscale(colscale > 0));
    if isempty(floorScale) || ~isfinite(floorScale) || floorScale == 0
        floorScale = 1e-6;
    end
    colscale(colscale < floorScale) = floorScale;
end