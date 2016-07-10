function diagnose(obj, varargin)
%DIAGNOSE  Plot diagnostic information on a DagNN network.
%   DIAGNOSE plots in the current window the average, maximum, and miminum
%   element for all the variables and parameters and their derivatives (if
%   present) of a network.
%
%   This function can be used to rapidly glance at the evolution
%   of the paramters during training. Furthermore it visually shows when a
%   particular parameter or a variable contains a NaN value (red asterisk).
%
%   DIAGNOSE(___, 'OPT', VAL, ...) accepts the following options:
%
%   `Vars`:: `true`
%   When true, plot diagnostics of network variables and its derivatives.
%
%   `Params`:: `true`
%   When true, plot diagnostics of network parameters and its derivatives.
%
%   `Time`:: `false`
%   When true, plot the forward and bacward time (when avaialable) per
%   layer.

% Copyright (C) 2014-16 Karel Lenc Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
opts.vars = true;
opts.params = true;
opts.time = false;
opts.skipvars = {'objective'};
opts = vl_argparse(opts, varargin);

toplot = {};
if opts.params
  pnames = {obj.params.name};
  pstats = collectstats({obj.params.value});
  toplot{end+1} = {pstats, pnames, 'params'};
  pderstats = collectstats({obj.params.der});
  if ~all(isnan(pderstats(:)))
    toplot{end+1} = {pderstats, pnames, 'dzdparams'};
  end
end

if opts.vars
  vnames = {obj.vars.name};
  vstats = collectstats({obj.vars.value});
  if ~all(isnan(vstats(:)))
    vstats(1:3, ismember(vnames, opts.skipvars)) = nan;
    toplot{end+1} = {vstats, vnames, 'variables'};
  end
  vderstats = collectstats({obj.vars.der});
  if ~all(isnan(vderstats(:)))
    toplot{end+1} = {vderstats, vnames, 'dzdvariables'};
  end
end

if opts.time
  lnames = {obj.layers.name};
  fwtstats = collectstats({obj.layers.forwardTime});
  if ~all(isnan(fwtstats(:)))
    toplot{end+1} = {fwtstats, lnames, 'FW Time'};
  end
  bwtstats = collectstats({obj.layers.backwardTime});
  if ~all(isnan(bwtstats(:)))
    toplot{end+1} = {bwtstats, vnames, 'BW Time'};
  end
end
% Plot all the collected results
clf;
for pli = 1:numel(toplot)
  subplot(numel(toplot), 1, pli);
  plotstats(toplot{pli}{:});
end
drawnow;
end

function stats = collectstats(data)
% Collect some data statistics, in this case [min, mean, max, isnan]
np = numel(data) ;
stats = nan(4, np) ;
for i=1:np
  x = gather(data{i}) ;
  if isempty(x), continue; end;
  stats(:, i) = [min(x(:)), mean(x(:)), max(x(:)), any(isnan(x(:)))] ;
end
end

function plotstats(stats, names, plottitle)
% Plot a single results set
assert(size(stats,2) == numel(names));
range = max(max(max(abs(stats(1:3,:))))*1.1, eps);
errorbar(1:size(stats,2), stats(2,:), stats(2,:) - stats(1,:), stats(3,:) - stats(2,:), 'bo');
set(gca, 'ylim', [-range, range]);
nan_values = find(stats(4, :) == 1);
if ~isempty(nan_values)
  hold on;
  plot(nan_values, zeros(1, numel(nan_values)), 'r*', 'MarkerSize', 10, 'LineWidth', 3);
end
set(gca, 'xtick', 1:size(stats,2));
set(gca, 'xtickLabel', names);
if ~verLessThan('matlab', '8.4'), set(gca, 'XTickLabelRotation', 90); end
grid on;
title(plottitle);
end