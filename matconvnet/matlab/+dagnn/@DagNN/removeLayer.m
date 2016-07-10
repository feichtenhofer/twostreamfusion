function removeLayer(obj, name, chainInputs)
%REMOVELAYER Remove a layer from the network
%   REMOVELAYER(OBJ, NAME) removes the layer NAME from the DagNN object
%   OBJ.

% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if nargin < 3, chainInputs = true; end
f = find(strcmp(name, {obj.layers.name})) ;
if isempty(f), error('There is no layer ''%s''.', name), end
layer = obj.layers(f) ;

if chainInputs
  % chain input of l that has layer as input
  for l = 1:numel(obj.layers)    
    for i = layer.outputIndexes
      sel = find(intersect(obj.layers(l).inputIndexes, i));
      if any(sel)
        obj.layers(l).inputs{sel} = layer.inputs{1};
      end
    end
  end
end
obj.layers(f) = [] ;
obj.rebuild() ;

