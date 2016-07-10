function net = insertLossLayers(net, varargin)

import dagnn.*
opts.numClasses = 101;
opts = vl_argparse(opts, varargin);

for p = 1:numel(net.params)
  name = net.params(p).name ;
  sz = size(net.params(p).value);
  if numel(sz) == 4
    if sz(4) == opts.numClasses
      lossAlreadyExists = false;
        for l=1:numel(net.layers)
          if ~isempty(net.layers(l).params)
            if any(strcmp(name, net.layers(l).params(:))), break; end
          end
        end
        pred_layer = l;
        for l = 1:numel(net.layers)
          if any(strcmp(net.layers(pred_layer).outputs, net.layers(l).inputs))  && ...
              isa(net.layers(l).block, 'dagnn.Loss') && isempty(strfind(net.layers(l).name, 'err'))
              lossAlreadyExists = true;
              break;
          end
        end
        if lossAlreadyExists, continue; end;
        name = sprintf('loss_%s',net.layers(pred_layer).name);
        net.addLayer(name, ...
          dagnn.Loss( 'loss', 'softmaxlog'), ...
             [net.layers(pred_layer).outputs{:} {'label'}], sprintf('objective_%s',net.layers(pred_layer).name)) ; 
    end
  end

end
