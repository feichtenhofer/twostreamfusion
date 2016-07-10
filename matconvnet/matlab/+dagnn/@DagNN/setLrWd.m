function net = setLrWd(net, varargin)

import dagnn.*

opts.filtersLRWD = [1 1];
opts.biasesLRWD = [2 0];

opts.convFiltersLRWD = [1 1];
opts.convBiasesLRWD = [2 0];

opts.fusionFiltersLRWD = [1 1];
opts.fusionBiasesLRWD = [2 0];

opts = vl_argparse(opts, varargin);

for l = 1:numel(net.layers)
  paramsIdx = net.getParamIndex(net.layers(l).params);
  for p = 1:numel(paramsIdx)
    sz = size(net.params(paramsIdx(p)).value);
    switch class(net.layers(l).block)
      case 'dagnn.BatchNorm'
          net.params(paramsIdx(p)).value = squeeze(net.params(paramsIdx(p)).value);
          if p == 1
            net.params(paramsIdx(p)).learningRate = 2;
            net.params(paramsIdx(p)).weightDecay = 0;
          elseif p==2
            net.params(paramsIdx(p)).learningRate = 1;
            net.params(paramsIdx(p)).weightDecay = 0;       
          elseif p==3
            net.params(paramsIdx(p)).learningRate = 0.05;
            net.params(paramsIdx(p)).weightDecay = 0;       
          end
      otherwise
        if p == 1
          net.params(paramsIdx(p)).learningRate = 1;
          net.params(paramsIdx(p)).weightDecay = 1;
        elseif p==2
          net.params(paramsIdx(p)).learningRate = 2;
          net.params(paramsIdx(p)).weightDecay = 0;       
        end
      end
  end
end
