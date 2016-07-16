function [ net ] = insert_conv_layers( net, layer_idx, varargin )

opts.weightDecay = 1 ;
opts.initMethod = 'xavierimproved';
opts.dropOutRatio = 0;
opts.name = '_fusion';
opts.batchNormalization = false;
opts.addReLU = true;
opts.addBiases = true;
opts.autoDims = false;
opts = vl_argparse(opts, varargin) ;

if isa(net,'dagnn.DagNN')
for k=numel(layer_idx):-1:1
   i = layer_idx(k);
   if isa(net.layers(i).block,'dagnn.Conv')
    out = size(net.params( net.getParamIndex( net.layers(i).params{1})).value,4);
    in = size(net.params( net.getParamIndex( net.layers(i).params{1})).value,3);
    
   else %relu
    out = size(net.params( net.getParamIndex( net.layers(i-1).params{1})).value,4);
    in = size(net.params( net.getParamIndex( net.layers(i-1).params{1})).value,3);

   end

   layerNr = num2str(k);
   name = [opts.name '_' layerNr];
   if ~opts.autoDims, in=2*out; end;
    blocks = add_conv_block(opts, name, 1, 1, in, out, 1, 0) ;
    for l =1:numel(blocks)  
         params = struct(...
        'name', {}, ...
        'value', {}, ...
        'learningRate', [], ...
        'weightDecay', []) ;
          if isfield(blocks{l}, 'name')
            name = blocks{l}.name ;
          else
            name = sprintf('layer%d',l) ;
          end
          switch blocks{l}.type
            case 'conv'
            sz = size(blocks{l}.weights{1}) ;
            hasBias = ~isempty(blocks{l}.weights{2}) ;
            params(1).name = sprintf('%sf',name) ;
            params(1).value = blocks{l}.weights{1} ;
            if hasBias
              params(2).name = sprintf('%sb',name) ;
              params(2).value = blocks{l}.weights{2} ;
            end
              if isfield(blocks{l},'learningRate')
                params(1).learningRate = blocks{l}.learningRate(1) ;
                if hasBias
                  params(2).learningRate = blocks{l}.learningRate(2) ;
                end
              end
              if isfield(blocks{l},'weightDecay')
                params(1).weightDecay = blocks{l}.weightDecay(1) ;
                if hasBias
                  params(2).weightDecay = blocks{l}.weightDecay(2) ;
                end
              end
          end
          switch blocks{l}.type
            case 'conv'
              block = dagnn.Conv() ;
              block.size = sz ;
              if isfield(blocks{l},'pad')
                block.pad = blocks{l}.pad ;
              end
              if isfield(blocks{l},'stride')
                block.stride = blocks{l}.stride ;
              end
          case {'relu'}
              lopts = {} ;
              if isfield(blocks{l}, 'leak'), lopts = {'leak', blocks{l}} ; end
              block = dagnn.ReLU('opts', lopts) ;
          case {'bnorm'}
              block = dagnn.BatchNorm() ;
              if isfield(blocks{l},'filters')
                params(1).name = sprintf('%sm',name) ;
                params(1).value = blocks{l}.filters ;
                params(2).name = sprintf('%sb',name) ;
                params(2).value = blocks{l}.biases ;
              else
                params(1).name = sprintf('%sm',name) ;
                params(1).value = blocks{l}.weights{1} ;
                params(2).name = sprintf('%sb',name) ;
                params(2).value = blocks{l}.weights{2} ;
              end
              if isfield(blocks{l},'learningRate')
                params(1).learningRate = blocks{l}.learningRate(1) ;
                params(2).learningRate = blocks{l}.learningRate(2) ;
              end
              if isfield(blocks{l},'weightDecay')
                params(1).weightDecay = blocks{l}.weightDecay(1) ;
                params(2).weightDecay = blocks{l}.weightDecay(2) ;
              end
          end
            net.addLayerAt(i,...
            name, ...
            block, ...
            net.layers(i).outputs, ...
            {['x_' name]}, ...
            {params.name}) ;
            net.layers(i+2).inputs{1} = ['x_' name]; % hang it in between
            i = i + 1;

             if ~isempty(params)
                findex = net.getParamIndex(params(1).name) ;
                bindex = net.getParamIndex(params(2).name) ;

                % if empty, keep default values
                if ~isempty(params(1).value)
                  net.params(findex).value = params(1).value ;
                end
                if ~isempty(params(2).value)
                  net.params(bindex).value = params(2).value ;
                end
                if ~isempty(params(1).learningRate)
                  net.params(findex).learningRate = params(1).learningRate ;
                end
                if ~isempty(params(2).learningRate)
                  net.params(bindex).learningRate = params(2).learningRate ;
                end
                if ~isempty(params(1).weightDecay)
                  net.params(findex).weightDecay = params(1).weightDecay ;
                end
                if ~isempty(params(2).weightDecay)
                  net.params(bindex).weightDecay = params(2).weightDecay ;
                end
              end
    end
    
 end    
    
else

info = vl_simplenn_display(net);
net_dimensions = info.dataSize(3,:);

 for k=numel(layer_idx):-1:1
   i = layer_idx(k);
   out = net_dimensions(i+1);
   if isfield(net.layers{i}, 'name')
     layerNr = net.layers{i}.name(end-3:end);
   else
     layerNr = num2str(k);
   end
   name = [opts.name '_' layerNr];
   block = add_conv_block(opts, name, 1, 1, out*2, out, 1, 0) ;
                      

   if opts.dropOutRatio > 0
    dropout = struct('type', 'dropout', 'name', ['dropout_' name], ...
                               'rate', opts.dropOutRatio) ;
    net.layers = {net.layers{1:i} dropout block{:} net.layers{i+1:end}};
   else
    net.layers = {net.layers{1:i} block{:} net.layers{i+1:end}};
   end
   
 end
end
end
function conv_block = add_conv_block(opts, name, h, w, in, out, stride, pad)

conv_block = {};

filters = init_weight(opts, [h, w, in, out], 'single');
if opts.addBiases
  weights = {filters, zeros(1,out,'single')};
else
  weights = {filters, zeros(1,out,'single')};
end
conv_block{end+1} = struct('type', 'conv', 'name', sprintf('conv%s', name), ...
                           'weights', {weights}, ...
                           'stride', stride, ...
                           'pad', pad, ...
                           'learningRate', [1 2], ...
                           'weightDecay', [opts.weightDecay 0]) ;
if   opts.addReLU
  conv_block{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',name)) ;
end
if opts.batchNormalization
  conv_block{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%s',name), ...
                             'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single')}}, ...
                             'learningRate', [2 1], ...
                             'weightDecay', [0 0]) ;
end
end
% -------------------------------------------------------------------------
function weights = init_weight(opts, layer_sz, type)
% -------------------------------------------------------------------------
  h = layer_sz(1); w = layer_sz(2); in = layer_sz(3); out = layer_sz(4);

  switch opts.initMethod
    case 'classic'
      weights = 0.01 * randn(h, w, in, out, type) ;   
    case 'Xavier'
      sc = sqrt(3/(h*w*in));
      weights = (rand(h, w, in, out, type)*2 - 1)*sc ; 
    case 'xavierimproved'
      sc = sqrt(2/(h*w*out)) ;
      weights = sc*randn(h, w, in, out, type) ;     
    case 'xavierimproved_down_scaled'
      sc = sqrt(2/(h*w*out));
      weights = sc*randn(h, w, in, out, type) / 10 ;           
    case 'xavierimprovedB100'
      sc = sqrt(2/(h*w*out));
      weights1 = sc*randn(h, w, in/2, out, type)  ;           
      weights2 = sc*randn(h, w, in/2, out, type) * 100;           
      weights = cat(3, weights1,weights2) ;    
    case 'sum+xavierimprovedB100'
      sc = sqrt(2/(h*w*out));
      weights1 = sc*randn(h, w, in/2, out, type);           
      weights2 = sc*randn(h, w, in/2, out, type) * 100 ; 
      diag = permute(eye(in/2,out,type), [3 4 1 2]);
      weights = cat(3, 1/100 * diag + weights1, 1 * diag+ weights2) ;
    case 'diagA+Bxavierimproved'
      sc = sqrt(2/(h*w*out));
      weights2 = sc*randn(h, w, in/2, out, type)  ; 
      diag = permute(eye(in/2,out,type), [3 4 1 2]);
      weights = cat(3, diag, weights2) ;      
    case 'uniformly'
      sc = 1/(h*w*in);
      weights = sc*ones(h, w, in, out, type) ;     
    case 'netA'
      diag = permute(eye(in/2,out,type), [3 4 1 2]);
      weights = cat(3, diag, zeros(h, w, in/2, out, type)) ;        
    case 'netB'
      diag = permute(eye(in/2,out,type), [3 4 1 2]);
      weights = cat(3,zeros(h, w, in/2, out, type), diag) ;  
    case 'netAclassicNetBdiag'
      diag = permute(eye(in/2,out,type), [3 4 1 2]);
      weights = cat(3, 0.01 * randn(h, w, in/2, out, type), diag) ;  
    case 'sumAB'
      diag = permute(eye(in/2,out,type), [3 4 1 2]);
      weights = cat(3, 1 * diag, 1 * diag) ;  
    case '2sumAB'
      diag = permute(eye(in/2,out,type), [3 4 1 2]);
      weights = cat(3, 0.25 * diag, 0.75 * diag) ;  
    case '100sumAB'
      diag = permute(eye(in/2,out,type), [3 4 1 2]);
      weights = cat(3, 1/100 * diag, 1 * diag) ;       
    otherwise
      error('Uknown weight init method''%s''', opts.initMethod) ;
  end

end

