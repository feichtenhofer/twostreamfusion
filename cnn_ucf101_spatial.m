function cnn_ucf101_spatial(varargin)

if ~isempty(gcp('nocreate')),
    delete(gcp)
end

opts = cnn_setup_environment();
opts.train.gpus =  1 ;
% opts.train.gpus = [ 1 : 3 ]

opts.dataSet = 'ucf101'; 
% opts.dataSet = 'hmdb51'; 

opts.train.memoryMapFile = fullfile(tempdir, 'ramdisk', ['matconvnet' num2str(feature('getpid')) '.bin']) ;
addpath('network_surgery');
opts.dataDir = fullfile(opts.dataPath, opts.dataSet) ;
opts.splitDir = [opts.dataSet '_splits']; 
opts.nSplit = 1 ;
opts.dropOutRatio = 0 ;
opts.train.cheapResize = 0 ; 
opts.inputdim  = [ 224,  224, 3] ;

opts.train.batchSize = 256  ;
opts.train.numSubBatches =  ceil(8 / max(numel(opts.train.gpus),1));

opts.train.epochFactor = 10 ;
opts.train.augmentation = 'borders25';

opts.train.backpropDepth =     cell(1, 2);
opts.train.backpropDepth(:) = {'pool5'};
opts.train.learningRate =  [1e-2*ones(1, 2) 1e-2*ones(1, 3) 1e-3*ones(1, 3) 1e-4*ones(1, 3)] ;
if strcmp(opts.dataSet, 'hmdb51')
  opts.train.learningRate =  [1e-2*ones(1, 2) 1e-2*ones(1, 1) 1e-3*ones(1, 1) 1e-4*ones(1, 1)] ;
end

model = ['img-res50-' opts.train.augmentation '-bs=' num2str(opts.train.batchSize) ...
  '-split' num2str(opts.nSplit) '-dr' num2str(opts.dropOutRatio)];

if  strfind(model, 'vgg16');
  baseModel = 'imagenet-vgg-verydeep-16.mat' ;
  opts.train.learningRate =  [1e-3*ones(1, 3) 5e-4*ones(1, 5) 5e-5*ones(1,2) 5e-6*ones(1,2)]  ;
  opts.train.backpropDepth =     cell(1, 3);
  opts.train.backpropDepth(:) = {'layer37'};
  opts.train.batchSize = 128  ;
  opts.train.numSubBatches =  ceil(16 / max(numel(opts.train.gpus),1));
elseif strfind(model, 'vgg-m');
  baseModel = 'imagenet-vgg-m-2048.mat' ;
elseif strfind(model, 'res152');
  baseModel = 'imagenet-resnet-152-dag.mat' ;
elseif strfind(model, 'res101');
  baseModel = 'imagenet-resnet-101-dag.mat' ;
elseif strfind(model, 'res50');
  baseModel = 'imagenet-resnet-50-dag.mat' ;
  opts.train.numSubBatches =  ceil(32 / max(numel(opts.train.gpus),1));
else
  error('Unknown model %s', model) ; 
end
opts.model = fullfile(opts.modelPath,baseModel) ;
opts.expDir = fullfile(opts.dataDir, [opts.dataSet '-' model]) ;
opts.imdbPath = fullfile(opts.dataDir, [opts.dataSet '_split' num2str(opts.nSplit) 'imdb.mat']);

opts.train.plotDiagnostics = 0 ;
opts.train.continue = 1 ;
opts.train.prefetch = 1 ;
opts.train.expDir = opts.expDir ;
opts.train.numAugments = 1;
opts.train.frameSample = 'random';
opts.train.nFramesPerVid = 1;
opts.train.uniformAugments  = false;

[opts, varargin] = vl_argparse(opts, varargin) ;


% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------
if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
else
  imdb = cnn_ucf101_setup_data('dataPath', opts.dataPath, 'flowDir',opts.flowDir, ...
    'dataSet', opts.dataSet, 'nSplit', opts.nSplit) ;
  save(opts.imdbPath, '-struct', 'imdb', '-v6') ;
end
nClasses = length(imdb.classes.name);
if ~exist(opts.model)
  fprintf('Downloading base model file: %s ...\n', baseModel);
  mkdir(fileparts(opts.model)) ;
  urlwrite(...
  ['http://www.vlfeat.org/matconvnet/models/' baseModel], ...
    opts.model) ;
end

net = load(opts.model);
if isfield(net, 'net'), net=net.net;end

if isstruct(net.layers)
  % replace 1000-way imagenet classifiers
  for p = 1 : numel(net.params)
    sz = size(net.params(p).value);
    if any(sz == 1000)
      sz(sz == 1000) = nClasses;
      fprintf('replace classifier layer of %s\n', net.params(p).name);
      if numel(sz) > 2
         net.params(p).value = 0.01 * randn(sz,  class(net.params(p).value));
      else
         net.params(p).value = zeros(sz,  class(net.params(p).value));
      end
    end
  end

  net.meta.normalization.border = [256 256] - net.meta.normalization.imageSize(1:2);
  net = dagnn.DagNN.loadobj(net);
  if strfind(model, 'bnorm')
    net = insert_bnorm_layers(net) ;
  end
else
%   net=vl_simplenn_tidy(net);
  if isfield(net, 'meta'),
    netNorm = net.meta.normalization;
  else
    netNorm = net.normalization;
  end
  if(netNorm.imageSize(3) == 3) && ~isempty(strfind(opts.model, 'imagenet'))
    netNorm.border = [256 256] - netNorm.imageSize(1:2);
    net = replace_last_layer(net, [1 2], [1 2], nClasses, opts.dropOutRatio);
  end
  if strfind(model, 'bnorm')
    net = insert_bnorm_layers(net) ;
  end

  net = dagnn.DagNN.fromSimpleNN(net) ;

end

net = dagnn.DagNN.setLrWd(net);

net.renameVar(net.vars(1).name, 'input');

if ~isnan(opts.dropOutRatio)
  dr_layers = find(arrayfun(@(x) isa(x.block,'dagnn.DropOut'), net.layers)) ;
  if ~isempty(dr_layers)
    if opts.dropOutRatio > 0
      for i=dr_layers, net.layers(i).block.rate = opts.dropOutRatio; end
    else
      net.removeLayer({net.layers(dr_layers).name});
    end
  else
    if opts.dropOutRatio > 0
      pool5_layer = find(arrayfun(@(x) isa(x.block,'dagnn.Pooling'), net.layers)) ;
      conv_layers = pool5_layer(end);
      for i=conv_layers
        block = dagnn.DropOut() ;   block.rate = opts.dropOutRatio ;
        newName = ['drop_' net.layers(i).name];

        net.addLayer(newName, ...
          block, ...
          net.layers(i).outputs, ...
          {newName}) ;

        for l = 1:numel(net.layers)-1
          for f = net.layers(i).outputs
             sel = find(strcmp(f, net.layers(l).inputs )) ;
             if ~isempty(sel)
              [net.layers(l).inputs{sel}] = deal(newName) ;
             end
          end
        end
      end
    end
  end
end

net.layers(~cellfun('isempty', strfind({net.layers(:).name}, 'err'))) = [] ;

opts.train.derOutputs = {} ;
for l=numel(net.layers):-1:1
  if isa(net.layers(l).block, 'dagnn.Loss') && isempty(strfind(net.layers(l).name, 'err'))
      opts.train.derOutputs = {opts.train.derOutputs{:}, net.layers(l).outputs{:}, 1} ;
  end
  if isa(net.layers(l).block, 'dagnn.SoftMax') 
    net.removeLayer(net.layers(l).name)
    l = l - 1;
  end
end

if isempty(opts.train.derOutputs)
  net = dagnn.DagNN.insertLossLayers(net, 'numClasses', nClasses) ;
  fprintf('setting derivative for layer %s \n', net.layers(end).name);
  opts.train.derOutputs = {opts.train.derOutputs{:}, net.layers(end).outputs{:}, 1} ;
end

lossLayers = find(arrayfun(@(x) isa(x.block,'dagnn.Loss') && strcmp(x.block.loss,'softmaxlog'),net.layers));


net.addLayer('top1error', ...
             dagnn.Loss('loss', 'classerror'), ...
             net.layers(lossLayers(end)).inputs, ...
             'top1error') ;

net.addLayer('top5error', ...
             dagnn.Loss('loss', 'topkerror', 'opts', {'topK', 5}), ...
             net.layers(lossLayers(end)).inputs, ...
             'top5error') ;
           
net.print() ;   
net.rebuild() ;

net.meta.normalization.rgbVariance = [];

opts.train.train = find(ismember(imdb.images.set, [1])) ;
opts.train.train = repmat(opts.train.train,1,opts.train.epochFactor);
opts.train.valmode = '250samples';
% opts.train.valmode = '30samples'

opts.train.denseEval = 1;
net.conserveMemory = 1 ;
fn = getBatchWrapper_ucf101_imgs(net.meta.normalization, opts.numFetchThreads, opts.train) ;
[info] = cnn_train_dag(net, imdb, fn, opts.train) ;

end
