function cnn_ucf101_temporal(varargin)

if ~isempty(gcp('nocreate')),
    delete(gcp)
end

opts = cnn_setup_environment();
opts.train.gpus =  1 ;
% opts.train.gpus = [ 1 : 3 ]

opts.dataSet = 'ucf101'; 
% opts.dataSet = 'hmdb51'; 

addpath('network_surgery');
opts.dataDir = fullfile(opts.dataPath, opts.dataSet) ;
opts.splitDir = [opts.dataSet '_splits']; 
opts.nSplit = 1 ;
opts.inputdim  = [ 224,  224, 20] ;
opts.train.memoryMapFile = fullfile(tempdir, 'ramdisk', ['matconvnet' num2str(feature('getpid')) '.bin']) ;
removeInputPadding = 0 ;
opts.train.cheapResize = 0;
opts.train.batchSize = 128  ;
opts.train.numSubBatches = 4 ;
opts.dropOutRatio = .8; % inserted after fully connected layers

opts.train.epochFactor = 100 ;
opts.train.learningRate = [ 1e-2*ones(1,1) 1e-3*ones(1, 1) 1e-4*ones(1,1) 1e-5*ones(1,1)]  ;

opts.train.augmentation = 'randCropFlipStretch';
opts.train.augmentation = 'randCropFlip';
opts.train.augmentation = 'borders5';
opts.train.augmentation = 'corners';
opts.train.augmentation = 'multiScaleRegular';

model = ['res50' opts.train.augmentation '-bs=' num2str(opts.train.batchSize) ...
  '-cheapRsz=' num2str(opts.train.cheapResize), ...
  '-split' num2str(opts.nSplit) '-dr' num2str(opts.dropOutRatio)];

if  strfind(model, 'vgg16');
  baseModel = 'imagenet-vgg-verydeep-16.mat' ; 
  opts.train.epochFactor = 100 ;
  opts.train.learningRate =  [ 5e-4*ones(1, 10) 5e-5*ones(1,5) 5e-6*ones(1,1)]  ;
  opts.train.batchSize = 128  ;
  opts.train.numSubBatches =  ceil(8 / max(numel(opts.train.gpus),1));
elseif strfind(model, 'vgg-m');
  baseModel = 'imagenet-vgg-m-2048.mat' ;
elseif strfind(model, 'res152');
  baseModel = 'imagenet-resnet-152-dag.mat' ;
elseif strfind(model, 'res101');
  baseModel = 'imagenet-resnet-101-dag.mat' ;
elseif strfind(model, 'res50');
  baseModel = 'imagenet-resnet-50-dag.mat' ;
else
  error('Unknown model %s', model) ; 
end
opts.model = fullfile(opts.modelPath,baseModel) ;

opts.expDir = fullfile(opts.dataDir, [opts.dataSet '-' model]) ;
opts.train.plotDiagnostics = 0 ;
opts.train.continue = 1 ;
opts.train.prefetch = 1 ;

opts.imdbPath = fullfile(opts.dataDir, [opts.dataSet '_split' num2str(opts.nSplit) 'imdb.mat']);
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
  imdb.flowDir = opts.flowDir;
else
  imdb = cnn_ucf101_setup_data(opts) ;
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
  if net.meta.normalization.imageSize(3) == 3
      net.meta.normalization.imageSize(3) = 20 ;
      diff =  net.meta.normalization.imageSize(3) - size(net.params(1).value,3);
      net.params(1).value = padarray(net.params(1).value, [0 0 diff 0], 'symmetric', 'post');
  end
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
else
  if isfield(net, 'meta'),
    netNorm = net.meta.normalization;
  else
    netNorm = net.normalization;
  end
  if(netNorm.imageSize(3) == 3)
    if strfind(opts.model,'vgg-m-2048'), 
      net.layers(7) = [] ; %remove norm2 layer as in Simonyan et al NIPS'14
    end
    opts.inputdim  = [netNorm.imageSize(1:2), 20] ;
    net.layers{1}.weights{1} = repmat(mean(net.layers{1}.weights{1},3), [1 1 opts.inputdim(3) 1]) ;
    net.meta.normalization.averageImage = [];
    net.meta.normalization.border = [256 256] - netNorm.imageSize(1:2);
    net = replace_last_layer(net, [1 2], [1 2], nClasses, opts.dropOutRatio);
    net.normalization.imageSize = opts.inputdim ;
  end
  if strfind(model, 'bnorm')
    net = insert_bnorm_layers(net) ;
  end
  net = dagnn.DagNN.fromSimpleNN(net) ;
end

if removeInputPadding
    padMN = [sum(net.layers(1).block.pad(1:2)) sum(net.layers(1).block.pad(3:4))];
    net.layers(1).block.pad  = zeros(1,numel(net.layers(1).block.size));
    net.meta.normalization.imageSize(1:2) = net.meta.normalization.imageSize(1:2) + padMN ;
    net.meta.normalization.averageImage = padarray(net.meta.normalization.averageImage, padMN / 2, 'symmetric');
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


net.meta.normalization.averageImage = ones(1,1,20)*128;   
if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end

net.meta.normalization.rgbVariance = [];

opts.train.train = find(ismember(imdb.images.set, [1  ])) ;
opts.train.train = repmat(opts.train.train,1,opts.train.epochFactor);
opts.train.val = find(ismember(imdb.images.set, [2]) );

opts.train.valmode = '250samples' ;
% opts.train.valmode = '30samples' ;
opts.train.denseEval = 1 ;
net.conserveMemory = 1 ;
fn = getBatchWrapper_ucf101_flow(net.meta.normalization, opts.numFetchThreads, opts.train) ;

[info] = cnn_train_dag(net, imdb, fn, opts.train) ;

end
