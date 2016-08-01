function stats = cnn_train_dag(net, imdb, getBatch, varargin)
%CNN_TRAIN_DAG Demonstrates training a CNN using the DagNN wrapper
%    CNN_TRAIN_DAG() is similar to CNN_TRAIN(), but works with
%    the DagNN wrapper instead of the SimpleNN wrapper.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.expDir = fullfile('data','exp') ;
opts.continue = false ;
opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [] ;
opts.prefetch = false ;
opts.numEpochs = 300 ;
opts.learningRate = 0.001 ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.derOutputs = {'objective', 1} ;
opts.extractStatsFn = @extractStats ;

opts.memoryMapFile = fullfile(tempdir, 'ramdisk', 'matconvnet.bin') ;
opts.startEpoch = 1;
opts.epochStep = 1;
opts.resetLRandWD = false;
opts.valmode = '30samples';
opts.temporalStride = 1;
opts.DropOutRatio = NaN;
opts.backpropDepth = [];
opts.numValFrames = 3;
opts.nFramesPerVid = 5;
opts.saveAllPredScores = false;
opts.denseEval = 0;
opts.cudnnWorkspaceLimit = [];
opts.plotDiagnostics = false;
[opts, varargin]  = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------

state.getBatch = getBatch ;
evaluateMode = isempty(opts.train) ;
if ~evaluateMode
  if isempty(opts.derOutputs)
    error('DEROUTPUTS must be specified when training.\n') ;
  end
end
stats = [] ;

% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus > 1
  if isempty(gcp('nocreate')),
    parpool('local',numGpus) ;
    spmd, gpuDevice(opts.gpus(labindex)), end
  end
  if exist(opts.memoryMapFile)
    delete(opts.memoryMapFile) ;
  end
elseif numGpus == 1
  gpuDevice(opts.gpus)
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

start = opts.continue * findLastCheckpoint(opts.expDir) ;
if start >= 1
  fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
  [net, stats] = loadState(modelPath(start)) ;
end

for epoch=start+1:opts.numEpochs
  
  
 % train one epoch
  state.epoch = epoch ;
  state.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
  state.train = opts.train(randperm(numel(opts.train))) ; % shuffle
  state.val = opts.val ;
  state.imdb = imdb ;

  if numGpus <= 1
    s_train = process_epoch(net, state, opts, 'train');
    s_val = process_epoch(net, state, opts, 'val');

    if epoch > 1
      stats.train = softAssignStruct(stats.train, s_train, epoch);
      stats.val = softAssignStruct(stats.val, s_val, epoch);
    else
      stats.train = s_train;
      stats.val = s_val;
    end
  else
    savedNet = net.saveobj() ;
    spmd
      set(0,'RecursionLimit',1000)
      net_ = dagnn.DagNN.loadobj(savedNet) ;
      stats_.train = process_epoch(net_, state, opts, 'train') ;
      stats_.val = process_epoch(net_, state, opts, 'val') ;
      if labindex == 1, savedNet_ = net_.saveobj() ; end
    end
    net = dagnn.DagNN.loadobj(savedNet_{1}) ;
    stats__ = accumulateStats(stats_) ;
    if epoch > 1
      stats.train = softAssignStruct(stats.train, stats__.train, epoch);
      stats.val = softAssignStruct(stats.val, stats__.val, epoch);
    else
      stats.train = stats__.train;
      stats.val = stats__.val;
    end
    clear net_ stats_ stats__ savedNet_ ;
  end
  
  fprintf('finished epoch %02d: val stats:\n',  state.epoch ) ;

 
  % save
  if ~evaluateMode
    saveState(modelPath(epoch), net, stats, opts) ;
  end

  figure(1) ; clf ;
  values = [] ;   values_loss = [] ;
  leg = {} ;   leg_loss = {} ;
  for s = {'train', 'val'}
    s = char(s) ;
    for f = setdiff(fieldnames(stats.(s))', {'num', 'time','scores', 'allScores'})
      f = char(f) ;
      if isempty(strfind(f,'err'))
        leg_loss{end+1} = sprintf('%s (%s)', f, s) ;
        tmp = [stats.(s).(f)] ;         
        values_loss(end+1,:) = tmp(1,:)' ;
      else
        leg{end+1} = sprintf('%s (%s)', f, s) ;
        tmp = [stats.(s).(f)] ;        
        values(end+1,:) = tmp(1,:)' ;
      end
      tmp = [stats.(s).(f)];
      fprintf('%s (%s):%.3f\n', f, s, tmp(end))
    end
  end

  if ~isempty(values_loss)
    subplot(1,2,1) ; plot(1:epoch, values_loss','o-') ; 
    legend(leg_loss{:},'Location', 'northoutside'); xlabel('epoch') ; ylabel('objective') ;
    subplot(1,2,2) ; plot(1:epoch, values','o-') ; ylim([0 1])
    legend(leg{:},'Location', 'northoutside') ; xlabel('epoch') ; ylabel('error') ;
    grid on ;
    drawnow ;
    print(1, modelFigPath, '-dpdf') ;
  end
  
    
end

% -------------------------------------------------------------------------
function stats = process_epoch(net, state, opts, mode)
% -------------------------------------------------------------------------

if strcmp(mode,'train')
  state.momentum = num2cell(zeros(1, numel(net.params))) ;
end

numGpus = numel(opts.gpus) ;
if numGpus >= 1
  net.move('gpu') ;
  if strcmp(mode,'train')
    state.momentum = cellfun(@gpuArray,state.momentum,'UniformOutput',false) ;
  end
end
if numGpus > 1
  mmap = map_gradients(opts.memoryMapFile, net, numGpus) ;
else
  mmap = [] ;
end

stats.time = 0 ;
stats.num = 0 ;
stats.scores = [] ;
stats2.err1 = 0;
stats2.err5 = 0;
net.backpropDepth = opts.backpropDepth;


if ~isempty(opts.cudnnWorkspaceLimit)
  net.cudnnWorkspaceLimit = opts.cudnnWorkspaceLimit;
end

subset = state.(mode) ;
start = tic ;
num = 0 ;


if ~strcmp(mode,'train')
  net.mode = 'test';
  
  dataset =  ceil(state.imdb.images.set(subset(1))/2); 
  nClasses = numel(state.imdb.classes.name);
  if nClasses < 5
    nClasses = numel(state.imdb.classes.name{dataset});
  end
  
  stats2.scores = zeros(nClasses, numel(subset));

  moreopts.frameSample = 'uniformly';
  moreopts.augmentation = 'uniform';

  if strcmp(opts.valmode,'30samples')
    % sample less frames and crops:
    moreopts.numAugments = 6;
    moreopts.nFramesPerVid = 5;
    opts.batchSize =  64*numlabs ;
    opts.numSubBatches =  1;
    moreopts.keepFramesDim = true;
  elseif strcmp(opts.valmode,'centreSamplesFast')
    % sample less frames and crops:
    moreopts.numAugments = 2;
    moreopts.nFramesPerVid = 3;
    opts.batchSize =  32*numlabs ;
    moreopts.keepFramesDim = true;
    opts.numSubBatches = 32;
  elseif strcmp(opts.valmode,'250samples') , 
    moreopts.numAugments = 10;
    moreopts.nFramesPerVid = 25;
    opts.batchSize =  16*numlabs ;
    opts.numSubBatches = 1;
    moreopts.keepFramesDim = true; % make getBatch output 5 dimensional  
  elseif strcmp(opts.valmode,'dense')
    moreopts.augmentation = 'none';
    moreopts.numAugments = 0;
    moreopts.nFramesPerVid = 25;
    moreopts.keepFramesDim = true; 
    opts.batchSize = numlabs;
    opts.numSubBatches = numlabs;
  elseif strcmp(opts.valmode,'temporalStrideRandom') , 
    moreopts.nFrameStack = opts.nFramesPerVid; 
    moreopts.frameSample = 'temporalStride';
    moreopts.temporalStride = ceil(median(opts.temporalStride));
    moreopts.temporalStride = max(opts.temporalStride);
    opts.batchSize =  32*numlabs ;
    opts.numSubBatches = opts.batchSize; % has to be
    moreopts.nFramesPerVid = opts.numValFrames;
    moreopts.keepFramesDim = true; 
  end
  
  if opts.denseEval 
    moreopts.augmentation = 'none';
    moreopts.numAugments = 2;
  end
  
  pred_layers = [];
  for l=1:numel(net.layers)
    if isempty( net.layers(l).params ), continue; end;
    if size(net.params(net.getParamIndex(net.layers(l).params{1})).value,4) == nClasses || ...
        size(net.params(net.getParamIndex(net.layers(l).params{1})).value,5) == nClasses % 3D FC layer
          pred_layers = [pred_layers net.layers(l).outputIndexes];
          net.vars(net.layers(l).outputIndexes).precious = 1;
    end
  end

  if opts.saveAllPredScores
    stats2.allScores = zeros(numel(pred_layers),moreopts.numAugments* moreopts.nFramesPerVid/opts.nFramesPerVid, nClasses,  numel(subset));
  end
else
  net.mode = 'normal';
  moreopts = [];
end

for t=1:opts.batchSize:numel(subset)
  batchSize = min(opts.batchSize, numel(subset) - t + 1) ;

  for s=1:opts.numSubBatches

    batchStart = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
    num = num + numel(batch) ;
    if numel(batch) == 0, continue ; end
    clear inputs;

    [inputs] = state.getBatch(state.imdb, batch, moreopts) ;
    moreopts.frameList = [];
    if strcmp(net.mode, 'test') && strcmp(opts.valmode,'temporalStrideRandom')
      for i = 2:4:numel(inputs)
        sz = size(inputs{i});
        inputs{i} = gather(inputs{i});
        nFramesPerVid = sz(5)/moreopts.numAugments;
        chunks = ceil(nFramesPerVid / opts.nFramesPerVid); 
        inputs{i} = reshape(inputs{i}, sz(1), sz(2), sz(3),   [], opts.nFramesPerVid);
        inputs{i} = permute(inputs{i} , [1 2 3 5 4]);
      end
    end
    net.meta.curNumFrames = repmat(size(inputs{2},4) / numel(inputs{4}),1,numel(net.layers)); % nFrames = instances/labels
        
      
    net.meta.curBatchSize = numel(batch);
    inputs{end+1} = 'inputSet'; inputs{end+1} = ceil(state.imdb.images.set(batch)/2); % dataset   

    if opts.prefetch
      if s == opts.numSubBatches
        batchStartNext = t + (labindex-1) + opts.batchSize ;
        batchEndNext = min(t+2*opts.batchSize-1, numel(subset)) ;
      else
        batchStartNext = batchStart + numlabs ; batchEndNext = batchEnd;
      end
      nextBatch = subset(batchStartNext : opts.numSubBatches * numlabs : batchEndNext) ;
      if ~isempty(nextBatch)
        moreopts.frameList = state.getBatch(state.imdb, nextBatch, moreopts) ;
      else 
        moreopts.frameList = NaN ;
      end
    end
      
    if ndims(inputs{2})>4  % average over frames
      dataset = inputs{end}(1);
      nClasses = numel(state.imdb.classes.name);
      if nClasses < 5
        nClasses = numel(state.imdb.classes.name{dataset});
      end
      frame_predictions = cell(numel(pred_layers),size(inputs{2},5));
      for fr = 1:size(inputs{2},5)
        frame_inputs = inputs; 
        net.meta.curNumFrames = repmat(size(inputs{2},4) / numel(inputs{4}),1,numel(net.layers)); % nFrames = instances/labels

        for i = 2:4:numel(inputs)
          if size(frame_inputs{i},5) > 1
            frame_inputs{i}=frame_inputs{i}(:,:,:,:,fr);
          end
        end

        if strcmp(mode, 'train')
          net.accumulateParamDers = (s ~= 1) ;
          net.eval(frame_inputs, opts.derOutputs) ;
        else
          net.eval(frame_inputs) ;
        end   
        [frame_predictions{:,fr}] = deal(net.vars(pred_layers).value) ;        
      end

      tmp = [];
      for k = 1:numel(pred_layers)
        frame_predictions(k,:)= cellfun(@(x) mean(mean(x,1),2), frame_predictions(k,:), 'UniformOutput', false);
        tmp = [tmp; frame_predictions{k,:}];
      end
      frame_predictions = tmp;

      if  opts.saveAllPredScores
        stats2.allScores(:,:,:,batchStart : opts.numSubBatches * numlabs : batchEnd) = gather(frame_predictions);
      end
      frame_predictions = mean(mean(mean(frame_predictions),1),2);
      if min(net.meta.curNumFrames) > 1
        frame_predictions = mean(frame_predictions,4);
      end
      
      [err1, err5] = error_multiclass(opts, inputs{4}, gather(frame_predictions));
      stats2.err1 = (stats2.err1 + err1);
      stats2.err5 = (stats2.err5 + err5);
  
    else
      if strcmp(mode, 'train')
        net.accumulateParamDers = (s ~= 1) ;
        net.eval(inputs, opts.derOutputs) ;
      else
        net.eval(inputs) ;
      end
    end
    
    if strcmp(mode, 'val') && ndims(inputs{2})>4 
      stats2.scores(:, batchStart : opts.numSubBatches * numlabs : batchEnd) = squeeze(gather(frame_predictions));
    end

  end
  
  % extract learning stats
  stats = opts.extractStatsFn(net) ;

  if ndims(inputs{2})>4  % average over frames
    for f = fieldnames(stats2)'
        f = char(f) ;  stats.(f) = stats2.(f);
    end 
  end

  [net.vars.value] = deal([]) ;
  [net.vars.der] = deal([]) ;
  
  % accumulate gradient
  if strcmp(mode, 'train')
    if ~isempty(mmap)
      write_gradients(mmap, net) ;
      labBarrier() ;
    end
    state = accumulate_gradients(state, net, opts, batchSize, mmap) ;
  end

  % print learning statistics
  time = toc(start) ;
  stats.num = num ;
  stats.time = toc(start) ;

  fprintf('%s: epoch %02d: %3d/%3d: lr: %.0e, %.1f Hz', ...
    mode, ...
    state.epoch, ...
    fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize), ...
    state.learningRate, ...
    stats.num/stats.time * max(numGpus, 1)) ;
  for f = setdiff(fieldnames(stats)',  {'num', 'time','scores', 'allScores'})
    f = char(f) ;
    if ndims(inputs{2})>4  && any(strcmp(f, {'err1', 'err5'}))
      n = (t + batchSize - 1) / max(1,numlabs) ;
      stats.(f) = stats.(f) / n;
    end
      fprintf(' %s:%.3f', f, stats.(f)) ;
  end
  fprintf('\n') ;
  
    % debug info
  if opts.plotDiagnostics && numGpus <= 1
    figure(2) ; net.diagnose('Vars',1,'Params',1,'Time',1) ; drawnow ;
  end
end

net.reset() ;
net.move('cpu') ;

% -------------------------------------------------------------------------
function state = accumulate_gradients(state, net, opts, batchSize, mmap)
% -------------------------------------------------------------------------
for p=1:numel(net.params)
  if isempty( net.params(p).der ), continue; end;

  % bring in gradients from other GPUs if any
  if ~isempty(mmap)
    numGpus = numel(mmap.Data) ;
    tmp = zeros(size(mmap.Data(labindex).(net.params(p).name)), 'single') ;
    for g = setdiff(1:numGpus, labindex)
      tmp = tmp + mmap.Data(g).(net.params(p).name) ;
    end
    net.params(p).der = net.params(p).der + tmp ;
  else
    numGpus = 1 ;
  end

  switch net.params(p).trainMethod

    case 'average' % mainly for batch normalization
      thisLR = net.params(p).learningRate ;
      net.params(p).value = ...
          (1 - thisLR) * net.params(p).value + ...
          (thisLR/batchSize/net.params(p).fanout) * net.params(p).der ;

    case 'gradient'
      thisDecay = opts.weightDecay * net.params(p).weightDecay ;
      thisLR = state.learningRate * net.params(p).learningRate ;
      state.momentum{p} = opts.momentum * state.momentum{p} ...
        - thisDecay * net.params(p).value ...
        - (1 / batchSize) * net.params(p).der ;
      net.params(p).value = net.params(p).value + thisLR * state.momentum{p} ;

    case 'otherwise'
      error('Unknown training method ''%s'' for parameter ''%s''.', ...
        net.params(p).trainMethod, ...
        net.params(p).name) ;
  end
end

% -------------------------------------------------------------------------
function mmap = map_gradients(fname, net, numGpus)
% -------------------------------------------------------------------------
format = {} ;
for i=1:numel(net.params)
  format(end+1,1:3) = {'single', size(net.params(i).value), net.params(i).name} ;
end
format(end+1,1:3) = {'double', [3 1], 'errors'} ;
if ~exist(fname) && (labindex == 1)
  f = fopen(fname,'wb') ;
  for g=1:numGpus
    for i=1:size(format,1)
      fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
    end
  end
  fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, 'Format', format, 'Repeat', numGpus, 'Writable', true) ;

% -------------------------------------------------------------------------
function write_gradients(mmap, net)
% -------------------------------------------------------------------------
for i=1:numel(net.params)
  if isempty( net.params(i).der ), continue; end;
  mmap.Data(labindex).(net.params(i).name) = gather(net.params(i).der) ;
end

% -------------------------------------------------------------------------
function stats = accumulateStats(stats_)
% -------------------------------------------------------------------------

stats = struct() ;

for s = {'train', 'val'}
  s = char(s) ;
  total = 0 ;

  for g = 1:numel(stats_)
    stats__ = stats_{g} ;
    num__ = stats__.(s).num ;
    total = total + num__ ;

    for f = setdiff(fieldnames(stats__.(s))', 'num')
      f = char(f) ;

      if g == 1
        stats.(s).(f) = 0 ;
      end
      stats.(s).(f) = stats.(s).(f) + stats__.(s).(f) * num__ ;

      if g == numel(stats_)
        stats.(s).(f) = stats.(s).(f) / total ;
      end
    end
  end
  stats.(s).num = total ;
end

% -------------------------------------------------------------------------
function stats = extractStats(net)
% -------------------------------------------------------------------------
sel = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block})) ;
stats = struct() ;
for i = 1:numel(sel)
  stats.(net.layers(sel(i)).name) = net.layers(sel(i)).block.average ;
end

% -------------------------------------------------------------------------
function saveState(fileName, net, stats, opts)
% -------------------------------------------------------------------------
net_ = net ;
net = net_.saveobj() ;
save(fileName, 'net', 'stats', 'opts') ;

% -------------------------------------------------------------------------
function [net, stats] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'net', 'stats') ;
if ~exist('stats','var') stats = struct; end;
net = dagnn.DagNN.loadobj(net) ;

function [err1, err5] = error_multiclass(opts, labels, predictions)
% -------------------------------------------------------------------------
[~,predictions] = sort(predictions, 3, 'descend') ;
error = ~bsxfun(@eq, predictions, reshape(labels, 1, 1, 1, [])) ;
err1 = sum(sum(sum(error(:,:,1,:)))) ;
err5 = sum(sum(sum(min(error(:,:,1:5,:),[],3)))) ;

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;

function s1 = softAssignStruct(s1, s2, i)
for f = fieldnames(s2)'
  f = char(f) ;
  s1(i).(f) = s2.(f);
end


