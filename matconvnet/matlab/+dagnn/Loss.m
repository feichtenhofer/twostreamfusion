classdef Loss < dagnn.ElementWise
  properties
    loss = 'softmaxlog'
    opts = {}
  end

  properties (Transient)
    average = 0
    numAveraged = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      sz = size(inputs{1}); if numel(sz) < 4, sz(4) = 1; end 
      nFrames = sz(4) / obj.net.meta.curBatchSize ;
      if any(sz(1:2) > 1) || nFrames > 1
        if strfind(obj.loss, 'error') % average predictions over frames and spatial locations
          f_idx = 1:nFrames:size(inputs{1},4);
          pred_avg = {}; pred_max = {};
          for i = 1:size(f_idx,2)
            pred_avg{i} = mean(mean(mean(inputs{1}(:,:,:,f_idx(i):f_idx(i)+nFrames-1),1),2),4);
          end
          
          inputs{1} = cat(4,pred_avg{:});
        else % replicate labels for all frames
          inputs{2} = repmat(inputs{2}, nFrames, 1);
          inputs{2} = inputs{2}(:);
        end
      end
      outputs{1} = vl_nnloss(inputs{1}, inputs{2}, [], 'loss', obj.loss, obj.opts{:}) ;
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      sz = size(inputs{1}); if numel(sz) < 4, sz(4) = 1; end 
      nFrames = sz(4) / obj.net.meta.curBatchSize ;
      
      if nFrames > 1 
        if strfind(obj.loss, 'error') % average predictions over frames
          f_idx = 1:nFrames:size(inputs{1},4);
          pred_avg = {};
          for i = 1:size(f_idx,2)
            pred_avg{i} = mean(mean(mean(inputs{1}(:,:,:,f_idx(i):f_idx(i)+nFrames-1),1),2),4);
          end
          inputs{1} = cat(4,pred_avg{:});
        else % replicate labels for all frames
          inputs{2} = repmat(inputs{2}, nFrames, 1);
          inputs{2} = inputs{2}(:);
        end
      end
      derInputs{1} = vl_nnloss(inputs{1}, inputs{2}, derOutputs{1}, 'loss', obj.loss, obj.opts{:}) ;
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
      rfs(3,1) = rfs(1,1) ;
    end

    function obj = Loss(varargin)
      obj.load(varargin) ;
    end
  end
end
