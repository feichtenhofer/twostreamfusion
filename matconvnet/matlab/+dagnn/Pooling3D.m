classdef Pooling3D < dagnn.Filter
  properties
    method = 'max'
    poolSize = [1 1 1]
  end

  methods
    function outputs = forward(obj, inputs, params)
      
      sz = size(inputs{1}); if numel(sz) < 4, sz(4) = 1; end 
      nFrames = sz(4) / obj.net.meta.curBatchSize ;

      inputs{1} = reshape(inputs{1}, sz(1), sz(2), sz(3),  nFrames, sz(4) / nFrames ) ;
      inputs{1} = permute(inputs{1}, [1 2 4 3 5]);
      sz_in = size(inputs{1});

      [outputs{1}, ~]  = mex_maxpool3d(inputs{1},...
        'pool',obj.poolSize, 'stride', obj.stride, 'pad', obj.pad);

      outputs{1} = permute(outputs{1}, [1 2 4 3 5]);
      sz_out = size(outputs{1});
      if numel(sz_out) < 4, sz_out(4) = 1; end % fixes the case of time being pooled 
	    nFrames = sz_out(4); % reset nframes
      obj.net.meta.curNumFrames(obj.layerIndex) = nFrames ;
      if numel(sz_out) < 5, sz_out(5) = 1; end % fixes the case of single batch 

      outputs{1} = reshape(outputs{1}, sz_out(1), sz_out(2), sz_out(3),  nFrames*sz_out(5) ) ;     
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      % currently doing forward and backward due to idx
      sz = size(inputs{1}); if numel(sz) < 4, sz(4) = 1; end 
      nFramesIn = sz(4) / obj.net.meta.curBatchSize ;

      inputs{1} = reshape(inputs{1}, sz(1), sz(2), sz(3),  nFramesIn, sz(4) / nFramesIn ) ;
      inputs{1} = permute(inputs{1}, [1 2 4 3 5]);
      sz_in = size(inputs{1});
      [~, idx]  = mex_maxpool3d(inputs{1},...
        'pool',obj.poolSize, 'stride', obj.stride, 'pad', obj.pad);
      
      sz = size(derOutputs{1}); if numel(sz) < 4, sz(4) = 1; end 
      nFrames = sz(4) / obj.net.meta.curBatchSize ;

   
      derOutputs{1} = reshape(derOutputs{1}, sz(1), sz(2), sz(3),  nFrames, sz(4) / nFrames ) ;
      derOutputs{1} = permute(derOutputs{1}, [1 2 4 3 5]);
     
 
      derInputs{1} = mex_maxpool3d(derOutputs{1}, idx, sz_in ,...
        'pool',obj.poolSize, 'stride', obj.stride, 'pad', obj.pad);
      
      derInputs{1} = permute(derInputs{1}, [1 2 4 3 5]);
      sz_out = size(derInputs{1});
      if numel(sz_out) < 4, sz_out(4) = 1; end % fixes the case of time being pooled 
      nFrames = sz_out(4); % reset nframes
      obj.net.meta.curNumFrames(obj.layerIndex) = nFrames; % reset nframes

      if numel(sz_out) < 5 , sz_out(5) = 1; end % fixes the case of single batch 
      derInputs{1} = reshape(derInputs{1}, sz_out(1), sz_out(2), sz_out(3),   nFrames*sz_out(5) ) ;      
      derParams = {};
    
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.poolSize ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = inputSizes{1}(3) ;
    end

    function obj = Pooling(varargin)
      obj.load(varargin) ;
    end
  end
end
