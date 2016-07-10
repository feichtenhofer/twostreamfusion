classdef Conv3D < dagnn.Filter
  properties
    size = [0 0 0 0]
    hasBias = true
    opts = {'cuDNN'}
  end
  methods
    function outputs = forward(obj, inputs, params)
      if ~obj.hasBias, params{2} = [] ; end
      
      sz = size(inputs{1}); if numel(sz) < 4, sz(4) = 1; end 
      
      nFrames = sz(4) / obj.net.meta.curBatchSize ;
      
      inputs{1} = reshape(inputs{1}, sz(1), sz(2), sz(3),  nFrames, sz(4) / nFrames ) ;
      inputs{1} = permute(inputs{1}, [1 2 4 3 5]);
      sz_in = size(inputs{1});
      
      outputs{1} = mex_conv3d(...
        inputs{1}, params{1}, params{2}, ...
        'pad', obj.pad, ...
        'stride', obj.stride) ;

      outputs{1} = permute(outputs{1}, [1 2 4 3 5]);
      sz_out = size(outputs{1});
      if numel(sz_out) == 4, sz_out(5) = 1; end % fixes the case of single batch 
      nFrames = sz_out(4); % reset nframes
      outputs{1} = reshape(outputs{1}, sz_out(1), sz_out(2), sz_out(3),  nFrames*sz_out(5) ) ;
      
    end
    
    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if ~obj.hasBias, params{2} = [] ; end
      sz_in = size(inputs{1}); sz_out = size(derOutputs{1});
      if numel(sz_in) == 3, sz_in(4) = 1; end  
      if numel(sz_out) == 3, sz_out(4) = 1; end
      nFramesIn = sz_in(4) / obj.net.meta.curBatchSize ;
      nFramesOut = sz_out(4) / obj.net.meta.curBatchSize ;

      inputs{1} = reshape(inputs{1}, sz_in(1), sz_in(2), sz_in(3),  nFramesIn, sz_in(4) / nFramesIn ) ;
      inputs{1} = permute(inputs{1}, [1 2 4 3 5]);
      derOutputs{1} = reshape(derOutputs{1}, sz_out(1), sz_out(2), sz_out(3),  nFramesOut, sz_out(4) / nFramesOut ) ;
      derOutputs{1} = permute(derOutputs{1}, [1 2 4 3 5]);
      [derInputs{1}, derParams{1}, derParams{2}] = mex_conv3d(...
        inputs{1}, params{1}, params{2}, derOutputs{1}, ...
        'pad', obj.pad, ...
        'stride', obj.stride) ;

      derInputs{1} = permute(derInputs{1}, [1 2 4 3 5]);
      sz_out = size(derInputs{1});
      if numel(sz_out) == 4, sz_out(5) = 1; end % fixes the case of single batch 
      nFrames = sz_out(4); % reset nframes
      
      derInputs{1} = reshape(derInputs{1}, sz_out(1), sz_out(2), sz_out(3),  nFrames*sz_out(5) ) ;

    end
    
    function kernelSize = getKernelSize(obj)
      kernelSize = obj.size(1:2) ;
    end
    
    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = obj.size(4) ;
    end
    
    function params = initParams(obj)
      sc = sqrt(2 / prod(obj.size(1:3))) ;
      params{1} = randn(obj.size,'single') * sc ;
      params{2} = zeros(obj.size(4),1,'single') * sc ;
    end
    
    function obj = Conv(varargin)
      obj.load(varargin) ;
    end
    
    function rfs = getReceptiveFields(obj)
      ks = obj.getKernelSize() ;
      y1 = 1 - obj.pad(1) ;
      y2 = 1 - obj.pad(1) + ks(1) - 1 ;
      x1 = 1 - obj.pad(2) ;
      x2 = 1 - obj.pad(2) + ks(2) - 1 ;
      h = y2 - y1 + 1 ;
      w = x2 - x1 + 1 ;
      rfs.size = [h, w] ;
      rfs.stride = obj.stride(1:2) ;
      rfs.offset = [y1+y2, x1+x2]/2 ;
    end
    
  end
end
