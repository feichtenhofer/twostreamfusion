classdef Concat < dagnn.ElementWise
  properties
    dim = 3
  end

  properties (Transient)
    inputSizes = {}
  end

  methods
    function outputs = forward(obj, inputs, params)
     
      szw = cellfun(@(x) size(x,2),inputs);
      szh = cellfun(@(x) size(x,1),inputs);
      
      max_w = max(szw); max_h = max(szh);
      if numel(unique(szw)) > 1 || numel(unique(szh)) > 1
        for l=1:numel(inputs)
          pad = [max_h max_w] - [szh(l) szw(l)];
          inputs{l} = padarray(inputs{l}, pad, 'post' );
        end
      end

      outputs{1} = vl_nnconcat(inputs, obj.dim) ;
      obj.inputSizes = cellfun(@size, inputs, 'UniformOutput', false) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = vl_nnconcat(inputs, obj.dim, derOutputs{1}, 'inputSizes', obj.inputSizes) ;
      realInputSizes = cellfun(@size, inputs, 'UniformOutput', false);
      for l=1:numel(derInputs)
            if size(realInputSizes{l}) < 4, 
              realInputSizes{l}(4) = 1; 
            end;
          s.type = '()' ;
          s.subs = {1:realInputSizes{l}(1), 1:realInputSizes{l}(2), 1:realInputSizes{l}(3), 1:realInputSizes{l}(4)} ;
          derInputs{l} = subsref(derInputs{l},s) ;
      end     
      derParams = {} ;
    end

    function reset(obj)
      obj.inputSizes = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      sz = inputSizes{1} ;
      for k = 2:numel(inputSizes)
        sz(obj.dim) = sz(obj.dim) + inputSizes{k}(obj.dim) ;
      end
      outputSizes{1} = sz ;
    end

    function rfs = getReceptiveFields(obj)
      numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
      if obj.dim == 3 || obj.dim == 4
        rfs = getReceptiveFields@dagnn.ElementWise(obj) ;
        rfs = repmat(rfs, numInputs, 1) ;
      else
        for i = 1:numInputs
          rfs(i,1).size = [NaN NaN] ;
          rfs(i,1).stride = [NaN NaN] ;
          rfs(i,1).offset = [NaN NaN] ;
        end
      end
    end

    function load(obj, varargin)
      s = dagnn.Layer.argsToStruct(varargin{:}) ;
      % backward file compatibility
      if isfield(s, 'numInputs'), s = rmfield(s, 'numInputs') ; end
      load@dagnn.Layer(obj, s) ;
    end

    function obj = Concat(varargin)
      obj.load(varargin{:}) ;
    end
  end
end
