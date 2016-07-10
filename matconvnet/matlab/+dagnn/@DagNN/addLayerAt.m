function addLayerAt(obj, f, name, block, inputs, outputs, params)
% OBJ.ADDLAYER(NAME, LAYER, INPUTS, OUTPUTS, PARAMS) adds the
% specified layer to the network. NAME is a string with the layer
% name, used as a unique indentifier. BLOCK is the object implementing
% the layer, which should be a subclass of the Layer. INPUTS, OUTPUTS
% are cell arrays of variable names, and PARAMS of parameter names.

f2 = find(strcmp(name, {obj.layers.name})) ;
if ~isempty(f2), error('There is already a layer with name ''%s''', name),
  name = [name 'x'];
  fprintf(', renaming to ''%s''.\n', name);
end

if nargin < 7, params = {} ; end
if isstr(inputs), inputs = {inputs} ; end
if isstr(outputs), outputs = {outputs} ; end
if isstr(params), params = {params} ; end



layer = struct(...
  'name', {name}, ...
  'inputs', {inputs}, ...
  'outputs', {outputs}, ...
  'params', {params}, ...
  'inputIndexes', {[]}, ...
  'outputIndexes', {[]}, ...
  'paramIndexes', {[]}, ...
  'forwardTime', {[]}, ...
  'backwardTime', {[]}, ...
  'block', {block}) ;


obj.layers = [obj.layers(1:f) layer obj.layers(f+1:end)];

block.net = obj ;
block.layerIndex = f ;

for input = inputs
  obj.addVar(char(input)) ;
end

for output = outputs
  obj.addVar(char(output)) ;
end

for param = params
 obj.addParam(char(param)) ;
end

obj.rebuild() ;
