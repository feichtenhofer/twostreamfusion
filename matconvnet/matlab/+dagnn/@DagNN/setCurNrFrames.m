function setCurNrFrames(obj, name, nFrames)
%setCurNrFrames  Modify the frame size of the folling layers

f = find(strcmp(name, {obj.layers.name})) ;

obj.meta.curNumFrames(f) = nFrames;

out = obj.layers(f).outputIndexes ;
if (obj.vars(out).fanout == 0)
  return; 
end
for layer = obj.layers
  if any(layer.inputIndexes == out)
    setCurNrFrames(obj, layer.name, nFrames);
  end
end
