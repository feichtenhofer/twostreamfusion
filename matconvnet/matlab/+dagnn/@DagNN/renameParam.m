function renameParam(obj, oldName, newName)
%RENAMEVAR Rename a variable
%   RENAMEVAR(OLDNAME, NEWNAME) changes the name of the parameter
%   OLDNAME into NEWNAME. NEWNAME should not be the name of an
%   existing parameter.

if any(strcmp(newName, {obj.params.name}))
  error('%s is the name of an existing variable', newName) ;
end

v = obj.getParamIndex(oldName) ;
if isnan(v)
  error('%s is not an existing variable', oldName) ;
end

for l = 1:numel(obj.layers)    
    sel = find(strcmp(obj.layers(l).params, oldName)) ;
    if any(sel), obj.layers(l).params{sel} = newName ; end;
end
obj.params(v).name = newName ;

% update variable name hash otherwise rebuild() won't find this par corectly
obj.paramNames.(newName) = v ;

obj.rebuild() ;
