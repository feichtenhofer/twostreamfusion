% Before compiling, set 'cudaRootDir' and 'cudnnRootDir' to your cuda and 
% cudnn directories, respectively.
% compile MatConvNet
addpath(fullfile(fileparts(mfilename('fullpath')),'matconvnet','matlab'));
vl_setupnn
if ~ispc
  cudaRootDir = '/usr/local/cuda-7.5';
  cudnnRootDir = '../../../../code/toolboxes/cudnn-7.5-windows10-x64-v5.0-ga';
  cudaMethod = 'mex';
else
  cudaRootDir = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5';
  cudnnRootDir = '..\..\..\..\code\toolboxes\cudnn-7.5-windows10-x64-v5.0-ga';
  cudaMethod = 'nvcc';
end
vl_compilenn('enableGpu', true, ...
               'cudaRoot', cudaRootDir, ...
               'cudaMethod', cudaMethod, ...
               'enableCudnn', true, ...
               'cudnnRoot', cudnnRootDir, ...
               'verbose', 1) ;

% compile MexConv3D
run(fullfile(fileparts(mfilename('fullpath')), ...
         'MexConv3D', 'make_all.m')) ;       
run(fullfile(fileparts(mfilename('fullpath')), ...
         'MexConv3D', 'setup_path.m')) ;   
