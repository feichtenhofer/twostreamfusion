function [ opts ] = cnn_setup_environment( varargin )

  run(fullfile(fileparts(mfilename('fullpath')), ...
         'matconvnet','matlab', 'vl_setupnn.m')) ;

  opts.dataPath = '/data';
  opts.modelPath ='/models';
  opts.flowDir = 'data/ucf101/tvl1_flow';
  opts.imageDir  = '/data/ucf101/jpegs_256';
  opts = vl_argparse(opts, varargin);

  
end

