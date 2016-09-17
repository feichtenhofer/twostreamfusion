function imdb = cnn_ucf101_setup_data(varargin)

opts.nSplit = 1;
opts.dataPath = '';
opts.flowDir = '';
opts.dataSet = 'ucf101' ;
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.dataDir = fullfile(opts.dataPath,opts.dataSet) ;

opts.imageDir = fullfile(opts.dataDir, 'jpegs_256') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

switch opts.dataSet
  case 'ucf101'
    nClasses = 101 ;
  case 'hmdb51'
    nClasses = 51 ;
    opts.flowDir = strrep(opts.flowDir, 'ucf101','hmdb51');
    opts.imageDir = strrep(opts.imageDir, 'ucf101','hmdb51');
end

opts.splitDir = [opts.dataSet '_splits']; 

% -------------------------------------------------------------------------
%                                                  Load categories metadata
% -------------------------------------------------------------------------

[Training_set, Testing_set, cats]= feval(sprintf('get_%s_split', lower(opts.dataSet)),opts.nSplit,opts.splitDir);
naction = length(Training_set);

nTrain = 0;       nTest = 0;
for iaction = 1:naction
  nTrain = nTrain + numel(Training_set{iaction});
  nTest = nTest + numel(Testing_set{iaction});
  labels_train{iaction} = repmat(iaction,1,numel(Training_set{iaction}));
  labels_test{iaction}  = repmat(iaction,1,numel(Testing_set{iaction}));
end

labels_train = [labels_train{:}];
labels_test = [labels_test{:}];

imdb.classes.name = cats ;
imdb.videoDir = fullfile(opts.dataDir, 'avis') ;

imdb.imageDir = opts.imageDir;
imdb.flowDir = opts.flowDir;

imdb.images.name = horzcat(Training_set{:},Testing_set{:}) ;
imdb.images.set = horzcat(ones(1, nTrain), 2*ones(1, nTest)) ;
imdb.images.label = horzcat(labels_train, labels_test) ;
imdb.images.labels = double(repmat(imdb.images.label,nClasses,1)' == repmat(1:nClasses,length(imdb.images.label),1));
imdb.images.nFrames = zeros(1, numel(imdb.images.name)) ;
imdb.images.flowScales = cell(1,numel(imdb.images.name));

% get frames
vids = imdb.images.name;

for v=1:numel(vids)
  vid_name = vids(v); vid_name = vid_name{1}(1:end-4);
  ims = dir(fullfile(imdb.imageDir,  vid_name, '*.jpg'));
  imdb.images.nFrames(v) = numel(ims); 
end


function [train_fnames,test_fnames, saction]= get_ucf101_split(isplit,splitdir)
  
  fid = fopen([splitdir '/classInd.txt']);

  classes =      textscan(fid, '%s');
  saction = classes{1}(2:2:end);
  fname = sprintf('%s/testlist0%d.txt',splitdir,isplit);
  fid = fopen(fname);

  test_fnames = cell(length(saction),1);
  train_fnames = cell(length(saction),1); 

  while 1
   tline = fgetl(fid);
   if tline==-1
     break
   end
    [tline, u] = strtok(tline,' ');   

    video = sprintf('%s.avi',tline(1:end-4));
    [className, vidName] = strtok(video,'/');
    iaction = find(strcmp(saction, className));

    test_fnames{iaction}{end+1} = vidName(2:end);
  end

  fclose(fid);
     itr = 1;
  fname = sprintf('%s/trainlist0%d.txt',splitdir,isplit);
  fid = fopen(fname);
  while 1
   tline = fgetl(fid);
   if tline==-1
     break
   end
    [tline, u] = strtok(tline,' ');   

    video = sprintf('%s.avi',tline(1:end-4));
    [className, vidName] = strtok(video,'/');
    iaction = find(strcmp(saction, className));

    train_fnames{iaction}{end+1} = vidName(2:end);
  end
  fclose(fid);


function [train_fnames,test_fnames, saction]= get_hmdb51_split(isplit,splitdir)
  saction =      {'brush_hair','cartwheel','catch','chew','clap','climb','climb_stairs',...
      'dive','draw_sword','dribble','drink','eat','fall_floor','fencing',...
      'flic_flac','golf','handstand','hit','hug','jump','kick_ball',...
      'kick','kiss','laugh','pick','pour','pullup','punch',...
      'push','pushup','ride_bike','ride_horse','run','shake_hands','shoot_ball',...
      'shoot_bow','shoot_gun','sit','situp','smile','smoke','somersault',...
      'stand','swing_baseball','sword_exercise','sword','talk','throw','turn',...
      'walk','wave'};


       for iaction = 1:length(saction)
	 itr = 1;
	 ite = 1;
	 fname = sprintf('%s/%s_test_split%d.txt',splitdir,saction{iaction},isplit);

	 fid = fopen(fname);
	 
	 while 1
	   tline = fgetl(fid);
	   if tline==-1
	     break
	   end
	   [tline, u] = strtok(tline,' ');   
	   u = str2num(u);
	   
	   video = sprintf('%s.avi',tline(1:end-4));
    
	   if u==1 % ignore testing
	       train_fnames{iaction}{itr} = tline;
	       itr = itr + 1;
	   elseif u==2
	       test_fnames{iaction}{ite} = tline;
	       ite = ite + 1;
	   end
	 end
	 fclose(fid);
       end
       
