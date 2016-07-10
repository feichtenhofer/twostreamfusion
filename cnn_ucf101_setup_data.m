function imdb = cnn_ucf101_setup_data(varargin)

opts.splitDir = 'ucf101_splits';
opts.nSplit = 1;
opts.dataPath = '';
opts.flowDir = '';

[opts, varargin] = vl_argparse(opts, varargin) ;
opts.dataDir = fullfile(opts.dataPath,'ucf101') ;


% -------------------------------------------------------------------------
%                                                  Load categories metadata
% -------------------------------------------------------------------------

[Training_set, Testing_set, cats]= feval(sprintf('get_%s_split','UCF101'),opts.nSplit,opts.splitDir);
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

imdb.imageDir = fullfile(opts.dataDir, 'jpegs_256') ;
imdb.flowDir = opts.flowDir;

imdb.images.name = horzcat(Training_set{:},Testing_set{:}) ;
imdb.images.set = horzcat(ones(1, nTrain), 2*ones(1, nTest)) ;
imdb.images.label = horzcat(labels_train, labels_test) ;
imdb.images.labels = double(repmat(imdb.images.label,101,1)' == repmat(1:101,length(imdb.images.label),1));
imdb.images.nFrames = zeros(1, numel(imdb.images.name)) ;
imdb.images.flowScales = cell(1,numel(imdb.images.name));

% get frames
vids = imdb.images.name;
for v=1:numel(vids)
  vid_name = vids(v); vid_name = vid_name{1}(1:end-4);
  flowsU = dir(fullfile(imdb.flowDir, 'u', vid_name, '*.jpg'));
  imdb.images.nFrames(v) = numel(flowsU);

  if ~isempty(strfind(opts.flowDir, 'scaled'))
    scaleFile = [opts.flowDir, filesep, 'u', filesep, vid_name, '.bin'];
    if ~exist(scaleFile),       scaleFile = [opts.flowDir, filesep, filesep, vid_name, '.bin']; end;
    imdb.images.flowScales{v} =    getFlowScale(scaleFile);
  end
end

function minMaxFlow = getFlowScale(file, frame)

fid = fopen(file,'rb');
minMaxFlow = fread(fid, [4, inf],'single');
fclose(fid);

function [train_fnames,test_fnames, saction]= get_UCF101_split(isplit,splitdir)
  
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



