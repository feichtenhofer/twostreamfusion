function imo = cnn_ucf101_get_frame_batch(images, varargin)

opts.nFramesPerVid = 1;
opts.numAugments = 1;
opts.frameSample = 'uniformly';
opts.uniformAugments = false;
opts.imageDir = '';
opts.temporalStride = 0;

opts.imageSize = [227, 227] ;
opts.border = [29, 29] ;
opts.averageImage = [] ;
opts.rgbVariance = [] ;

opts.augmentation = 'croponly' ;
opts.interpolation = 'bilinear' ;
opts.numAugments = 1 ;
opts.numThreads = 0 ;
opts.prefetch = false ;
opts.keepAspect = true;
opts.cheapResize = 0;
opts.frameList = NaN;
opts.nFrames = [];
opts.imReadSz = [];
opts.stretchAspect = 4/3 ;
opts.stretchScale = 1.2 ;
[opts, varargin] = vl_argparse(opts, varargin);

imgDir = opts.imageDir;

prefetch = opts.prefetch & isempty(opts.frameList);

switch opts.augmentation
  case 'randCropFlip'
  case 'croponly'
    tfs = [.5 ; .5 ; 0 ];
  case 'f5'
    tfs = [...
      .5 0 0 1 1 .5 0 0 1 1 ;
      .5 0 1 0 1 .5 0 1 0 1 ;
       0 0 0 0 0  1 1 1 1 1] ;
  case 'f25'
    [tx,ty] = meshgrid(linspace(0,1,50)) ;
    tfs = [tx(:)' ; ty(:)' ; zeros(1,numel(tx))] ;
    tfs_ = tfs ;
    tfs_(3,:) = 1 ;
    tfs = [tfs,tfs_] ;
  case 'borders25'
    [tx1,ty1] = meshgrid(linspace(.75,1,20)) ;
    [tx2,ty2] = meshgrid(linspace(0,.25,20)) ;
    tx = [tx1 tx2];     ty = [ty1 ty2];
    tfs = [tx(:)' ; ty(:)' ; zeros(1,numel(tx))] ;
    tfs_ = tfs ;
    tfs_(3,:) = 1 ;
    tfs = [tfs,tfs_] ;  
  case 'borders'
    opts.imReadSz(1) = 224 + (342-224).*rand();
    opts.imReadSz(2) = 300 + (480-300).*rand();  
    opts.cheapResize = true;

  otherwise
    opts.imReadResize = 0; 
      
end

if iscell(opts.frameList)
  if isempty(opts.imReadSz)
    im = vl_imreadjpeg(opts.frameList{1}, 'numThreads', opts.numThreads ) ; 
  else
    im = vl_imreadjpeg(opts.frameList{1}, 'numThreads', opts.numThreads, 'resize', opts.imReadSz ) ; 
  end
  sampled_frame_nr = opts.frameList{2};
else
  sampled_frame_nr = cell(numel(images),1);
  frames = cell(numel(images), opts.nFramesPerVid);
  for i=1:numel(images)
    vid_name = images{i};
    nFrames = opts.nFrames(i);
    if  strcmp(opts.frameSample, 'uniformly')
      sampleRate = max(floor((nFrames)/opts.nFramesPerVid),1);
      frameSamples =1:sampleRate:nFrames ;
    elseif strcmp(opts.frameSample, 'temporalStride')
      shift = floor(mod(nFrames,opts.temporalStride)/2); % shift to centre of long videos
      frameSamples = 1:opts.temporalStride:nFrames;
    elseif strcmp(opts.frameSample, 'random')
      frameSamples = randperm(nFrames);
    elseif strcmp(opts.frameSample, 'temporalStrideRandom')
      sampleStart = randi(opts.temporalStride) ; 
      frameSamples = sampleStart:opts.temporalStride:nFrames;
      if isempty(frameSamples)
        frameSamples = 1:opts.temporalStride:nFrames;
      end
    end   
    if length(frameSamples) < opts.nFramesPerVid,
        diff =  opts.nFramesPerVid - length(frameSamples);
        frameSamples = padarray(frameSamples,[0 diff],'symmetric','post');
    elseif length(frameSamples) > opts.nFramesPerVid,
      s = randi(length(frameSamples)-opts.nFramesPerVid);
      frameSamples = frameSamples(s:s+opts.nFramesPerVid-1);
    end

    for k = 1:opts.nFramesPerVid
      frames{i,k} = fullfile(vid_name, ['frame' sprintf('%06d.jpg', frameSamples(k))]) ;
    end 

    if iscell(opts.imageDir)
        imgDir = opts.imageDir{i};
    end
    
    frames(i,1:opts.nFramesPerVid) = strcat([imgDir filesep], frames(i,1:opts.nFramesPerVid));
    sampled_frame_nr{i} = frameSamples;

  end

  if opts.numThreads > 0
    if prefetch
      if isempty(opts.imReadSz)
        vl_imreadjpeg(frames, 'numThreads', opts.numThreads, 'prefetch' ) ;
      else
        vl_imreadjpeg(frames, 'numThreads', opts.numThreads, 'prefetch', 'resize', opts.imReadSz ) ;
      end
      imo = {frames sampled_frame_nr}  ;
      return ;
    end   
    if isempty(opts.imReadSz)
      im = vl_imreadjpeg(frames, 'numThreads', opts.numThreads) ;
    else
      im = vl_imreadjpeg(frames, 'numThreads', opts.numThreads, 'resize', opts.imReadSz ) ;
    end
  end
end

%% no augm
if strcmp(opts.augmentation, 'none')
          
  szw = cellfun(@(x) size(x,2),im);
  szh = cellfun(@(x) size(x,1),im);

  h_min = min(szh(:));
  w_min =  min(szw(:));
  sz = [h_min w_min] ;

  sz = max(opts.imageSize(1:2), sz);
  sz = min(2*opts.imageSize(1:2), sz);
  scal = ([h_min w_min] ./ sz);
    
  imo = ( zeros(sz(1), sz(2), opts.imageSize(3), ...
            numel(images), 2 * opts.nFramesPerVid, 'single') );
  
  for i=1:numel(images)
    si = 1 ;
    for k = 1:opts.nFramesPerVid
      if numel(unique(szw)) > 1 || numel(unique(szh)) > 1
          im{i,k} = im{i,k}(1:h_min,1:w_min,:);
      end   
      imt = cat(3, im{i,k}) ;
      if any(scal ~= 1)
        imt = imresize(imt, sz );
      end
      imo(:, :, :, i, si) = imt;
      imo(:, :, :, i, si+1) = imt(:, end:-1:1,:);      
      si = si + 2 ;
    end
  end  
  
  if ~isempty(opts.averageImage)
    opts.averageImage = mean(mean(opts.averageImage,1),2) ;   
    imo = bsxfun(@minus, imo,opts.averageImage) ;
  end
  return;
end

%% augment now
if exist('tfs', 'var')
  [~,transformations] = sort(rand(size(tfs,2), numel(images)*opts.nFramesPerVid), 1) ;
end


imo = ( zeros(opts.imageSize(1), opts.imageSize(2), opts.imageSize(3), ...
            numel(images), opts.numAugments * opts.nFramesPerVid, 'single') ) ;


szw = cellfun(@(x) size(x,2),im);
szh = cellfun(@(x) size(x,1),im);  
  
for i=1:numel(images)
  si = 1 ;

  
  h_min = min(szh(:));
  w_min =  min(szw(:));

  if  strcmp( opts.augmentation, 'borders25')
      sz = round(min(opts.imageSize(1:2)' .* (.75+0.5*rand(2,1)), [h_min; w_min])) ; % 0.75 +- 0.5, not keep aspect  
  elseif strcmp( opts.augmentation, 'stretch')
      aspect = exp((2*rand-1) * log(opts.stretchAspect)) ;
      scale = exp((2*rand-1) * log(opts.stretchScale)) ;
      tw = opts.imageSize(2) * sqrt(aspect) * scale ;
      th = opts.imageSize(1) / sqrt(aspect) * scale ;
      reduce = min([w_min / tw, h_min / th, 1]) ;
      sz = round(reduce * [th ; tw]) ;
  else
      sz(1) = round(160 + (256-160).*rand());
      sz(2) = round(160 + (256-160).*rand());  
  end

  for k = 1:opts.nFramesPerVid
    imt = im{i,k} ;
    w = size(imt,2) ;
    h = size(imt,1) ;
    if ~strcmp(opts.augmentation, 'uniform')
      if ~isempty(opts.rgbVariance) 
        opts.averageImage = bsxfun(@plus, opts.averageImage, reshape(opts.rgbVariance * randn(opts.imageSize(3),1), 1,1,opts.imageSize(3))) ;
      end
      for ai = 1:opts.numAugments
        if k == 1
          switch opts.augmentation
            case 'stretch'
              dx = randi(w - sz(2) + 1 ) ;
              dy = randi(h - sz(1) + 1 ) ;
              flip = rand > 0.5 ;
            case 'multiScaleRegular'
              dy = [0 h-sz(1) 0 h-sz(1)  floor((h-sz(1)+1)/2)] + 1; 
              dx = [0 w-sz(2) w-sz(2) 0 floor((w-sz(2)+1)/2)] + 1;
              corner = randi(5);
              dx = dx(corner); dy = dy(corner); 
              flip = rand > 0.5 ;          
            case 'borders25'
              tf = tfs(:, transformations(mod(ai-1, numel(transformations)) + 1)) ;
              dx = floor((w - sz(2)) * tf(2)) + 1 ;
              dy = floor((h - sz(1)) * tf(1)) + 1 ;
              flip = tf(3) ;   
            case 'borders'
              randx = rand > 0.5 ;
              if randx
                top = rand > 0.5 ;
                dx = randi(w - sz(2) + 1 ) ;
                dy = floor((h - sz(1)) * top) + 1 ;
              else
                right = rand > 0.5 ;
                dx = floor((w - sz(2)) * right) + 1 ;
                dy = randi(h - sz(1) + 1 ) ;
              end
              flip = rand > 0.5 ;
            otherwise
              tf = tfs(:, transformations(mod(ai-1, numel(transformations)) + 1)) ;
              dx = floor((w - sz(2)) * tf(2)) + 1 ;
              dy = floor((h - sz(1)) * tf(1)) + 1 ;
              flip = tf(3) ;          
          end
        end
        if opts.cheapResize
          sx = round(linspace(dx, sz(2)+dx-1, opts.imageSize(2))) ;
          sy = round(linspace(dy, sz(1)+dy-1, opts.imageSize(1))) ;
        else
          factor = [opts.imageSize(1)/sz(1) ...
              opts.imageSize(2)/sz(2)];
          
                   
          if any(abs(factor - 1) > 0.0001)
            imt =   imresize(gather(imt(dy:sz(1)+dy-1,dx:sz(2)+dx-1,:)), opts.imageSize(1:2));
          end
                   
          sx = 1:opts.imageSize(2); sy = 1:opts.imageSize(1);
        end

        if flip
          sx = fliplr(sx) ;
          imo(:,:,:,i,si) = imt(sy,sx,:) ;
        else
          imo(:,:,:,i,si) = imt(sy,sx,:) ;
        end

        si = si + 1 ;
      end
    else
      % oversample (4 corners, center, and their x-axis flips)
      indices_y = [0 h-opts.imageSize(1)] + 1;
      indices_x = [0 w-opts.imageSize(2)] + 1;
      center_y = floor(indices_y(2) / 2)+1;
      center_x = floor(indices_x(2) / 2)+1;

      if opts.numAugments == 6,  indices_y = center_y;   
      elseif opts.numAugments == 2,  indices_x = [];   indices_y = [];   
      elseif opts.numAugments ~= 10, error('only 6 or 10 uniform crops allowed');  end

      for y = indices_y
        for x = indices_x
          imo(:, :, :, i, si) = ...
              imt(y:y+opts.imageSize(1)-1, x:x+opts.imageSize(2)-1, :);
          imo(:, :, :, i, si+1) = imo(:, end:-1:1, :, i, si);

          si = si + 2 ;
        end
      end
      imo(:,:,:, i,si) = imt(center_y:center_y+opts.imageSize(1)-1,center_x:center_x+opts.imageSize(2)-1,:);
      imo(:,:,:, i,si+1) = imo(:, end:-1:1, :, i, si);

      si = si + 2;

    end
  end
end

if ~isempty(opts.averageImage)
  imo = bsxfun(@minus, imo, opts.averageImage) ;
end
end