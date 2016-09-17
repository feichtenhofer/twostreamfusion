function imo = cnn_ucf101_get_flow_batch(images, varargin)

opts.subTractFlow = 'off';
opts.nFramesPerVid = 1;
opts.numAugments = 1;
opts.frameSample = 'uniformly';
opts.imageDir = '';
opts.temporalStride = 0;

opts.imageSize = [227, 227 20] ;
opts.border = [29, 29] ;
opts.averageImage = [] ;
opts.rgbVariance = [] ;
opts.augmentation = 'croponly' ;
opts.interpolation = 'bilinear' ;
opts.numAugments = 1 ;
opts.numThreads = 0 ;
opts.prefetch = false ;
opts.keepAspect = true;
opts.doResize = false;
opts.cheapResize = false;
opts.nFrames = [];
opts.frameList = NaN;
opts.imReadSz = [];
opts.subMedian = false;
opts.stretchAspect = 4/3 ;
opts.stretchScale = 1.2 ;
[opts, varargin] = vl_argparse(opts, varargin);

prefetch = opts.prefetch & isempty(opts.frameList);

switch opts.augmentation
  case 'croponly'
    tfs = [.5 ; .5 ; 0 ];
  case 'f5'
    tfs = [...
      .5 0 0 1 1 .5 0 0 1 1 ;
      .5 0 1 0 1 .5 0 1 0 1 ;
       0 0 0 0 0  1 1 1 1 1] ;
  case 'f25'
    [tx,ty] = meshgrid(linspace(0,1,5)) ;
    tfs = [tx(:)' ; ty(:)' ; zeros(1,numel(tx))] ;
    tfs_ = tfs ;
    tfs_(3,:) = 1 ;
    tfs = [tfs,tfs_] ;
  case {'f25noCtr','borders'}
    [tx1,ty1] = meshgrid(linspace(.75,1,20)) ;
    [tx2,ty2] = meshgrid(linspace(0,.25,20)) ;
    tx = [tx1 tx2];     ty = [ty1 ty2];
    tfs = [tx(:)' ; ty(:)' ; zeros(1,numel(tx))] ;
    tfs_ = tfs ;
    tfs_(3,:) = 1 ;
    tfs = [tfs,tfs_] ;   
  case {'borders15'}
    [tx1,ty1] = meshgrid(linspace(.85,1,20)) ;
    [tx2,ty2] = meshgrid(linspace(0,.15,20)) ;
    tx = [tx1 tx2];     ty = [ty1 ty2];
    tfs = [tx(:)' ; ty(:)' ; zeros(1,numel(tx))] ;
    tfs_ = tfs ;
    tfs_(3,:) = 1 ;
    tfs = [tfs,tfs_] ;   
  case {'borders5'}
    [tx1,ty1] = meshgrid(linspace(.95,1,20)) ;
    [tx2,ty2] = meshgrid(linspace(0,.05,20)) ;
    tx = [tx1 tx2];     ty = [ty1 ty2];
    tfs = [tx(:)' ; ty(:)' ; zeros(1,numel(tx))] ;
    tfs_ = tfs ;
    tfs_(3,:) = 1 ;
    tfs = [tfs,tfs_] ; 
end

nStack = opts.imageSize(3);

if iscell(opts.frameList)
  if isempty(opts.imReadSz)
    im = vl_imreadjpeg(opts.frameList{1}, 'numThreads', opts.numThreads ) ; 
  else
    im = vl_imreadjpeg(opts.frameList{1}, 'numThreads', opts.numThreads, 'resize', opts.imReadSz ) ; 
  end
  sampled_frame_nr = opts.frameList{2};
else
  sampleFrameLeftRight = floor(nStack/4);
  frameOffsets = [-sampleFrameLeftRight:sampleFrameLeftRight-1]';

  frames = cell(numel(images), nStack, opts.nFramesPerVid);

  sampled_frame_nr = cell(numel(images),1);
  for i=1:numel(images)
    vid_name = images{i};
    nFrames = opts.nFrames(i);
    if  strcmp(opts.frameSample, 'uniformly')
      sampleRate = max(floor((nFrames-nStack/2)/opts.nFramesPerVid),1);
      frameSamples = nStack/4+1:sampleRate:nFrames - nStack/4 ;
      frameSamples = vl_colsubset(nStack/4+1:nFrames-nStack/4, opts.nFramesPerVid, 'uniform') ;
    elseif strcmp(opts.frameSample, 'temporalStride')
      shift = floor(mod(nFrames-nStack/2,opts.temporalStride)/2); 
      frameSamples = nStack/4:opts.temporalStride:nFrames-nStack/4-1;
      frameSamples = frameSamples + shift + 1;
      frameSamples = vl_colsubset(frameSamples, opts.nFramesPerVid, 'uniform') ;
    elseif strcmp(opts.frameSample, 'random')
      frameSamples = randperm(nFrames-nStack/2)+nStack/4;
    elseif strcmp(opts.frameSample, 'temporalStrideRandom')
      sampleStart = randi(opts.temporalStride) + nStack/4 ; 
      frameSamples = sampleStart:opts.temporalStride:nFrames - nStack/4;
      if isempty(frameSamples)
        frameSamples = nStack/4+1:opts.temporalStride:nFrames - nStack/4;
      end
    end   

    if length(frameSamples) < opts.nFramesPerVid,
      frameSamples = padarray(frameSamples,[0 opts.nFramesPerVid - length(frameSamples)],'symmetric','post');
    elseif length(frameSamples) > opts.nFramesPerVid,
      s = randi(length(frameSamples)-opts.nFramesPerVid);
      frameSamples = frameSamples(s:s+opts.nFramesPerVid-1);
    end
    frameSamples =  repmat(frameSamples,nStack/2,1) +  repmat(frameOffsets,1,size(frameSamples,2));
      
    for k = 1:opts.nFramesPerVid
      for j = 1:nStack/2
        frames{i,(j-1)*2+1, k} = ['u' filesep vid_name 'frame' sprintf('%06d.jpg', frameSamples(j,k)) ] ;
        frames{i,(j-1)*2+2, k} = ['v' frames{i,(j-1)*2+1, k}(2:end)];
      end
    end
    sampled_frame_nr{i} = frameSamples;

  end
  frames = strcat([ opts.imageDir filesep], frames);
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
      for l=1:size(im,2)
          im{i,l,k} = im{i,l,k}(1:h_min,1:w_min,:);
      end
    end   

    imt = (cat(3, im{i,:,k}) );      
%     imt = gpuArray(imt); % if you have plenty of gpu mem (also init imo
%     as gpuArray then)
    if any(scal ~= 1)
      imt = imresize(imt, sz );
    end

    imo(:, :, :, i, si) = imt;
    imo(:, :, 1:2:nStack, i, si+1) = -imt(:, end:-1:1, 1:2:nStack) + 255; 
    imo(:, :, 2:2:nStack, i, si+1) = imt(:, end:-1:1, 2:2:nStack);      

    si = si + 2;

    end
  end  
  if ~isempty(opts.averageImage)
    opts.averageImage = mean(mean(opts.averageImage,1),2) ;
    imo = bsxfun(@minus, imo, opts.averageImage) ;
  end
  return;
end

%% augment now
if exist('tfs', 'var')
  [~,transformations] = sort(rand(size(tfs,2), numel(images)*opts.nFramesPerVid), 1) ;
end

imo =  ( zeros(opts.imageSize(1), opts.imageSize(2), opts.imageSize(3), ...
            numel(images), opts.numAugments * opts.nFramesPerVid, 'single') );

for i=1:numel(images)
  si = 1 ;

  w = size(im{i,1,1},2) ;
  h = size(im{i,1,1},1) ;
  
  
  if  strcmp( opts.augmentation, 'multiScaleRegular')
    reg_szs = [256, 224, 192, 168] ;          
    sz(1) = reg_szs(randi(4)); sz(2) = reg_szs(randi(4));
  elseif strcmp( opts.augmentation, 'stretch')
    aspect = exp((2*rand-1) * log(opts.stretchAspect)) ;
    scale = exp((2*rand-1) * log(opts.stretchScale)) ;
    tw = opts.imageSize(2) * sqrt(aspect) * scale ;
    th = opts.imageSize(1) / sqrt(aspect) * scale ;
    reduce = min([w / tw, h / th, 1]) ;
    sz = round(reduce * [th ; tw]) ;
  elseif any(strcmp( opts.augmentation, {'corners','borders15','borders5'}))
     sz(1) = round(160 + (256-160).*rand());
     sz(2) = round(160 + (256-160).*rand());  
  else
    sz = round(min(opts.imageSize(1:2)' .* (.75+0.5*rand(2,1)), [h; w])) ; % 0.75 +- 0.5, not keep aspect  
  end
  for k = 1:opts.nFramesPerVid
    imt = cat(3, im{i,:,k}) ;
%     imt = gpuArray(imt); % if plenty of gpu mem do augmentations on gpu

    w = size(imt,2) ;
    h = size(imt,1) ;
    factor = [(opts.imageSize(1)+opts.border(1))/h ...
              (opts.imageSize(2)+opts.border(2))/w];

    if opts.keepAspect
      factor = max(factor) ;
    end
    if opts.doResize && any(abs(factor - 1) > 0.0001)
      imt = imresize(imt, ...
                     'scale', factor, ...
                     'method', opts.interpolation) ;
    end
    w = size(imt,2) ;
    h = size(imt,1) ;
    if ~strcmp(opts.augmentation, 'uniform')
      if ~isempty(opts.rgbVariance) 
        offset = zeros(size(imt));
        offset = bsxfun(@minus, offset, reshape(opts.rgbVariance * randn(opts.imageSize(3),1), 1,1,opts.imageSize(3))) ;
        imt = bsxfun(@minus, imt, offset) ;
      end
      for ai = 1:opts.numAugments
        switch opts.augmentation
          case 'stretch'
            dx = randi(w - sz(2) + 1, 1) ;
            dy = randi(h - sz(1) + 1, 1) ;
            flip = rand > 0.5 ;
          case 'randCropFlipStretch'
            sz = round(min(opts.imageSize(1:2)' .* (.75+0.5*rand(2,1)), [h;w])) ; % 0.75 +- 0.5, not keep aspect            
            dx = randi(w - sz(2) + 1, 1) ;
            dy = randi(h - sz(1) + 1, 1) ;                     
            flip = rand > 0.5 ;
          case 'bodersOnly'
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
          case {'multiScaleRegular','corners'}
            dy = [0 h-sz(1) 0 h-sz(1)  floor((h-sz(1)+1)/2)] + 1;
            dx = [0 w-sz(2) w-sz(2) 0 floor((w-sz(2)+1)/2)] + 1;
            corner = randi(5);
            dx = dx(corner); dy = dy(corner); 
            flip = rand > 0.5 ;  
          case {'f25noCtr','borders','borders15','borders5'}
            tf = tfs(:, transformations(mod(i+ai-1, numel(transformations)) + 1)) ;
            dx = floor((w - sz(2)) * tf(2)) + 1 ;
            dy = floor((h - sz(1)) * tf(1)) + 1 ;
            flip = tf(3) ;  
          otherwise
            sz = opts.imageSize(1:2) ;
            tf = tfs(:, transformations(mod(ai-1, numel(transformations)) + 1)) ;
            dx = floor((w - sz(2)) * tf(2)) + 1 ;
            dy = floor((h - sz(1)) * tf(1)) + 1 ;
            flip = tf(3) ;              
        end
        if opts.cheapResize
          sx = round(linspace(dx, sz(2)+dx-1, opts.imageSize(2))) ;
          sy = round(linspace(dy, sz(1)+dy-1, opts.imageSize(1))) ;
        else
          factor = [opts.imageSize(1)/sz(1) ...
              opts.imageSize(2)/sz(2)];
          if any(abs(factor - 1) > 0.0001)
              imt =   imresize(gather(imt(dy:sz(1)+dy-1,dx:sz(2)+dx-1,:)), [opts.imageSize(1:2)], 'Antialiasing', false);
          end
          sx = 1:opts.imageSize(2); sy = 1:opts.imageSize(1);

        end

        if flip
          sx = fliplr(sx) ;
          imo(:,:,1:2:nStack,i,si) = -imt(sy,sx,1:2:nStack) + 255; 

          imo(:,:,2:2:nStack,i,si) = imt(sy,sx,2:2:nStack) ;
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
      elseif opts.numAugments ~= 10, error('only 6 or 10 uniform crops allowed');  
      end

      for y = indices_y
        for x = indices_x
          imo(:, :, :, i, si) = ...
              imt(y:y+opts.imageSize(1)-1, x:x+opts.imageSize(2)-1, :);
          imo(:, :, 1:2:nStack, i, si+1) = -imo(:, end:-1:1, 1:2:nStack, i, si) + 255;
          imo(:, :, 2:2:nStack, i, si+1) = imo(:, end:-1:1, 2:2:nStack, i, si);

          si = si + 2 ;
        end
      end
      imo(:,:,:, i,si) = imt(center_y:center_y+opts.imageSize(1)-1,center_x:center_x+opts.imageSize(2)-1,:);
      imo(:,:,1:2:nStack, i,si+1) = -imo(:, end:-1:1, 1:2:nStack, i, si) + 255; 
      imo(:,:,2:2:nStack, i,si+1) = imo(:, end:-1:1, 2:2:nStack, i, si);

      si = si + 2;

    end

  end
end


if ~isempty(opts.averageImage)
  imo = bsxfun(@minus, imo, opts.averageImage) ;
end

end