function imo = cnn_ucf101_get_im_flow_batch(images, varargin)

opts.nFramesPerVid = 1;
opts.numAugments = 1;
opts.frameSample = 'uniformly';
opts.flowDir = '';
opts.imageDir = '';
opts.temporalStride = 0;
opts.imageSize = [224, 224] ;
opts.border = [32, 32] ;
opts.averageImage = [] ;
opts.rgbVariance = [] ;
opts.augmentation = 'croponly' ;
opts.interpolation = 'bilinear' ;
opts.numAugments = 1 ;
opts.numThreads = 1 ;
opts.prefetch = false ;
opts.keepAspect = true;
opts.cheapResize = 0;
opts.nFrameStack = 10;
opts.frameList = NaN;
opts.nFrames = [];
[opts, varargin] = vl_argparse(opts, varargin);

flowDir = opts.flowDir;
imgDir = opts.imageDir;
% prefetch is used to load images in a separate thread
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
  case 'noCtr'
    [tx1,ty1] = meshgrid(linspace(.75,1,20)) ;
    [tx2,ty2] = meshgrid(linspace(0,.25,20)) ;
    tx = [tx1 tx2];     ty = [ty1 ty2];
    tfs = [tx(:)' ; ty(:)' ; zeros(1,numel(tx))] ;
    tfs_ = tfs ;
    tfs_(3,:) = 1 ;
    tfs = [tfs,tfs_] ;       
end

nStack = opts.imageSize(3);

if iscell(opts.frameList)
  im = vl_imreadjpeg(opts.frameList{1}, 'numThreads', opts.numThreads) ; 
  sampled_frame_nr = opts.frameList{2};
else
  sampleFrameLeftRight = floor(nStack/4); % divide by 4 because of left,right,u,v
  frameOffsets = [-sampleFrameLeftRight:sampleFrameLeftRight-1]';
  frames = cell(numel(images), nStack, opts.nFramesPerVid);
  frames_rgb = cell(numel(images), 1, opts.nFramesPerVid);
  sampled_frame_nr = cell(numel(images),1);

  
  for i=1:numel(images)
    vid_name = images{i};
    nFrames = opts.nFrames(i);
    if  strcmp(opts.frameSample, 'uniformly')
      sampleRate = max(floor((nFrames-nStack/2)/opts.nFramesPerVid),1);
      opts.temporalStride = sampleRate;
      frameSamples = vl_colsubset(nStack/4+1:nFrames-nStack/4, opts.nFramesPerVid, 'uniform') ;
    elseif strcmp(opts.frameSample, 'temporalStride')
      frameSamples = nStack/4+1:opts.temporalStride:nFrames-nStack/4 ;
      if length(frameSamples) < opts.nFrameStack,
          frameSamples = round(linspace(nStack/4+1, nFrames - nStack/4, opts.nFrameStack)) ;
          opts.temporalStride = frameSamples(2) - frameSamples(1);
      end 
    elseif strcmp(opts.frameSample, 'random')
      frameSamples = randperm(nFrames-nStack/2)+nStack/4;
    elseif strcmp(opts.frameSample, 'temporalStrideRandom')
      frameSamples = nStack/4 +1:opts.temporalStride:nFrames - nStack/4 ;
   end  

    if length(frameSamples) < opts.nFramesPerVid,
        if length(frameSamples) > opts.nFrameStack
          frameSamples = frameSamples(1:length(frameSamples)-mod(length(frameSamples),opts.nFrameStack));
        end
        diff =  opts.nFramesPerVid - length(frameSamples);

        addFrames = 0;
        while diff > 0
         last_frame = min(frameSamples(end), max(nFrames - nStack/4 - opts.nFrameStack,nStack/4 )); 
          if mod(addFrames,2) % add to the front
            addSamples = nStack/4+1:opts.temporalStride:nFrames - nStack/4;
            addSamples = addSamples(1: length(addSamples) - mod(length(addSamples),opts.nFrameStack));
            if length(addSamples) > diff, addSamples = addSamples(1:diff); end
          else % add to the back
            addSamples = fliplr(nFrames - nStack/4 : -opts.temporalStride: nStack/4+1);           
            addSamples = addSamples(mod(length(addSamples),opts.nFrameStack)+1:length(addSamples));
            if length(addSamples) > diff, addSamples = addSamples(end-diff+1:end); end
          end

          if addFrames > 20,  
                addSamples = round(linspace(nStack/4+1, nFrames - nStack/4, opts.nFrameStack)) ;
          end
          frameSamples = [frameSamples addSamples]; 
          diff = opts.nFramesPerVid - length(frameSamples); 
          opts.temporalStride = max(ceil(opts.temporalStride-1), 1);
          addFrames = addFrames+1;
        end
    end
    if length(frameSamples) > opts.nFramesPerVid   
        s = randi(length(frameSamples)-opts.nFramesPerVid);
        frameSamples = frameSamples(s+1:s+opts.nFramesPerVid);
    end

    for k = 1:opts.nFramesPerVid
      frames_rgb{i,1,k} = [vid_name 'frame' sprintf('%06d.jpg', frameSamples(k))] ;
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
  
  frames_rgb = strcat([imgDir filesep], frames_rgb);
  if ~isempty(flowDir)
    frames = strcat([flowDir filesep], frames);
    frames = cat(2, frames, frames_rgb);
  else
    frames = frames_rgb;
  end
  
  if opts.numThreads > 0
    if prefetch
      vl_imreadjpeg(frames, 'numThreads', opts.numThreads, 'prefetch') ;
      imo = {frames sampled_frame_nr}  ;
      return ;
    end
    im = vl_imreadjpeg(frames, 'numThreads', opts.numThreads) ;
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

  imo = ( zeros(sz(1), sz(2), opts.imageSize(3)+3, ...
            numel(images), 2 * opts.nFramesPerVid, 'single') );
        
  for i=1:numel(images)
    si = 1 ;
    for k = 1:opts.nFramesPerVid
      if numel(unique(szw)) > 1 || numel(unique(szh)) > 1
        for l=1:size(im,2)
            im{i,l,k} = im{i,l,k}(1:h_min,1:w_min,:);
        end
      end   
      imt = cat(3, im{i,:,k}) ;      
      if any(scal ~= 1)
        imt = imresize(imt, sz );
      end
      imo(:, :, :, i, si) = imt;
      imo(:, :, :, i, si+1) = imt(:, end:-1:1,:);      
      imo(:, :, 1:2:nStack, i, si+1) = -imt(:, end:-1:1, 1:2:nStack) + 255; %invert u if we flip   
      si = si + 2 ;
    end
  end  
  
  if ~isempty(opts.averageImage)
    opts.averageImage = mean(mean(opts.averageImage,1),2) ;   
    imo = bsxfun(@minus, imo,opts.averageImage) ;
  end
  return;
end

% augment now
if exist('tfs', 'var')
  [~,transformations] = sort(rand(size(tfs,2), numel(images)*opts.nFramesPerVid), 1) ;
end

imo = ( zeros(opts.imageSize(1), opts.imageSize(2), opts.imageSize(3)+3, ...
            numel(images), opts.numAugments * opts.nFramesPerVid, 'single') ) ;
for i=1:numel(images)
  si = 1 ;
  szw = cellfun(@(x) size(x,2),im);
  szh = cellfun(@(x) size(x,1),im);  
  h_min = min(szh(:));
  w_min =  min(szw(:));

  sz = round(min(opts.imageSize(1:2)' .* (.75+0.5*rand(2,1)), [h_min; w_min])) ; % 0.75 +- 0.5, not keep aspect  
  for k = 1:opts.nFramesPerVid
    if numel(unique(szw)) > 1 || numel(unique(szh)) > 1
      for l=1:size(im,2)
        im{i,l,k} = im{i,l,k}(1:h_min,1:w_min,:);
      end
    end
    imt = cat(3, im{i,:,k}) ;
    w = size(imt,2) ;
    h = size(imt,1) ;
    if ~strcmp(opts.augmentation, 'uniform')
      if ~isempty(opts.rgbVariance) % colour jittering only in training case
        offset = zeros(size(imt));
        offset = bsxfun(@minus, offset, reshape(opts.rgbVariance * randn(opts.imageSize(3),1), 1,1,opts.imageSize(3))) ;
        imt = bsxfun(@minus, imt, offset) ;
      end

      for ai = 1:opts.numAugments
        switch opts.augmentation
          case 'randCropFlip'
            sz = opts.imageSize(1:2) ;
            dx = randi(w - sz(2) + 1 ) ;
            dy = randi(h - sz(1) + 1 ) ;
            flip = rand > 0.5 ;
          case 'noCtr'
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
            imt =   imresize(gather(imt(dy:sz(1)+dy-1,dx:sz(2)+dx-1,:)), [opts.imageSize(1:2)]);
          end                   
          sx = 1:opts.imageSize(2); sy = 1:opts.imageSize(1);
        end
        
        if flip
          sx = fliplr(sx) ;
          imo(:,:,:,i,si) = imt(sy,sx,:) ;
          imo(:,:,1:2:nStack,i,si) = -imt(sy,sx,1:2:nStack) + 255; %invert u if we flip
        else
          imo(:,:,:,i,si) = imt(sy,sx,:) ;
        end

        si = si + 1 ;
      end
    else
      % sample (4 corners, center, and their x-axis flips)
      w = size(imt,2) ; h = size(imt,1) ;
      indices_y = [0 h-opts.imageSize(1)] + 1;
      indices_x = [0 w-opts.imageSize(2)] + 1;
      center_y = floor(indices_y(2) / 2)+1;
      center_x = floor(indices_x(2) / 2)+1;
      if opts.numAugments == 6,  indices_y = center_y;   
      elseif opts.numAugments == 2,  indices_x = [];   indices_y = [];  
      elseif opts.numAugments ~= 10, error('only 6 or 10 uniform crops allowed');  
      end
      for y = indices_y
      for x = indices_x
        imo(:, :, :, i, si) = ...
            imt(y:y+opts.imageSize(1)-1, x:x+opts.imageSize(2)-1, :);
        imo(:, :, :, i, si+1) = imo(:, end:-1:1, :, i, si);          
        imo(:, :, 1:2:nStack, i, si+1) = -imo(:, end:-1:1, 1:2:nStack, i, si) + 255; %invert u if we flip
        si = si + 2 ;
      end
      end
      imo(:,:,:, i,si) = imt(center_y:center_y+opts.imageSize(1)-1,center_x:center_x+opts.imageSize(2)-1,:);
      imo(:,:,:, i,si+1) = imo(:, end:-1:1, :, i, si);        
      imo(:,:,1:2:nStack, i,si+1) = -imo(:, end:-1:1, 1:2:nStack, i, si) + 255; %invert u if we flip
      si = si + 2;
    end
  end
end

if ~isempty(opts.averageImage)
  imo = bsxfun(@minus, imo, opts.averageImage) ;
end

end
