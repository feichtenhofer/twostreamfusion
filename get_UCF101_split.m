

function [train_fnames,test_fnames, saction]= get_UCF101_split(isplit,splitdir)
  saction2 =      {'brush_hair','cartwheel','catch','chew','clap','climb','climb_stairs',...
      'dive','draw_sword','dribble','drink','eat','fall_floor','fencing',...
      'flic_flac','golf','handstand','hit','hug','jump','kick_ball',...
      'kick','kiss','laugh','pick','pour','pullup','punch',...
      'push','pushup','ride_bike','ride_horse','run','shake_hands','shoot_ball',...
      'shoot_bow','shoot_gun','sit','situp','smile','smoke','somersault',...
      'stand','swing_baseball','sword_exercise','sword','talk','throw','turn',...
      'walk','wave'};
  
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
end


       
