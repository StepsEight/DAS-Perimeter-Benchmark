% Read the downloaded data, independent of MATLAB's current directory.
repository_dir=fileparts(fileparts(mfilename('fullpath')));
S=load(fullfile(repository_dir,'data','spatiotemporal','walking.mat'));
i=241; % 1-based class instance index; global IDs are in data/metadata.csv.
stored=S.instances(:,:,i); % single [2000,50], valid region min-max scaled.
valid=logical(S.valid_data_mask(:));
restored=double(stored);
restored(valid,:)=restored(valid,:)*(S.normalization_max(i)-S.normalization_min(i))+S.normalization_min(i);
restored(~valid,:)=0;
% Match the 2D baseline's float32 restoration followed by float64 statistics.
patch=double(single(restored));
values=patch(valid,:);
patch(valid,:)=(values-mean(values,'all'))/(std(values(:),1)+1e-8);
patch(~valid,:)=0;
patch=single(patch'); % [50,2000], space x time.
T=load(fullfile(repository_dir,'data','temporal','DAS_1D_2kHz_4000.mat'));
row=find(strcmp(T.sample_ids,S.sample_ids{i}));
waveform=T.waveforms(row,:); % Already standardized; [1,2000].
fprintf('%s: split=%s, space-time=%dx%d, temporal=%d points\n', ...
    S.sample_ids{i},S.split{i},size(patch,1),size(patch,2),numel(waveform));
