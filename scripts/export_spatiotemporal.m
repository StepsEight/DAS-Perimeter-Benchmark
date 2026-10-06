function export_spatiotemporal(source_dir, repository_dir)
% Export the paper's first 800 instances per class as native MATLAB v7.3 files.
% Source MAT files are read-only. Existing destination files are never replaced.
if nargin < 2
    repository_dir = fileparts(fileparts(mfilename('fullpath')));
end
classes = {'background','digging','raining','vehicle','walking'};
destination = fullfile(repository_dir, 'data', 'spatiotemporal');
if ~isfolder(destination), mkdir(destination); end
instance_fields = {'channel_start','channel_end','group_id','instance_index', ...
    'normalization_min','normalization_max','source_bin_index'};
bin_fields = {'bin_acquisition_datetime','bin_filenames','bin_sha256','bin_time_seconds'};
common_fields = {'event_name','sampling_rate','original_sampling_rate', ...
    'n_channels','n_time_samples','normalization','ordering','processing', ...
    'time_seconds','instance_time_seconds','valid_data_mask','valid_sample_indices', ...
    'valid_data_samples','valid_data_duration_seconds','instance_duration_seconds', ...
    'zero_padding_start_samples','zero_padding_end_samples','padding_policy'};
for c = 1:numel(classes)
    target = fullfile(destination, [classes{c} '.mat']);
    assert(~isfile(target), 'Refusing to replace an existing release MAT: %s', target);
    source = load(fullfile(source_dir,[classes{c} '.mat']));
    assert(isequal(size(source.instances,1),2000) && size(source.instances,2)==50);
    assert(size(source.instances,3)>=800);
    public = struct;
    public.instances = source.instances(:,:,1:800);
    for j=1:numel(instance_fields)
        name=instance_fields{j}; public.(name)=reshape(source.(name)(1:800),[],1);
    end
    for j=1:numel(bin_fields)
        name=bin_fields{j}; public.(name)=reshape(source.(name)(1:200),[],1);
    end
    for j=1:numel(common_fields)
        name=common_fields{j}; public.(name)=source.(name);
    end
    public.n_instances=uint32(800);
    public.class_index=uint8(c-1);
    public.class_names=classes';
    public.sample_ids=arrayfun(@(i)sprintf('%s_%06d',classes{c},i),(1:800)','UniformOutput',false);
    public.split=[repmat({'train'},480,1);repmat({'val'},160,1);repmat({'test'},160,1)];
    public.split_code=uint8([ones(480,1);2*ones(160,1);3*ones(160,1)]);
    public.train_indices=uint32((1:480)');
    public.val_indices=uint32((481:640)');
    public.test_indices=uint32((641:800)');
    public.spatial_sampling_interval_m=1;
    public.gauge_length_m=10;
    public.selected_channel_in_patch=uint16(25);
    public.dataset_version='1.0.0';
    public.license='CC-BY-NC-4.0';
    public.completed=true;
    save(target,'-struct','public','-v7.3');
    info=whos('-file',target,'instances');
    assert(isequal(info.size,[2000 50 800]) && strcmp(info.class,'single'));
    reloaded=matfile(target);
    for start=1:100:800
        stop=min(start+99,800);
        assert(isequal(reloaded.instances(:,:,start:stop),source.instances(:,:,start:stop)));
    end
    fprintf('Verified %s: 800 instances, native MATLAB v7.3, values unchanged.\n',classes{c});
    clear source public reloaded
end
end
