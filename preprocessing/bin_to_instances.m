function [instances, metadata] = bin_to_instances(filename)
% Reproduce the recorded conversion for one original 20 kHz BIN.
% Optional acquisition utility: requires fftfilt/resample and mapminmax.
% The released benchmark can be used without original BINs or this function.
here=fileparts(mfilename('fullpath'));
design=jsondecode(fileread(fullfile(here,'filter_design.json')));
fid=fopen(filename,'r','ieee-be');
assert(fid>=0,'Cannot open BIN');
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
header=fread(fid,2,'*int32');
assert(isequal(header,int32([20000;700])),'Unexpected source header');
raw=fread(fid,[700,inf],'*int16');
assert(isequal(size(raw),[700 20000]),'Unexpected source payload');
x=double(raw(80:279,:))';
x=x-mean(x,1);
filtered=fftfilt(design.bandpass_coefficients(:)',x);
resampled=resample([zeros(1,200);filtered],1,10,design.antialias_coefficients(:)');
resampled=resampled(42:2000,:);
instances=zeros(2000,50,4,'single');
metadata.normalization_min=zeros(4,1);
metadata.normalization_max=zeros(4,1);
for p=1:4
    patch=resampled(:,(p-1)*50+(1:50));
    v=reshape(patch,1,[]);
    metadata.normalization_min(p)=min(v);
    metadata.normalization_max(p)=max(v);
    normalized=(mapminmax(v)+1)/2;
    instances(42:end,:,p)=reshape(single(normalized),1959,50);
end
metadata.sampling_rate=2000;
metadata.original_sampling_rate=20000;
metadata.channel_start=uint16([80;130;180;230]);
metadata.channel_end=metadata.channel_start+uint16(49);
metadata.valid_data_mask=[false(41,1);true(1959,1)];
metadata.processing=design;
end
