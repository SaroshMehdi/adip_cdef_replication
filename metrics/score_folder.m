function score_folder(folder, out_dir, metrics, shard, shards)
%SCORE_FOLDER Score every image in a folder with the authors' metric code.
%   score_folder(folder, out_dir, metrics, shard, shards)
%
%   metrics is a comma-separated list from UCIQE, CEIQ, FDUM, CCF. shard and
%   shards split the images across several MATLAB processes.
%
%   Each score is appended to out_dir/<metric>_shard<k>.csv as soon as it is
%   computed, and images already in those files are skipped, so a stopped run
%   resumes. This file only loops over images: every score comes from code in
%   metrics/third_party, which is unmodified. An image a metric cannot score
%   gets NaN, and the error goes to out_dir/errors_shard<k>.txt.
%
%   PNG and JPEG files are scored, including those in subfolders (UCCS keeps
%   one folder per colour cast). File names must be unique across subfolders.

root = fullfile(fileparts(mfilename('fullpath')), 'third_party');
files = [dir(fullfile(folder, '**', '*.png')); dir(fullfile(folder, '**', '*.jpg')); ...
         dir(fullfile(folder, '**', '*.jpeg'))];
locations = containers.Map({files.name}, {files.folder});
% Each process gets a fixed share. It must not depend on what is already
% scored, because the processes reach each metric at different times.
my_names = sort(unique({files.name}));
my_names = my_names(shard:shards:numel(my_names));

for metric = strsplit(metrics, ',')
    name = metric{1};
    names = setdiff(my_names, scored_images(out_dir, name));
    if isempty(names)
        continue
    end

    [folders, scorer] = metric_setup(name, root);
    saved = path;
    addpath(folders{:});
    out_csv = fullfile(out_dir, sprintf('%s_shard%d.csv', name, shard));
    for i = 1:numel(names)
        try
            value = scorer(imread(fullfile(locations(names{i}), names{i})));
        catch err
            value = NaN;
            log_error(out_dir, shard, name, names{i}, err.message);
        end
        fid = fopen(out_csv, 'a');
        fprintf(fid, '%s,%.10g\n', names{i}, value);
        fclose(fid);
    end
    path(saved);
end
end


function names = scored_images(out_dir, metric)
% Images this metric has already scored, in any shard.
names = {};
parts = dir(fullfile(out_dir, sprintf('%s_shard*.csv', metric)));
for i = 1:numel(parts)
    lines = readlines(fullfile(out_dir, parts(i).name));
    lines = lines(strlength(lines) > 0);
    names = [names, cellstr(extractBefore(lines, ','))']; %#ok<AGROW>
end
end


function log_error(out_dir, shard, metric, image, message)
fid = fopen(fullfile(out_dir, sprintf('errors_shard%d.txt', shard)), 'a');
fprintf(fid, '%s %s: %s\n', metric, image, message);
fclose(fid);
end


function [folders, scorer] = metric_setup(name, root)
% The metrics stay on separate paths: CCF and CEIQ both ship a model.mat.
switch name
    case 'UCIQE'
        folders = {fullfile(root, 'UCIQE')};
        scorer = @(I) UCIQE(I);
    case 'CEIQ'
        folders = {fullfile(root, 'CEIQ'), fullfile(root, 'CEIQ', 'utils'), ...
                   fullfile(root, 'CEIQ', 'data')};
        scorer = @quiet_ceiq;
    case 'FDUM'
        folders = {fullfile(root, 'FDUM', 'extracted', 'Code-for-FDUM-method', 'Code')};
        scorer = @fdum;
    case 'CCF'
        folders = {fullfile(root, 'CCF')};
        scorer = @quiet_ccf;
    otherwise
        error('unknown metric: %s', name);
end
end


function value = quiet_ceiq(I)
% svmpredict prints to the console.
[~, value] = evalc('CEIQ(I)');
end


function value = quiet_ccf(I)
% CCF.m has no semicolon on its result line.
[~, value] = evalc('CCF(I)');
end


function value = fdum(I)
% The body of the loop in demo_FDUM.m, which is a script with a fixed folder.
Coe = [0.2982 0.4439 0.028];
Color = Colorfulness(I);
Con = Contrast(I);
Sharp = UIQMSharpness(I);
value = Coe(1) * Color + Coe(2) * Con + Coe(3) * Sharp;
end
