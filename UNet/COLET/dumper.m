load("data_v3.mat")

n = width(Data);

si_table = table();
for i = 1:n
    si_table = [si_table; Data(i).subject_info];
end
if ~isempty(si_table)
    writetable(si_table, 'subject_info.csv')
end

gaze_dataset = table();
for i = 1:n
    si = Data(i).subject_info;
    for t = 1:4
        gaze = Data(i).task(t).gaze;
        if ~isempty(gaze)
            si.task_number = t;
            h = height(gaze);
            si_expanded = repmat(si, h, 1);
            si_expanded = [si_expanded, gaze];
            gaze_dataset = [gaze_dataset; si_expanded];
        end
    end
end
if ~isempty(gaze_dataset)
    writetable(gaze_dataset, 'gaze_dataset.csv')
end

pupil_dataset = table();
for i = 1:n
    si = Data(i).subject_info;
    for t = 1:4
        pupil = Data(i).task(t).pupil;
        if ~isempty(pupil)
            si.task_number = t;
            h = height(pupil);
            si_expanded = repmat(si, h, 1);
            si_expanded = [si_expanded, pupil];
            pupil_dataset = [pupil_dataset; si_expanded];
        end
    end
end
if ~isempty(pupil_dataset)
    writetable(pupil_dataset, 'pupil_dataset.csv')
end

blinks_dataset = table();
for i = 1:n
    si = Data(i).subject_info;
    
    % If subject_info is a struct, convert it to a table
    if isstruct(si)
        si = struct2table(si, 'AsArray', true);
    end
    
    % Standardize 'base_data' if it is present in table 'si'
    if ismember('base_data', si.Properties.VariableNames)
        if ~iscell(si.base_data) && ~isstring(si.base_data)
            % Convert to a cell array if it is numeric or another type
            si.base_data = num2cell(si.base_data);
        end
    end

    for t = 1:4
        blinks = Data(i).task(t).blinks;
        if ~isempty(blinks)
            % Assign task_number
            si_temp = si;
            si_temp.task_number = t;
            
            % Standardize 'base_data' in 'blinks' as well if the column exists there
            if ismember('base_data', blinks.Properties.VariableNames) && ~iscell(blinks.base_data)
                blinks.base_data = num2cell(blinks.base_data);
            end

            h = height(blinks);
            si_expanded = repmat(si_temp, h, 1);
            
            % Horizontal concatenation (columns)
            si_expanded = [si_expanded, blinks];
            
            % Vertical concatenation (rows)
            blinks_dataset = [blinks_dataset; si_expanded]; 
        end
    end
end
if ~isempty(blinks_dataset)
    writetable(blinks_dataset, 'blinks_dataset.csv');
end

annotation_dataset = table();
for i = 1:n
    si = Data(i).subject_info;
    for t = 1:4
        annotation = Data(i).task(t).annotation;
        if ~isempty(annotation)
            si.task_number = t;
            h = height(annotation);
            si_expanded = repmat(si, h, 1);
            si_expanded = [si_expanded, annotation];
            annotation_dataset = [annotation_dataset; si_expanded];
        end
    end
end
if ~isempty(annotation_dataset)
    writetable(annotation_dataset, 'annotation_dataset.csv')
end