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
    
    % Convert subject_info to a table if it is a struct
    if isstruct(si)
        si = struct2table(si, 'AsArray', true);
    end
    
    % Standardize 'base_data' if it is present in table 'si'
    if ismember('base_data', si.Properties.VariableNames)
        if ~iscell(si.base_data) && ~isstring(si.base_data)
            si.base_data = num2cell(si.base_data);
        end
    end

    for t = 1:4
        annotation = Data(i).task(t).annotation;
        if ~isempty(annotation)
            % Assign task_number
            si_temp = si;
            si_temp.task_number = t;
            
            % Standardize 'base_data' in 'annotation' as well if the column exists there
            if ismember('base_data', annotation.Properties.VariableNames) && ~iscell(annotation.base_data)
                annotation.base_data = num2cell(annotation.base_data);
            end

            h = height(annotation);
            si_expanded = repmat(si_temp, h, 1);
            
            % Horizontal concatenation (columns)
            si_expanded = [si_expanded, annotation];
            
            % --- FIXED VERTICAL CONCATENATION ---
            if isempty(annotation_dataset)
                % On the first iteration, initialize the dataset
                annotation_dataset = si_expanded; 
            else
                % 1. Find columns present in si_expanded but NOT in the main dataset
                missing_in_dataset = setdiff(si_expanded.Properties.VariableNames, annotation_dataset.Properties.VariableNames);
                for v = 1:length(missing_in_dataset)
                    colName = missing_in_dataset{v};
                    % Create an empty column of the correct type in the dataset
                    if iscell(si_expanded.(colName))
                        annotation_dataset.(colName) = cell(height(annotation_dataset), 1);
                    elseif isstring(si_expanded.(colName))
                        annotation_dataset.(colName) = repmat(string(missing), height(annotation_dataset), 1);
                    else
                        annotation_dataset.(colName) = NaN(height(annotation_dataset), 1);
                    end
                end
                
                % 2. Find columns present in the main dataset but NOT in si_expanded
                missing_in_si = setdiff(annotation_dataset.Properties.VariableNames, si_expanded.Properties.VariableNames);
                for v = 1:length(missing_in_si)
                    colName = missing_in_si{v};
                    % Create an empty column of the correct type in si_expanded
                    if iscell(annotation_dataset.(colName))
                        si_expanded.(colName) = cell(height(si_expanded), 1);
                    elseif isstring(annotation_dataset.(colName))
                        si_expanded.(colName) = repmat(string(missing), height(si_expanded), 1);
                    else
                        si_expanded.(colName) = NaN(height(si_expanded), 1);
                    end
                end
                
                % 3. Reorder columns so they match exactly
                si_expanded = si_expanded(:, annotation_dataset.Properties.VariableNames);
                
                % 4. Now we can safely concatenate
                annotation_dataset = [annotation_dataset; si_expanded]; 
            end
        end
    end
end
if ~isempty(annotation_dataset)
    writetable(annotation_dataset, 'annotation_dataset.csv');
end