load("data_v3.mat")

n = width(Data);

si_table = table();
for i = 1:n
    si_table = [si_table; Data(i).subject_info];
end
writetable(si_table, 'subject_info.csv')

gaze_dataset = table();
for i = 1:n
    si = Data(i).subject_info;
    for t = 1:4
        si.task_number = t;
        gaze = Data(i).task(t).gaze;
        h = height(gaze);
        si_expanded = repmat(si, h, 1);
        si_expanded = [si_expanded, gaze];
        gaze_dataset = [gaze_dataset; si_expanded];
    end
end
writetable(gaze_dataset, 'gaze_dataset.csv')

pupil_dataset = table();
for i = 1:n
    si = Data(i).subject_info;
    for t = 1:4
        si.task_number = t;
        pupil = Data(i).task(t).pupil;
        h = height(pupil);
        si_expanded = repmat(si, h, 1);
        si_expanded = [si_expanded, pupil];
        pupil_dataset = [pupil_dataset; si_expanded];
    end
end
writetable(pupil_dataset, 'pupil_dataset.csv')

blinks_dataset = table();
for i = 1:n
    si = Data(i).subject_info;
    for t = 1:4
        si.task_number = t;
        blinks = Data(i).task(t).blinks;
        h = height(blinks);
        si_expanded = repmat(si, h, 1);
        si_expanded = [si_expanded, blinks];
        if ~isempty(blinks_dataset)
            blinks_dataset = [blinks_dataset; si_expanded];
        end
    end
end
writetable(blinks_dataset, 'blinks_dataset.csv')

annotation_dataset = table();
for i = 1:n
    si = Data(i).subject_info;
    for t = 1:4
        si.task_number = t;
        annotation = Data(i).task(t).annotation;
        h = height(annotation);
        si_expanded = repmat(si, h, 1);
        si_expanded = [si_expanded, annotation];
        if ~isempty(annotation_dataset)
            annotation_dataset = [annotation_dataset; si_expanded];
        end
    end
end
writetable(annotation_dataset, 'annotation_dataset.csv')