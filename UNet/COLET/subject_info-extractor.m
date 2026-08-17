n = width(Data.Data);
subject_id = (1:n)';
visual_acuity = zeros(n,1);
gender = strings(n,1);
age = zeros(n,1);
education = zeros(n,1);
for i = 1:n
    si = Data.Data(i).subject_info;
    visual_acuity(i) = si.VisualAcuity_logMAR_;
    gender(i) = string(si.Gender{1});   % Gender is a cell containing a char, unwrap it
    age(i) = si.Age;
    education(i) = si.Education;
end
subj_table = table(subject_id, visual_acuity, gender, age, education,'VariableNames', {'subject_id','visual_acuity_logmar','gender','age','education'});
writetable(subj_table, 'subject_info.csv');
disp('subject_info.csv saved.');