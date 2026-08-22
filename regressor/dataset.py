import os

import pandas as pd

from sklearn.model_selection import train_test_split


colet_path = '/work/cvcs2026/LZMM/COLET/dumped/'

print('Loading dataset from disk...')
pupil_df = pd.read_csv(os.path.join(colet_path, 'pupil_dataset.csv'), sep=',')
print('Dataset loaded!')

print('Preparing dataset...')

pupil_drop = ['Education', 'world_index', 'norm_pos_x', 'norm_pos_y', 'method', 'ellipse_center_x', 'ellipse_center_y', 'ellipse_axis_a', 'ellipse_axis_b', 'ellipse_angle', 'diameter_3d', 'model_confidence', 'model_id', 'sphere_center_x', 'sphere_center_y', 'sphere_center_z', 'sphere_radius', 'circle_3d_center_x', 'circle_3d_center_y', 'circle_3d_center_z', 'circle_3d_normal_x', 'circle_3d_normal_y', 'circle_3d_normal_z', 'circle_3d_radius', 'theta', 'phi', 'projected_sphere_center_x', 'projected_sphere_center_y', 'projected_sphere_axis_a', 'projected_sphere_axis_b', 'projected_sphere_angle']
pupil_df_reduced = pupil_df.drop(columns=pupil_drop)

annotation_df = pd.read_csv(os.path.join(colet_path, 'annotation_dataset.csv'), sep=',')    # ground truth

metrics = ['mental', 'physical', 'temporal', 'performance', 'effort', 'frustration', 'mean']

for m in metrics:
    annotation_df[m] = annotation_df.apply(
        lambda row: row[f"{m}_{int(row['task_number'])}"], 
        axis=1
    )

cols_to_drop = [f"{m}_{i}" for m in metrics for i in range(1, 5)]
annotation_df = annotation_df.drop(columns=cols_to_drop)

pupil_df_reduced = pupil_df_reduced.sort_values(['subject_id', 'task_number', 'pupil_timestamp'], ascending=True)

pupil_activities = []

for subj in range(1, 48):
    for task in range(1, 5):
        for eye in range(2):
            pupil_activities.append(pupil_df_reduced[(pupil_df_reduced.subject_id == subj) & (pupil_df_reduced.task_number == task) & (pupil_df_reduced.eye_id == eye)].reset_index(drop=True))

dataset_features = ['subject_id', 'visual_acuity', 'gender', 'age', 'task_number', 'time', 'eye_id', 'confidence', 'diameter_avg', 'diameter_std']
pupil_dataset = pd.DataFrame(columns=dataset_features)

for pupil_activity in pupil_activities:
    summary = pd.DataFrame({
        'subject_id' : [pupil_activity.iloc[0]['subject_id']],
        'visual_acuity' : [pupil_activity.iloc[0]['VisualAcuity_logMAR_']],
        'gender' : [pupil_activity.iloc[0]['Gender']],
        'age' : [pupil_activity.iloc[0]['Age']],
        'task_number' : [pupil_activity.iloc[0]['task_number']],
        'time' : [pupil_activity['pupil_timestamp'].max() - pupil_activity['pupil_timestamp'].min()],
        'eye_id' : [pupil_activity.iloc[0]['eye_id']],
        'confidence' : [pupil_activity['confidence'].mean()],
        'diameter_avg' : [pupil_activity['diameter'].mean()],
        'diameter_std' : [pupil_activity['diameter'].std()]
    })

    pupil_dataset = pd.concat([pupil_dataset, summary], ignore_index=True)

pupil_dataset = pupil_dataset.merge(annotation_df[['subject_id', 'task_number', 'mental', 'physical', 'temporal', 'performance', 'effort', 'frustration', 'mean']], on=['subject_id', 'task_number'])
pupil_dataset['gender'] = (pupil_dataset['gender'] == 'F').astype(int)

pupil_dataset = pupil_dataset.drop(columns=['subject_id', 'task_number', 'mental', 'physical', 'temporal', 'performance', 'effort', 'frustration'])
X, y = pupil_dataset.drop(columns=['mean']), pupil_dataset['mean']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)

print('Dataset ready!')