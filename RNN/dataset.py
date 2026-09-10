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
annotation_df = annotation_df.drop(columns=['VisualAcuity_logMAR_', 'Gender', 'Age', 'Education', 'physical', 'temporal', 'performance', 'effort', 'frustration', 'mean'])
annotation_df['cload'] = (annotation_df['mental'] >= 50).astype(int)
annotation_df = annotation_df.drop(columns=['mental'])

pupil_df_reduced = pupil_df_reduced.merge(annotation_df, on=['subject_id', 'task_number'])

pupil_activities = []

for subj in range(1, 48):
    for task in range(1, 5):
        for eye in range(2):
            pupil_activity = pupil_df_reduced[(pupil_df_reduced.subject_id == subj) & (pupil_df_reduced.task_number == task) & (pupil_df_reduced.eye_id == eye)].reset_index(drop=True)
            pupil_activity = pupil_activity.sort_values('pupil_timestamp')
            pupil_activities.append(pupil_activity)

# Train features:
#   - gender
#   - age
#   - diameter
# Label: cload (binary high=1 low=0)
