#python deepEye_fairness_gazebase.py -inspect_key "random" -use_percentages 1 -use_trial_types "['TEX']" -number_train 100 -seconds_per_user 80 -num_folds 10 -GPU 1
#python deepEye_fairness_gazebase.py -inspect_key "random" -use_percentages 1 -use_trial_types "['BLG']" -number_train 100 -seconds_per_user 80 -num_folds 10 -GPU 1
#python deepEye_fairness_gazebase.py -inspect_key "random" -use_percentages 1 -use_trial_types "['FXS']" -number_train 100 -seconds_per_user 80 -num_folds 10 -GPU 1
#python deepEye_fairness_gazebase.py -inspect_key "random" -use_percentages 1 -use_trial_types "['HSS']" -number_train 100 -seconds_per_user 80 -num_folds 10 -GPU 1
#python deepEye_fairness_gazebase.py -inspect_key "random" -use_percentages 1 -use_trial_types "['RAN']" -number_train 100 -seconds_per_user 80 -num_folds 10 -GPU 1
#python deepEye_fairness_gazebase.py -inspect_key "random" -use_percentages 1 -use_trial_types "['VD1,VD2']" -number_train 100 -seconds_per_user 80 -num_folds 10 -GPU 1
python deepEye_fairness_gazebase.py -inspect_key "random" -use_percentages 1 -use_trial_types "['TEX','BLG','FXS','HSS','RAN','VD1,VD2']" -number_train 100 -seconds_per_user 80 -num_folds 10 -GPU 1
