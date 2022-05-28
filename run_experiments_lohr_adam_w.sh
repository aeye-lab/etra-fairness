# Fix adam_w
python lohr_fairness_gazebase.py  -use_trial_types "['TEX']" -number_train 100  -num_folds 10 -GPU 1 -feature Fix -optimizer adam_w -save_dir saved_lohr_embeddings_adam_w/ -save_prefix fix_
python lohr_fairness_gazebase.py  -use_trial_types "['BLG']" -number_train 100  -num_folds 10 -GPU 1 -feature Fix -optimizer adam_w -save_dir saved_lohr_embeddings_adam_w/ -save_prefix fix_
python lohr_fairness_gazebase.py  -use_trial_types "['FXS']" -number_train 100  -num_folds 10 -GPU 1 -feature Fix -optimizer adam_w -save_dir saved_lohr_embeddings_adam_w/ -save_prefix fix_
python lohr_fairness_gazebase.py  -use_trial_types "['HSS']" -number_train 100  -num_folds 10 -GPU 1 -feature Fix -optimizer adam_w -save_dir saved_lohr_embeddings_adam_w/ -save_prefix fix_
python lohr_fairness_gazebase.py  -use_trial_types "['RAN']" -number_train 100  -num_folds 10 -GPU 1 -feature Fix -optimizer adam_w -save_dir saved_lohr_embeddings_adam_w/ -save_prefix fix_
python lohr_fairness_gazebase.py  -use_trial_types "['VD1','VD2']" -number_train 100  -num_folds 10 -GPU 1 -feature Fix -optimizer adam_w -save_dir saved_lohr_embeddings_adam_w/ -save_prefix fix_

# Sac adam
python lohr_fairness_gazebase.py  -use_trial_types "['TEX']" -number_train 100  -num_folds 10 -GPU 1 -feature Sac -optimizer adam_w -save_dir saved_lohr_embeddings_adam_w/ -save_prefix sac_
python lohr_fairness_gazebase.py  -use_trial_types "['BLG']" -number_train 100  -num_folds 10 -GPU 1 -feature Sac -optimizer adam_w -save_dir saved_lohr_embeddings_adam_w/ -save_prefix sac_
python lohr_fairness_gazebase.py  -use_trial_types "['FXS']" -number_train 100  -num_folds 10 -GPU 1 -feature Sac -optimizer adam_w -save_dir saved_lohr_embeddings_adam_w/ -save_prefix sac_
python lohr_fairness_gazebase.py  -use_trial_types "['HSS']" -number_train 100  -num_folds 10 -GPU 1 -feature Sac -optimizer adam_w -save_dir saved_lohr_embeddings_adam_w/ -save_prefix sac_
python lohr_fairness_gazebase.py  -use_trial_types "['RAN']" -number_train 100  -num_folds 10 -GPU 1 -feature Sac -optimizer adam_w -save_dir saved_lohr_embeddings_adam_w/ -save_prefix sac_
python lohr_fairness_gazebase.py  -use_trial_types "['VD1','VD2']" -number_train 100  -num_folds 10 -GPU 1 -feature Sac -optimizer adam_w -save_dir saved_lohr_embeddings_adam_w/ -save_prefix sac_

# PSO adam
python lohr_fairness_gazebase.py  -use_trial_types "['TEX']" -number_train 100  -num_folds 10 -GPU 1 -feature PSO -optimizer adam_w -save_dir saved_lohr_embeddings_adam_w/ -save_prefix pso_
python lohr_fairness_gazebase.py  -use_trial_types "['BLG']" -number_train 100  -num_folds 10 -GPU 1 -feature PSO -optimizer adam_w -save_dir saved_lohr_embeddings_adam_w/ -save_prefix pso_
python lohr_fairness_gazebase.py  -use_trial_types "['FXS']" -number_train 100  -num_folds 10 -GPU 1 -feature PSO -optimizer adam_w -save_dir saved_lohr_embeddings_adam_w/ -save_prefix pso_
python lohr_fairness_gazebase.py  -use_trial_types "['HSS']" -number_train 100  -num_folds 10 -GPU 1 -feature PSO -optimizer adam_w -save_dir saved_lohr_embeddings_adam_w/ -save_prefix pso_
python lohr_fairness_gazebase.py  -use_trial_types "['RAN']" -number_train 100  -num_folds 10 -GPU 1 -feature PSO -optimizer adam_w -save_dir saved_lohr_embeddings_adam_w/ -save_prefix pso_
python lohr_fairness_gazebase.py  -use_trial_types "['VD1','VD2']" -number_train 100  -num_folds 10 -GPU 1 -feature PSO -optimizer adam_w -save_dir saved_lohr_embeddings_adam_w/ -save_prefix pso_
