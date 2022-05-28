python -u -c "from Evaluation import score_dict_creation as sdc; sdc.create_score_dicts_lohr('saved_lohr_embeddings/','saved_lohr_score_dicts/')"
python -u -c "from Evaluation import score_dict_creation as sdc; sdc.create_score_dicts_lohr('saved_lohr_embeddings_adam_w/','saved_lohr_score_dicts_adam_w/')"
python -u -c "from Evaluation import score_dict_creation as sdc; sdc.create_score_dicts_lohr_merged('saved_lohr_embeddings/','saved_lohr_score_dicts/')"
python -u -c "from Evaluation import score_dict_creation as sdc; sdc.create_score_dicts_lohr_merged('saved_lohr_embeddings_adam_w/','saved_lohr_score_dicts_adam_w/')"
