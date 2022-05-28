import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.stats import ttest_ind
from sklearn import metrics
from tqdm import tqdm

threshold_str = {
    0.1: '$10^{-1}$',
    0.01: '$10^{-2}$',
    0.001: '$10^{-3}$',
}


def avg_fnr_fpr_curve(
    fprs, tprs, label, plot_random=False,
    title=None, plot_statistics=False,
    loc='best', plot_legend=True,
    plot_points=10000, ncol=1,
    bbox_to_anchor=None,
    starting_point=None,
    fontsize=14, xscale=None,
    setting='verification',
):
    """
    Plot average roc curve from multiple fpr and tpr arrays of multiple cv-folds

    :param fprs: list of fpr arrays for the different folds
    :param tprs: list of tpr arrays for the different folds
    :label: name for the legend
    :plot_random: indicator, indicating if the random guessing curve should be plotted
    :title: title of plot; no title if 'None'
    :plot_statistics: if True, statistics for all the folds are plotted
    :loc: location of legend
    :plot_legend: if True legend is plotted
    :plot_points: number of points to plot
    :ncol: number of columns for legend
    :bbox_to_anchor: bounding box for legend outside of plot
    :starting_point: indicates the starting point of drawing the curves
    :fontsize: fontsize
    :xscale: scale for x-axis
    :setting: verification or identification
    """
    if xscale is not None:
        plt.xscale(xscale)

    tprs_list = []
    aucs = []
    for i in range(0, len(fprs)):
        fpr = fprs[i]
        fnr = 1 - tprs[i]

        tprs_list.append(interpolate.interp1d(fpr, fnr))
        aucs.append(metrics.auc(fprs[i], tprs[i]))
    aucs = np.array(aucs)
    x = np.linspace(0, 1, plot_points)
    if starting_point is not None:
        x = x[x > starting_point]

    if plot_random:
        y = 1 - x
        plt.plot(
            x, y, color='grey', linestyle='dashed',
            label='random guessing',
        )

    # plot average and std error of those roc curves:
    ys = np.vstack([f(x) for f in tprs_list])
    ys_mean = ys.mean(axis=0)
    ys_std = ys.std(axis=0) / np.sqrt(len(fprs))
    cur_label = label
    if plot_statistics:
        cur_label += r' (AUC={} $\pm$ {})'.format(
            np.round(np.mean(aucs), 4),
            np.round(np.std(aucs), 4),
        )
    plt.plot(x, ys_mean, label=cur_label)
    plt.fill_between(x, ys_mean - ys_std, ys_mean + ys_std, alpha=0.2)
    if plot_legend:
        if bbox_to_anchor is None:
            plt.legend(loc=loc, ncol=ncol, fontsize=fontsize)
        else:
            plt.legend(
                loc=loc, ncol=ncol,
                bbox_to_anchor=bbox_to_anchor, fontsize=fontsize,
            )

    if setting == 'verification':
        plt.xlabel('FMR', fontsize=fontsize)
        plt.ylabel('FNMR', fontsize=fontsize)
    elif setting == 'identification':
        plt.xlabel('FPIR', fontsize=fontsize)
        plt.ylabel('FNIR', fontsize=fontsize)

    plt.grid('on')
    if title is not None:
        plt.title(title)

    return aucs


def get_metric_dict_for_setting(
    inspect_key='normalized Ethnicity',
    inspect_list=[
        'Caucasian',
        'Black', 'Asian', 'Hispanic',
    ],
    window_size=10,
    num_folds=10,
    use_trial_types=['TEX'],
    used_train_key='random',
    used_train_percentage='1',
    used_train_seconds_per_user=80,
    data_dir='saved_score_dicts/',
    demo_dict=dict(),
    model='deepeye',
):
    metric_dict = dict()
    for fold_nr in tqdm(np.arange(num_folds)):
        if model == 'deepeye':
            cur_data_path = data_dir + 'key_' + used_train_key +\
                '_trials_' + str(use_trial_types).replace(',', '\',_\'') + '_fold_' +\
                str(fold_nr) + '_percentage_' + str(used_train_percentage) +\
                '_seconds_per_user_' + \
                str(used_train_seconds_per_user) + '.joblib'
        elif model == 'lohr':
            cur_data_path = data_dir + used_train_key +\
                '_trials_' + str(use_trial_types).replace(',', '\',_\'') + '_fold_' +\
                str(fold_nr) + '_percentage_' + str(used_train_percentage) +\
                '.joblib'
        else:
            print('not implemented')
            return None
        if not os.path.exists(cur_data_path):
            print(cur_data_path + ' does not exist')
            continue
        cur_data = joblib.load(cur_data_path)
        score_dicts = cur_data['score_dicts']
        label_dicts = cur_data['label_dicts']
        person_one_dicts = cur_data['person_one_dicts']
        person_two_dicts = cur_data['person_two_dicts']

        person_one_list = person_one_dicts[str(window_size)]
        person_two_list = person_two_dicts[str(window_size)]
        person_one_label = []
        for i in range(len(person_one_list)):
            person_one_label.append(demo_dict[inspect_key][person_one_list[i]])

        person_two_label = []
        for i in range(len(person_two_list)):
            person_two_label.append(demo_dict[inspect_key][person_two_list[i]])

        # all persons
        fpr, tpr, thresholds = metrics.roc_curve(
            label_dicts[str(window_size)], score_dicts[
                str(
                    window_size,
                )
            ], pos_label=1,
        )
        if 'all' not in metric_dict:
            metric_dict['all'] = {
                'fprs': [],
                'tprs': [],
                'thresholds': [],
            }
        metric_dict['all']['fprs'].append(fpr)
        metric_dict['all']['tprs'].append(tpr)
        metric_dict['all']['thresholds'].append(thresholds)

        # all combinations
        for key_1 in inspect_list:
            for key_2 in inspect_list:
                use_ids = []
                for i in range(len(person_one_label)):
                    # all instances matching the criterion
                    if person_one_label[i] == key_1 and person_two_label[i] == key_2:
                        use_ids.append(i)
                    # all matches; matching the criterion
                    if person_one_label[i] == key_1 and label_dicts[str(window_size)][i] == 1:
                        use_ids.append(i)
                use_ids = list(set(use_ids))
                fpr, tpr, thresholds = metrics.roc_curve(
                    np.array(label_dicts[str(window_size)])[use_ids],
                    np.array(score_dicts[str(window_size)])[use_ids],
                    pos_label=1,
                )
                cur_key = key_1 + ' - ' + str(key_2)
                if cur_key not in metric_dict:
                    metric_dict[cur_key] = {
                        'fprs': [],
                        'tprs': [],
                        'thresholds': [],
                    }
                metric_dict[cur_key]['fprs'].append(fpr)
                metric_dict[cur_key]['tprs'].append(tpr)
                metric_dict[cur_key]['thresholds'].append(thresholds)
    return metric_dict


def get_metric_dict_for_setting_lohr(
    inspect_key='normalized Ethnicity',
    inspect_list=[
        'Caucasian', 'Black', 'Asian', 'Hispanic',
    ],
    window_size='all',
    num_folds=10,
    use_trial_types=['TEX'],
    used_train_key='random',
    data_dir='saved_lohr_score_dicts/',
    demo_dict=dict(),
    prefix='',
):
    metric_dict = dict()
    for fold_nr in tqdm(np.arange(num_folds)):
        cur_data_path = data_dir + prefix + 'key_' + used_train_key +\
            '_trials_' + str(use_trial_types).replace(',', '\',_\'') + '_fold_' +\
            str(fold_nr) + '.joblib'
        if not os.path.exists(cur_data_path):
            print(cur_data_path + ' does not exist')
            continue

        cur_data = joblib.load(cur_data_path)

        scores = np.array(cur_data['score_dicts'][window_size])
        label = np.array(cur_data['label_dicts'][window_size])
        person_one_list = np.array(cur_data['person_one_dicts'][window_size])
        person_two_list = np.array(cur_data['person_two_dicts'][window_size])

        # delete nans
        ids_nan = np.isnan(scores)
        use_ids = np.where(ids_nan == False)[0]
        scores = scores[use_ids]
        label = label[use_ids]
        person_one_list = person_one_list[use_ids]
        person_two_list = person_two_list[use_ids]

        if len(np.unique(scores)) == 1 or np.sum(np.isnan(scores)) > 0:
            print(
                'en(np.unique(scores)): ' +
                str(len(np.unique(scores))),
            )
            print(
                'np.sum(np.isnan(scores)): ' +
                str(np.sum(np.isnan(scores))),
            )
            continue

        person_one_label = []
        for i in range(len(person_one_list)):
            person_one_label.append(demo_dict[inspect_key][person_one_list[i]])

        person_two_label = []
        for i in range(len(person_two_list)):
            person_two_label.append(demo_dict[inspect_key][person_two_list[i]])

        # all persons
        fpr, tpr, thresholds = metrics.roc_curve(label, scores, pos_label=1)
        if 'all' not in metric_dict:
            metric_dict['all'] = {
                'fprs': [],
                'tprs': [],
                'thresholds': [],
            }
        metric_dict['all']['fprs'].append(fpr)
        metric_dict['all']['tprs'].append(tpr)
        metric_dict['all']['thresholds'].append(thresholds)

        # all combinations
        for key_1 in inspect_list:
            for key_2 in inspect_list:
                use_ids = []
                for i in range(len(person_one_label)):
                    # all instances matching the criterion
                    if person_one_label[i] == key_1 and person_two_label[i] == key_2:
                        use_ids.append(i)
                    # all matches; matching the criterion
                    if person_one_label[i] == key_1 and label[i] == 1:
                        use_ids.append(i)
                use_ids = list(set(use_ids))
                fpr, tpr, thresholds = metrics.roc_curve(
                    np.array(label)[use_ids],
                    np.array(scores)[use_ids],
                    pos_label=1,
                )
                cur_key = key_1 + ' - ' + str(key_2)
                if cur_key not in metric_dict:
                    metric_dict[cur_key] = {
                        'fprs': [],
                        'tprs': [],
                        'thresholds': [],
                    }
                metric_dict[cur_key]['fprs'].append(fpr)
                metric_dict[cur_key]['tprs'].append(tpr)
                metric_dict[cur_key]['thresholds'].append(thresholds)
    return metric_dict


def get_fdr_metric_dict(
    metric_dict,
    inspect_thresholds=[0.1, 0.01, 0.001],
    plot_points=5000,
    fdr_alpha=0.5,
):
    try:
        threshold_lists = metric_dict['all']['thresholds']
        fprs_lists = metric_dict['all']['fprs']
    except KeyError:
        return dict(), dict()

    fdr_metric_dict = dict()
    fdr_dict = dict()

    for i in range(len(threshold_lists)):
        cur_t_list = threshold_lists[i]
        cur_fmr = fprs_lists[i]
        for j in range(len(inspect_thresholds)):
            cur_th = inspect_thresholds[j]
            if cur_th not in fdr_metric_dict:
                fdr_metric_dict[cur_th] = dict()
            if cur_th not in fdr_dict:
                fdr_dict[cur_th] = []
            cur_min_id = np.argmin(np.abs(cur_fmr - cur_th))
            cur_use_th = cur_t_list[cur_min_id]
            fmr_equal_list = []
            fnmr_equal_list = []
            for legend_key in metric_dict:
                if legend_key == 'all':
                    continue
                if legend_key not in fdr_metric_dict[cur_th]:
                    fdr_metric_dict[cur_th][legend_key] = {
                        'fmr': [], 'fnmr': [], 'eer': [],
                    }
                key_fmr = metric_dict[legend_key]['fprs'][i]
                key_fnmr = 1 - metric_dict[legend_key]['tprs'][i]
                key_ths = metric_dict[legend_key]['thresholds'][i]
                cur_min_th_id = np.argmin(np.abs(key_ths - cur_use_th))
                cur_fnmr_key = key_fnmr[cur_min_th_id]
                cur_fmr_key = key_fmr[cur_min_th_id]
                fdr_metric_dict[cur_th][legend_key]['fmr'].append(cur_fmr_key)
                fdr_metric_dict[cur_th][legend_key]['fnmr'].append(
                    cur_fnmr_key,
                )
                # eer
                cur_inter = interpolate.interp1d(key_fmr, key_fnmr)
                fprs_inter = np.linspace(0, 1, plot_points)
                fnrs_inter = cur_inter(fprs_inter)
                cur_eer = fprs_inter[
                    np.nanargmin(
                        np.absolute(fnrs_inter - fprs_inter),
                    )
                ]
                fdr_metric_dict[cur_th][legend_key]['eer'].append(cur_eer)
                legend_split = legend_key.split(' - ')
                if legend_split[0] == legend_split[1]:
                    fmr_equal_list.append(cur_fmr_key)
                    fnmr_equal_list.append(cur_fnmr_key)
            max_fmr_difference = np.max(
                fmr_equal_list,
            ) - np.min(fmr_equal_list)
            max_fnmr_difference = np.max(
                fnmr_equal_list,
            ) - np.min(fnmr_equal_list)
            fdr_dict[cur_th].append(
                1 - ((fdr_alpha * max_fmr_difference) +
                     ((1-fdr_alpha) * max_fnmr_difference)),
            )
    return fdr_metric_dict, fdr_dict


def print_fmr_fnmr_table(
    metric_dict,
    fdr_metric_dict, fdr_dict,
    inspect_thresholds=[0.1, 0.01, 0.001],
    decimals=2,
    print_eer=False,
    print_fdr=False,
    print_roc_auc=False,
    compare_key=None,
    p_value=0.05,
):

    header_str = '\\begin{tabular}{l||' + ''.join(['r' for a in range(len(inspect_thresholds))]) +\
        '|' + ''.join(['r' for a in range(len(inspect_thresholds))])
    if print_eer:
        header_str += '||c'
    if print_roc_auc:
        header_str += '||c'
    header_str += '} \n'
    header_str += '\\toprule \n$	\\tau = FMR_{10^x}$'
    for th in inspect_thresholds:
        header_str += ' & ' + threshold_str[th]
    for th in inspect_thresholds:
        header_str += ' & ' + threshold_str[th]
    if print_eer:
        header_str += ' & '
    if print_roc_auc:
        header_str += ' & '
    header_str += '\\\\ \\hline \nDemographics  (e-p)'
    header_str += r'& \multicolumn{' + \
        str(len(inspect_thresholds)) + '}{c|}{$FMR_x(\\tau)$}'
    header_str += r'& \multicolumn{' + str(len(inspect_thresholds)) + '}{c'
    header_str += '||}{$FNMR_x(\\tau)$}'
    if print_eer:
        header_str += '& $EER$ '
    if print_roc_auc:
        header_str += '& $ROC_{AUC}$ '
    header_str += '\\\\ \\hline \n'
    print(header_str)
    for key in fdr_metric_dict[inspect_thresholds[0]]:
        cur_table_line = key
        # FMR
        for th in inspect_thresholds:
            cur_table_line += ' & ' + str(np.round(np.mean(fdr_metric_dict[th][key]['fmr']), decimals=decimals)) +\
                r' $ \pm$ ' + \
                str(
                    np.round(
                        np.std(fdr_metric_dict[th][key]['fmr']), decimals=decimals,
                    ),
                )
        # FNMR
        for th in inspect_thresholds:
            cur_table_line += ' & ' + str(np.round(np.mean(fdr_metric_dict[th][key]['fnmr']), decimals=decimals)) +\
                r' $ \pm$ ' + \
                str(
                    np.round(
                        np.std(fdr_metric_dict[th][key]['fnmr']), decimals=decimals,
                    ),
                )
        if print_eer:
            # EER
            cur_table_line += ' & ' + str(np.round(np.mean(fdr_metric_dict[th][key]['eer']), decimals=decimals)) +\
                r' $ \pm$ ' + \
                str(
                    np.round(
                        np.std(fdr_metric_dict[th][key]['eer']), decimals=decimals,
                    ),
                )
            if compare_key is not None:
                cur_eer = fdr_metric_dict[th][key]['eer']
                key_eer = fdr_metric_dict[th][compare_key]['eer']
                if np.mean(cur_eer) > np.mean(key_eer):
                    tt_test_results = ttest_ind(cur_eer, key_eer)

                    cur_p_value = tt_test_results[1]
                    if cur_p_value < p_value:
                        extra = '*'
                    else:
                        extra = ''
                    cur_table_line += extra
        if print_roc_auc:
            fprs = metric_dict[key]['fprs']
            tprs = metric_dict[key]['tprs']
            aucs = []
            for i in range(len(fprs)):
                aucs.append(metrics.auc(fprs[i], tprs[i]))
            cur_table_line += ' & ' + str(np.round(np.mean(aucs), decimals=decimals)) +\
                r' $ \pm$ ' + str(np.round(np.std(aucs), decimals=decimals))
        cur_table_line += '\\\\'
        print(cur_table_line)

    if print_fdr:
        cur_table_line = '$FDR(\\tau)$'
        for th in inspect_thresholds:
            cur_table_line += ' & ' + str(np.round(np.mean(fdr_dict[th]), decimals=decimals)) +\
                r' $ \pm$ ' + \
                str(np.round(np.std(fdr_dict[th]), decimals=decimals))
        print(cur_table_line)
    print('\\bottomrule\n\\end{tabular}')


def print_fdr_table(
    metric_dict,
    trial_fdr_metric_dict, trial_fdr_dict,
    inspect_thresholds=[0.1, 0.01, 0.001],
    use_trial_type_list=[
        ['TEX'],
        ['BLG'],
        ['FXS'],
        ['HSS'],
        ['RAN'],
        ['VD1,VD2'],
    ],
    decimals=2,
    plot_points=5000,
    plot_intersectedAUC=True,
    plot_fdrAUC=True,
    fdr_thresholds=[0.1, 0.05, 0.01, 0.005, 0.001],
    fdr_alpha=0.5,
):
    header_str = '\\begin{tabular}{l||' + \
        ''.join(['r' for a in range(len(inspect_thresholds))])
    if plot_intersectedAUC:
        header_str += '||r'
    if plot_fdrAUC:
        header_str += '||r'
    header_str += '}\n'
    header_str += '\\toprule \n$	\\tau = FMR_{10^x}$'
    for th in inspect_thresholds:
        header_str += ' & ' + threshold_str[th]
    if plot_intersectedAUC:
        header_str += ' & '
    if plot_fdrAUC:
        header_str += ' & '
    header_str += '\\\\ \\hline \n Task'
    header_str += r'& \multicolumn{' + \
        str(len(inspect_thresholds)) + '}{c||}{$FDR(\\tau)$}'
    if plot_intersectedAUC:
        header_str += ' & $iAUC$'
    if plot_fdrAUC:
        header_str += ' & $FDR_{AUC}$'
    header_str += ' \\\\ \\hline'
    print(header_str)
    for use_trial_types in use_trial_type_list:
        trial_str = str(use_trial_types)
        try:
            fdr_dict = trial_fdr_dict[trial_str]
        except KeyError:
            continue
        print_name = trial_str.replace('[', '').replace(']', '')
        cur_table_line = print_name

        for th in inspect_thresholds:
            cur_table_line += ' & ' + str(np.round(np.mean(fdr_dict[th]), decimals=decimals)) +\
                r' $ \pm$ ' + \
                str(np.round(np.std(fdr_dict[th]), decimals=decimals))
        if plot_intersectedAUC:
            x = np.linspace(0, 1, plot_points)
            all_keys = list(metric_dict[trial_str].keys())
            ys_lists = []
            use_keys = []
            for key in all_keys:
                key_split = key.split(' - ')
                if key != 'all' and key_split[0] == key_split[1]:
                    fprs = metric_dict[str(use_trial_types)][key]['fprs']
                    tprs = metric_dict[str(use_trial_types)][key]['tprs']
                    tprs_list = []
                    for i in range(0, len(fprs)):
                        fpr = fprs[i]
                        tpr = tprs[i]

                        tprs_list.append(interpolate.interp1d(fpr, tpr))
                    ys = np.vstack([f(x) for f in tprs_list])
                    ys_lists.append(ys)
                    use_keys.append(key)
            tmp_inter_aucs = []
            for i in range(ys_lists[0].shape[0]):
                cur_vals = []
                for j in range(len(ys_lists)):
                    if j == 0:
                        cur_vals = np.reshape(
                            ys_lists[j][i], [len(ys_lists[j][i]), 1],
                        )
                    else:
                        cur_vals = np.hstack([
                            cur_vals, np.reshape(
                                ys_lists[j][i], [len(ys_lists[j][i]), 1],
                            ),
                        ])
                tmp_inter_aucs.append(metrics.auc(x, np.min(cur_vals, axis=1)))
            cur_table_line += ' & ' + str(np.round(np.mean(tmp_inter_aucs), decimals=decimals)) +\
                r' $ \pm$ ' + \
                str(np.round(np.std(tmp_inter_aucs), decimals=decimals))
        if plot_fdrAUC:
            x = np.linspace(0, 1, plot_points)
            all_keys = list(metric_dict[trial_str].keys())
            ys_lists = []
            use_keys = []
            b_fprs = metric_dict[str(use_trial_types)]['all']['fprs']
            b_ths = metric_dict[str(use_trial_types)]['all']['thresholds']
            use_full_keys = []
            for key in all_keys:
                key_split = key.split(' - ')
                if key != 'all' and key_split[0] == key_split[1]:
                    use_full_keys.append(key)

            fdr_aucs = []
            for i in range(len(b_fprs)):
                fdrs = []
                for cur_th in fdr_thresholds:
                    cur_min_id = np.argmin(np.abs(b_fprs[i] - cur_th))
                    cur_use_th = b_ths[i][cur_min_id]

                    fmr_equal_list = []
                    fnmr_equal_list = []
                    for use_full_key in use_full_keys:
                        c_fprs = metric_dict[
                            str(
                                use_trial_types,
                            )
                        ][use_full_key]['fprs']
                        c_tprs = metric_dict[
                            str(
                                use_trial_types,
                            )
                        ][use_full_key]['tprs']
                        c_ths = metric_dict[
                            str(
                                use_trial_types,
                            )
                        ][use_full_key]['thresholds']
                        cur_min_th_id = np.argmin(
                            np.abs(c_ths[i] - cur_use_th),
                        )
                        cur_fnmr_key = c_tprs[i][cur_min_th_id]
                        cur_fmr_key = c_fprs[i][cur_min_th_id]
                        fmr_equal_list.append(cur_fmr_key)
                        fnmr_equal_list.append(cur_fnmr_key)
                    max_fmr_difference = np.max(
                        fmr_equal_list,
                    ) - np.min(fmr_equal_list)
                    max_fnmr_difference = np.max(
                        fnmr_equal_list,
                    ) - np.min(fnmr_equal_list)
                    cur_fdr = (
                        1 - ((fdr_alpha * max_fmr_difference) +
                             ((1-fdr_alpha) * max_fnmr_difference))
                    )
                    fdrs.append(cur_fdr)
                fdrs = np.array(fdrs)
                min_th = np.min(fdr_thresholds)
                max_th = np.max(fdr_thresholds)
                use_thresholds = (
                    np.array(fdr_thresholds) -
                    min_th
                ) / (max_th - min_th)
                cur_fdr_auc = metrics.auc(use_thresholds, fdrs)
                fdr_aucs.append(cur_fdr_auc)
            cur_table_line += ' & ' + str(np.round(np.mean(fdr_aucs), decimals=decimals)) +\
                r' $ \pm$ ' + \
                str(np.round(np.std(fdr_aucs), decimals=decimals))
        cur_table_line += '\\\\'
        print(cur_table_line)
    print('\\bottomrule\n\\end{tabular}')


def plot_eer_percentage(
    fdr_metric_dict,
    inspect_thresholds=[0.1, 0.01, 0.001],
    x_label='percentage of persons that are Caucasian',
    y_label='EER',
    font_size=12,
):
    percentages = list(fdr_metric_dict.keys())

    keys = list(fdr_metric_dict[percentages[0]][inspect_thresholds[0]])
    for key in keys:
        eers = []
        stds = []
        for per in percentages:
            cur_eers = fdr_metric_dict[per][inspect_thresholds[0]][key]['eer']
            eers.append(np.mean(cur_eers))
            stds.append(np.std(cur_eers))
        plt.errorbar(percentages, eers, stds, label=key)
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    plt.legend(fontsize=font_size)


def plot_fdr_percentage(
    fdr_dict,
    inspect_thresholds=[0.1, 0.01, 0.001],
    x_label='percentage of persons that are Caucasian',
    y_label='FDR',
    font_size=12,
):

    legend_threshold_str = {
        0.1: '10^{-1}',
        0.01: '10^{-2}',
        0.001: '10^{-3}',
    }

    percentages = list(fdr_dict.keys())
    for th in inspect_thresholds:
        fdrs = []
        stds = []
        for per in percentages:
            cur_list = fdr_dict[per][th]
            fdrs.append(np.mean(cur_list))
            stds.append(np.std(cur_list))
        plt.errorbar(
            percentages, fdrs, stds,
            label='$\\tau = FMR_{' + legend_threshold_str[th] + '}$',
        )
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    plt.legend(fontsize=font_size)
