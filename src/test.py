# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions provided under the following terms:
# Copyright (c) UCLA-DM
# Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)
# (see LICENSE-THIRD-PARTY for full text)

from pairwise import (
    get_pairwise_data_loaders,
    gather_eval_pair_data,
    get_data_dict_by_gname,
    evaluate_ranking_performance,
    RANKING_METRICS,
)
from model import get_y_with_target

from config import FLAGS
from saver import saver
from utils import (
    _get_y_with_target,
    create_pred_dict,
    print_stats,
    create_loss_dict_get_target_list,
    update_loss_dict,
    deduce_load_model_path,
    get_num_graphs,
)
from data import split_dataset, torch_generator, get_num_designs

from model import feature_extract
from model_factory import create_model

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    max_error,
    mean_absolute_percentage_error,
    classification_report,
    confusion_matrix,
)
import numpy as np
import torch
from scipy.stats import rankdata, kendalltau
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import pandas as pd
from os.path import dirname, basename


def inference(dataset, task):
    saver.log_info(f"Inference begins")
    train_loader, val_loader, test_loader, transductive_test_loader = split_dataset(
        dataset
    )
    load_model = deduce_load_model_path(task, FLAGS)

    model = create_model(dataset, None, task, load_model=load_model)
    saver.log_model_architecture(model)

    loaded_model_info = None

    if load_model != None and load_model != "None":
        model, loaded_model_info = saver.load_trained_model(load_model, model)

    if FLAGS.feature_extract:
        feature_extract(model, "MLPs", FLAGS.fix_gnn_layer)
    # print(model)
    # saver.log_model_architecture(model)

    # li = [('train_loader', train_loader, False)] # 03/06/2024: Let's not look at train_loader to be quicker.

    li = []

    if transductive_test_loader is not None:
        li += [("transductive_test_loader", transductive_test_loader, False, False)]

    if len(test_loader) > 0:
        li += [("test_loader", test_loader, False, False)]

    if hasattr(FLAGS, "skip_pointwise_infer") and FLAGS.skip_pointwise_infer:
        li = []

    data_dict = None
    eval_pairwise_global = getattr(FLAGS, "eval_pairwise", False)
    if eval_pairwise_global:  # TODO: tune it
        data_dict = get_data_dict_by_gname(
            dataset
        )  # loads all graphs into memory for analysis
        # eval_pairwise = True
    if (
        FLAGS.pairwise_class
        # and 'pariwise_comparison' in FLAGS.pairwise_what_branches
        # and FLAGS.loss_components == "both"
    ):

        for (
            inference_time_pragma_differing_by
        ) in FLAGS.inference_time_pragma_differing_by_li:

            _, _, pairwise_test_loader, pairwise_transductive_test_loader = (
                get_pairwise_data_loaders(  # tricky: not really using the pairwise val loader
                    dataset,
                    torch_generator,
                    data_dict=data_dict,
                    pragma_differing_by=inference_time_pragma_differing_by,
                )
            )

            pairwise_test_loader.pragma_differing_by = (
                inference_time_pragma_differing_by
            )
            pairwise_transductive_test_loader.pragma_differing_by = (
                inference_time_pragma_differing_by
            )

            forward_pairwise = 'pariwise_comparison' in FLAGS.pairwise_what_branches

            if 'atefeh' in FLAGS.load_model:
                # Very tricky code below! If we are testing Atefeh's model,
                # we should only use the pointwise branch regardless of
                # whether the pairwise branch exists or not.
                forward_pairwise = False

            if len(pairwise_test_loader) > 0:
                li.append(
                    (
                        f"pairwise_test_loader_diff_{inference_time_pragma_differing_by}",
                        pairwise_test_loader,
                        forward_pairwise,
                        eval_pairwise_global,
                    )
                )
            else:
                saver.log_info(f'len(pairwise_test_loader)==0')
            if len(pairwise_transductive_test_loader) > 0:
                li.append(
                    (
                        f"pairwise_transductive_test_loader_diff_{inference_time_pragma_differing_by}",
                        pairwise_transductive_test_loader,
                        forward_pairwise,
                        eval_pairwise_global,
                    )
                )
            else:
                saver.log_info(f'len(pairwise_transductive_test_loader)==0')
            # 03/06/2024: Let's ignore pairwise_train_loader in inference.
            # Reason: Some model may not be trained in the same way as other models, e.g. the
            # exact training designs may not be consistent across models.
            # Just look at test kernels (inductive).

    final_print_dict = OrderedDict()
    for test_name, loader, forward_pairwise, eval_pairwise in li:
        saver.log_info(f"-" * 100)

        saver.log_info(f"*" * 50)
        saver.log_info(f"@@@ New testing section @@@")
        saver.log_info(f"{test_name} starts")
        saver.log_info(f"*" * 50)

        if task == "regression":
            csv_dict = {"header": ["gname", "pragma"]}
            testr, loss_dict, encode_loss, eval_df_dict = test(
                loader,
                "test",
                model,
                0,
                plot_test=True,
                csv_dict=csv_dict,
                data_dict=data_dict,
                forward_pairwise=forward_pairwise,
                eval_pairwise=eval_pairwise,
                test_name=test_name,
                task=task,
                dataset=dataset,
            )
            assert eval_df_dict["point"]["df"].iloc[-1]["target"] == "tot/avg"

            if not forward_pairwise and not eval_pairwise:
                rmse = eval_df_dict["point"]["df"].iloc[-1]["rmse"]
                final_print_dict[
                    f"{test_name}_point_rmse ({eval_df_dict['point']['support']})"
                ] = f"{rmse:.4f}"
            else:
                pass  # do not print the point rmse if this data loader is meant to be pairwise
            # Below code is tricky!
            # Basically, for pointwise regression, it is always possible to further evaluate some pairwise results.
            # For example, if each design --> perf prediction,
            # then we can evaluate the pairwise comparison accuracy for the perf target over all the relevant pairs
            # (e.g. pairs differing by 1 pragma)
            # We should gather the accuracy and print to final summary as well.
            if "pair" in eval_df_dict:
                final_print_dict[
                    f"{test_name}_pairwise_acc ({eval_df_dict['pair']['macro avg']['support']})"
                ] = f"{eval_df_dict['pair']['accuracy']:.4f}"
            if "rank" in eval_df_dict:
                for rm in RANKING_METRICS:
                    final_print_dict[
                        f"{test_name}_pairwise_to_rank-->{rm} ({eval_df_dict['rank']['support']:.4f})"
                    ] = f"{eval_df_dict['rank'][rm]:.4f}"
            saver.log_info(f"loss_dict={loss_dict}")
            saver.log_info(
                f"{test_name} loss: {testr:.7f}, encode loss: {encode_loss:.7f}"
            )
            # saver.log_dict_of_dicts_to_csv(f'{test_name} actual-prediction', csv_dict, csv_dict['header'])
        else:
            testr, loss_dict, encode_loss, eval_df_dict = test(
                loader,
                "test",
                model,
                0,
                forward_pairwise=False,
                eval_pairwise=False,
                task=task,
                dataset=dataset,
            )

            accuracy = eval_df_dict["point"]["accuracy"]
            final_print_dict[f"{test_name}_point_acc"] = f"{accuracy:.4f}"

            saver.log_info(("Test loss: {:.3f}".format(testr)))
        saver.log_info(f"-" * 100)
        saver.log_info("")
    saver.log_info(
        f"FLAGS.load_model=\n{FLAGS.load_model}\n{basename(dirname(FLAGS.load_model))}\n{loaded_model_info}"
    )

    ks = "\t".join(final_print_dict.keys())
    vs = "\t".join(final_print_dict.values())

    summary_str = f"{task}:"
    summary_str = saver.log_info(f"Summary of inference:", build_str=summary_str)
    summary_str = saver.log_info(f"{ks}", build_str=summary_str)
    summary_str = saver.log_info(f"{vs}", build_str=summary_str)
    saver.log_info_new_file(summary_str, "final_report.txt")

    if FLAGS.save_emb:
        saver.save_emb_save_to_disk(f"embeddings.pickle")

    saver.log_info(f"Inference done")


def get_true_perf(perf, util_LUT, util_FF, util_DSP, util_BRAM):
    return (
        perf
        * (util_LUT <= 0.8).float()
        * (util_FF <= 0.8).float()
        * (util_DSP <= 0.8).float()
        * (util_BRAM <= 0.8).float()
    )


@torch.no_grad()
def test(
    loader,
    tvt,
    model,
    epoch,
    plot_test=False,
    test_losses=[-1],
    csv_dict=None,
    data_dict=None,
    forward_pairwise=False,
    eval_pairwise=False,
    test_name=None,
    task=FLAGS.task,
    dataset=None,
):
    model.eval()

    inference_loss = 0
    correct, total = 0, 0
    total_true_perf = 0
    # i = 0

    loss_dict, target_list = create_loss_dict_get_target_list(FLAGS, task)

    points_dict = create_pred_dict(target_list)
    points_pred_by_gname = defaultdict(OrderedDict)
    pairs_pred_by_gname = defaultdict(OrderedDict)

    # while True:
    for test_iter, data in enumerate(tqdm(loader)):
        # for attr_name, attr_value in data.__dict__.items():
        #     if torch.is_tensor(attr_value):
        #         saver.log_info(f"Attribute: {attr_name}")
        #         saver.log_info(f"Device: {attr_value.device}")
        #         saver.log_info("---")

        if FLAGS.mode == "acc_launch":
            pass
        else:
            assert FLAGS.mode == "standalone"
            # model = model.to(FLAGS.device)
            data = data.to(FLAGS.device)
            saver.log_info_once(f'data.x.device={data.x.device}')

        # for attr_name, attr_value in data.__dict__.items():
        #     if torch.is_tensor(attr_value):
        #         saver.log_info(f"@Attribute: {attr_name}")
        #         saver.log_info(f"@Device: {attr_value.device}")
        #         saver.log_info("@---")

        # data = data.to(FLAGS.device)
        with torch.no_grad():  # TODO: double check this to ensure no problem
            out_dict, loss, loss_dict_, gae_loss = model(
                data,
                forward_pairwise=forward_pairwise,
                tvt=tvt,
                iter=test_iter,
                test_name=test_name,
            )

        if task == "regression":
            total += loss  # * get_num_graphs(
            # data
            # )  # .item() # TODO: check what happens for num_graphs if pairwise loader

            # true_perf = get_true_perf(
            #     get_y_with_target(data, "perf"),
            #     get_y_with_target(data, "util-LUT"),
            #     get_y_with_target(data, "util-FF"),
            #     get_y_with_target(data, "util-DSP"),
            #     get_y_with_target(data, "util-BRAM"),
            # )
            # pred_true_perf = get_true_perf(
            #     out_dict["perf"],
            #     out_dict["util-LUT"],
            #     out_dict["util-FF"],
            #     out_dict["util-DSP"],
            #     out_dict["util-BRAM"],
            # )
            # true_perf_loss = torch.mean(torch.abs(true_perf - pred_true_perf))
            # total_true_perf += true_perf_loss.item()
            if not FLAGS.SSL:
                loss_dict = update_loss_dict(
                    loss_dict, loss_dict_, target_list, FLAGS, data
                )
            pred = None
        else:
            loss, pred = torch.max(out_dict[FLAGS.target[0]], 1)
            labels = _get_y_with_target(data, FLAGS.target[0])
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        if not FLAGS.SSL:

            for target_name in target_list:
                out = _get_out_from_out_dict(out_dict, target_name, pred, task)
                for i in range(len(out)):
                    out_value = out[i].item()
                    if FLAGS.encode_log and target_name == "actual_perf":
                        out_value = 2 ** (out_value) * (1 / FLAGS.normalizer)

                    true_val = _get_y_with_target(data, target_name)[i].item()
                    points_dict[target_name]["pred"].append(out_value)
                    points_dict[target_name]["true"].append(true_val)

                if "inference" in FLAGS.subtask and eval_pairwise:
                    if forward_pairwise:
                        li = data.xy_dict_programl["point"]
                        assert type(li) is list and len(li) == 1
                        points_li = li[0]
                        assert len(points_li) % 2 == 0
                        sp = len(points_li) // 2
                        for i in range(sp):
                            key1, key2 = points_li[i], points_li[sp + i]

                            assert (
                                target_list[0] == "perf"
                            )  # only care about comparison for perf (but still looping through target_list for coding simplicity)
                            d = {
                                target_name: torch.argmax(
                                    _get_out_from_out_dict(
                                        out_dict,
                                        f"{target_list[0]}_pairwise_class",  # tricky code! always take perf
                                        pred,
                                        task,
                                    )[i]
                                ).item()
                                for target_name in target_list
                            }

                            pairs_pred_by_gname[data.gname[0][i]][(key1, key2)] = {
                                'comp_label': d,
                                'comp_logits': out_dict[
                                    f"{target_list[0]}_pairwise_class"
                                ][i].view(1, 2),
                            }
                    else:
                        # if not FLAGS.sequence_modeling:  # TODO: enable it
                        assert type(data.xy_dict_programl["point"]) is list
                        assert (
                            len(data.xy_dict_programl["point"]) == 1
                        ), 'Must be a double list'
                        assert type(data.gname) is list
                        assert len(data.gname) == 1, 'Must be a double list'
                        assert len(data.xy_dict_programl["point"][0]) == len(
                            data.gname[0]
                        )

                        for i, data_key in enumerate(
                            data.xy_dict_programl["point"][0]
                        ):
                            d = {
                                target_name: _get_out_from_out_dict(
                                    out_dict, target_name, pred, task
                                )[i].item()
                                for target_name in target_list
                            }

                            # d['emb_T'] = out_dict['emb_T'][i].detach().cpu().numpy() # TODO: enable it

                            # try:
                            points_pred_by_gname[data.gname[0][i]][data_key] = d
                            # except Exception as e:

    if (
        FLAGS.plot_pred_points
        and tvt == "test"
        and (plot_test or (test_losses and (total / len(loader)) < min(test_losses)))
    ):
        from utils import plot_points, plot_points_with_subplot

        saver.log_info(f"@@@ plot_pred_points {test_name}")
        if not FLAGS.multi_target:
            plot_points(
                {
                    f"{FLAGS.target[0]}-pred_points": points_dict[f"{FLAGS.target[0]}"][
                        "pred"
                    ],
                    f"{FLAGS.target[0]}-true_points": points_dict[f"{FLAGS.target[0]}"][
                        "true"
                    ],
                },
                f"epoch_{epoch + 1}_{tvt}_{test_name}",
                saver.plotdir,
            )
            print(f"done plotting with {correct} corrects out of {total}")
        else:
            assert isinstance(FLAGS.target, list)
            plot_points_with_subplot(
                points_dict,
                f"epoch_{epoch + 1}_{tvt}_{test_name}",
                saver.plotdir,
                target_list,
            )

    eval_df_dict = {}
    eval_df = {}
    # if FLAGS.subtask in ["inference", "train", "inference_dse"]:
    if FLAGS.subtask in ["inference", "inference_dse"]:
        if task == "regression":
            eval_df = _report_rmse_etc(points_dict, f"epoch {epoch}:", True)
            eval_df_dict["point"] = eval_df
            if eval_pairwise:
                saver.log_info(f"len(points_pred_by_gname)={len(points_pred_by_gname)}")
                tot_points = 0
                for gname, d in points_pred_by_gname.items():
                    saver.log_info(f"\t{gname}: {len(d)} points", silent=True)
                    tot_points += len(d)
                saver.log_info(f"{tot_points} points in total")
                saver.log_info(f"len(pairs_pred_by_gname)={len(pairs_pred_by_gname)}")

                for gname, d in pairs_pred_by_gname.items():
                    saver.log_info(f"\t{gname}: {len(d)} pairs")

                assert target_list[0] == "perf"
                pair_dict, pred_dict_by_target_global = gather_eval_pair_data(
                    data_dict,
                    points_pred_by_gname,
                    pairs_pred_by_gname,
                    [target_list[0]],  # tricky: just need to compare perf!
                    test_name,
                    pragma_differing_by=getattr(loader, 'pragma_differing_by', 1),
                )
                # if points_pred_by_gname is not None:[]
                eval_df_class = _report_pairwise_class_result(
                    pred_dict_by_target_global,
                    f"{len(data_dict)}_pairwise_kernels_pred_dict_by_target_global:",
                )
                eval_df_dict["pair"] = eval_df_class

                if loader.pragma_differing_by == 'all':
                    eval_df_ranking = evaluate_ranking_performance(
                        data_dict, points_pred_by_gname, pairs_pred_by_gname
                    )

                    saver.log_info(f'eval_df_ranking={eval_df_ranking}')
                    eval_df_dict["rank"] = eval_df_ranking

                    # exit()

        elif task == "class":
            eval_df = report_class_loss(points_dict)
            eval_df_dict["point"] = eval_df
        else:
            raise NotImplementedError()

    if task == "regression":
        if "inference" in FLAGS.subtask:
            # if len(loader) == 0:
            #     target_dict = {}
            #     total_avg = total
            #     inference_loss_avg = inference_loss
            # else:
            assert len(loader) > 0, f'Check HARP_setup parameter in config.py'
            target_dict = {key: v / len(loader) for key, v in loss_dict.items()}
            # target_dict['true_perf'] = (
            #     total_true_perf / len(loader)
            # )
            total_avg = total / len(loader)
            inference_loss_avg = inference_loss / len(loader)  # / FLAGS.batch_size

            rtn = (
                total_avg,
                target_dict,
                inference_loss_avg,
                eval_df_dict,
            )
        else:
            target_dict = {key: v / len(loader) for key, v in loss_dict.items()}
            # target_dict["true_perf"] = (
            #     total_true_perf / len(loader)
            # )
            rtn = (
                total / len(loader),
                target_dict,
                gae_loss.item(),
                eval_df_dict,
            )
    else:
        rtn = (
            1 - correct / total,
            {key: v / len(loader) for key, v in loss_dict.items()},
            gae_loss.item(),
            eval_df_dict,
        )
    # saver.log_info(f'len(loader)={len(loader)}')
    return rtn


def _report_pairwise_class_result(points_dict, label):
    saver.log_info(label)
    for target_name, d in points_dict.items():

        # saver.log_info(f'points_dict.keys = {points_dict.keys()}')
        # exit()

        labels = d["true"]
        pred = d["pred"]
        target_names = ["d1 <= d2", "d1 > d2"]
        assert len(labels) == len(pred)
        saver.log_info(
            f"-----\n{target_name} classification report {label} ({len(labels)} data points)"
        )

        try:
            saver.log_info(
                classification_report(labels, pred, target_names=target_names, digits=4)
            )
            # cm = confusion_matrix(labels, pred, labels=[0, 1])
            # saver.log_info(f'Confusion matrix:\n{cm}')
        except ValueError as e:

            saver.log_info(
                f"ValueError encountered in classification_report!\n{e}\nset(labels)={set(labels)}\npoints_dict={points_dict}"
            )
            if not FLAGS.DEBUG:
                raise e  # quite critical!
            else:
                saver.log_info(
                    f"Since it is debugging mode, ignore this error (most likely due to too few data points thus some label does not exist)"
                )

        emb_diff_li = d.get("emb_diff")
        # print('@@@@@'*100)
        # print('emb_diff_li is', emb_diff_li)
        if emb_diff_li is not None:
            assert type(emb_diff_li) is list
            print_stats(emb_diff_li, "emb_diff_li", saver=saver)
        saver.log_info(f"\n----")
        try:
            rtn = _get_class_report_as_dict(labels, pred, target_names, None)
        except ValueError as e:
            saver.log_info(f"ValueError encountered in classification_report!\n{e}")
            rtn = {
                "accuracy": -float("inf")
            }  # to be safe, do -inf instead 0,0 -- to indicate something wrong with data
        return rtn


def report_class_loss(points_dict):
    d = points_dict[FLAGS.target[0]]
    labels = d["true"]
    pred = d["pred"]
    target_names = ["invalid", "valid"]
    saver.info("classification report")
    if len(labels) == 0:
        raise RuntimeError(f"len(labels) == 0")
    labels_for_report = np.arange(0, len(set(labels)), 1)
    s = classification_report(
        labels, pred, target_names=target_names, digits=4, labels=labels_for_report
    )
    saver.log_info(s)
    # cm = confusion_matrix(labels, pred, labels=[0, 1])
    # saver.info(f'Confusion matrix:\n{cm}')
    rtn = _get_class_report_as_dict(labels, pred, target_names, labels_for_report)
    return rtn


def _get_class_report_as_dict(labels, pred, target_names, labels_for_report):
    rtn = classification_report(
        labels,
        pred,
        target_names=target_names,
        output_dict=True,
        labels=labels_for_report,
    )
    if "accuracy" not in rtn:
        accuracy = rtn["micro avg"]["f1-score"]
        # assert accuracy == 0.0, f'accuracy={accuracy}'
        rtn["accuracy"] = accuracy
    return rtn


def _report_rmse_etc(points_dict, label, print_result=True):
    if print_result:
        saver.log_info(label)
    data = defaultdict(list)
    tot_mape, tot_rmse, tot_mse, tot_mae, tot_max_err, tot_tau, tot_std = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    num_data = None
    try:
        for target_name, d in points_dict.items():
            # true_li = d['true']
            # pred_li = d['pred']
            true_li = d["true"]
            pred_li = d["pred"]
            num_data = len(true_li)
            mape = mean_absolute_percentage_error(true_li, pred_li)
            rmse = mean_squared_error(true_li, pred_li, squared=False)
            mse = mean_squared_error(true_li, pred_li, squared=True)
            mae = mean_absolute_error(true_li, pred_li)
            max_err = max_error(true_li, pred_li)

            true_rank = rankdata(true_li)
            pred_rank = rankdata(pred_li)
            tau = kendalltau(true_rank, pred_rank)[0]
            data["target"].append(target_name)
            data["mape"].append(mape)
            data["rmse"].append(rmse)
            data["mse"].append(mse)
            data["mae"].append(mae)
            data["max_err"].append(max_err)
            data["tau"].append(tau)

            # data['rmse'].append(f'{rmse:.4f}')
            # data['mse'].append(f'{mse:.4f}')
            # data['tau'].append(f'{tau: .4f}')
            tot_mape += mape
            tot_rmse += rmse
            tot_mse += mse
            tot_mae += mae
            tot_max_err += max_err
            tot_tau += tau

            pred_std = d.get("pred_std")
            if pred_std is not None:
                assert type(pred_std) is np.ndarray, f"{type(pred_std)}"
                pred_std = np.mean(pred_std)
                data["pred_std"].append(pred_std)
                tot_std += pred_std
        data["target"].append("tot/avg")
        data["mape"].append(tot_mape)
        data["rmse"].append(tot_rmse)
        data["mse"].append(tot_mse)
        data["mae"].append(tot_mae)
        data["max_err"].append(tot_max_err)
        data["tau"].append(tot_tau / len(points_dict))
        if "pred_std" in data:
            data["pred_std"].append(tot_std / len(points_dict))
    except ValueError as v:
        saver.log_info(f"Error {v}")
        data = defaultdict(list)

    # data['rmse'].append(f'{tot_rmse:.4f}')
    # data['mse'].append(f'{tot_mse:.4f}')
    # data['tau'].append(f'{tot_tau / len(points_dict):.4f}')
    df = pd.DataFrame(data)
    pd.set_option("display.max_columns", None)
    if print_result:
        saver.log_info(num_data)
        saver.log_info(df.round(4))
    # exit()
    return {"df": df, "support": num_data}
    # exit()


def _get_out_from_out_dict(out_dict, target_name, pred, task):
    if task == "class":
        out = pred
    elif FLAGS.encode_log and "perf" in target_name:
        out = out_dict["perf"]
    else:
        out = out_dict[target_name]
    return out


# def inference_loss_function(pred, true):
#     return (pred - true) ** 2
