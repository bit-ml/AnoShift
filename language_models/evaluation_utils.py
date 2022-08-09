import torch
import numpy as np
import random
import math

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def anomaly_score_bert(model, text, tokenizer, num_repeats=15):
    batch_size = text.shape[0]
    mask_percent = 0.15

    anomaly_score_batch = [[] for i in range(batch_size)]

    pad_token = tokenizer.pad_token_id
    mask_token = tokenizer.mask_token_id

    for trial in range(num_repeats):
        masked_text_pos = text.copy()
        masks = []

        for batch_idx in range(batch_size):
            pad_pos = np.where(
                masked_text_pos[batch_idx, :] == pad_token)[0]

            if len(pad_pos) == 0:
                seq_len = len(masked_text_pos[batch_idx, :])
            else:
                seq_len = pad_pos[0] + 1

            num_masks = math.ceil(seq_len * mask_percent)
            mask_token_ids = random.sample(range(seq_len), num_masks)

            for pos_id in mask_token_ids:
                masked_text_pos[batch_idx, pos_id] = mask_token

            masks.append(mask_token_ids)

        output = model(torch.tensor(masked_text_pos).to(device))
        out_logits = output.logits

        y_pred = torch.nn.functional.softmax(
            out_logits, dim=2).cpu().detach().numpy()

        for k in range(batch_size):
            mask_token_ids = masks[k]
            for pos_id in mask_token_ids:
                anomaly_score_batch[k] += [y_pred[k]
                                           [pos_id][text[k, pos_id]], ]

    return [1 - (sum(n) / len(n)) for n in anomaly_score_batch]


def eval_rocauc_ds(model, ds_test_name, ds_test, bs_eval, tokenizer):
    class_scores = {cls: 0 for cls in ["inlier", "outlier"]}

    y_scores = []
    y_true = []
    y_scores_ids = {}

    print(ds_test)
    for cls in ds_test.keys():
        ds_cls = ds_test[cls]
        num_steps = max(ds_cls.shape[0] // bs_eval - 1, 1)

        for step in tqdm(range(num_steps)):
            examples = ds_cls[step * bs_eval: (step + 1) * bs_eval]
            num_examples = len(examples["input_ids"])

            if "count" in examples.keys():
                occurrences = np.array([1, ] * num_examples)
            else:
                occurrences = np.array([1, ] * num_examples)

            anomaly_score_lines = anomaly_score_bert(
                model, text=np.array(examples["input_ids"]), tokenizer=tokenizer
            )

            if cls == "inlier":
                true_label = 0
            else:
                true_label = 1

            y_true += [
                true_label,
            ] * occurrences.sum()

            for i in range(num_examples):
                y_scores += [anomaly_score_lines[i], ] * occurrences[i]

            anomaly_score_lines = np.array(anomaly_score_lines)
            class_scores[cls] += (occurrences *
                                  anomaly_score_lines).sum() / occurrences.sum()

        class_scores[cls] /= num_steps
        print("Class: {} Anomaly score: {}".format(cls, class_scores[cls]))
    return y_true, y_scores, y_scores_ids, class_scores


def eval_rocauc(model, dss_test, bs_eval, tb_writer, tokenizer, epoch):
    model.eval()
    with torch.no_grad():
        for (test_part_name, ds_test_part) in dss_test:
            y_true, y_scores, y_scores_ids, class_scores = eval_rocauc_ds(
                model, test_part_name, ds_test_part, bs_eval=bs_eval, tokenizer=tokenizer,
            )

            roc_auc = roc_auc_score(y_true, y_scores)

            precision_inlier, recall_inlier, _ = precision_recall_curve(
                y_true, [1 - x for x in y_scores], pos_label=0)
            precision_outlier, recall_outlier, _ = precision_recall_curve(
                y_true, y_scores, pos_label=1)

            # Use AUC function to calculate the area under the curve of precision recall curve
            auc_pr_inlier = auc(recall_inlier, precision_inlier)
            auc_pr_outlier = auc(recall_outlier, precision_outlier)

            f1_scores_inlier = (
                2 * (precision_inlier * recall_inlier) /
                (precision_inlier + recall_inlier)
            )
            f1_scores_outlier = (
                2
                * (precision_outlier * recall_outlier)
                / (precision_outlier + recall_outlier)
            )

            print(f"ROC AUC       {test_part_name}: {roc_auc}")
            print(f"AUCPR INLIER  {test_part_name}: {auc_pr_inlier}")
            print(f"AUCPR OUTLIER {test_part_name}: {auc_pr_outlier}")

            print(f"F1 INLIER {test_part_name}: {np.nanmax(f1_scores_inlier)}")
            print(
                f"F1 OUTLIER {test_part_name}: {np.nanmax(f1_scores_outlier)}")

            if tb_writer is not None:
                with open("exp_results.txt", "a+") as f:
                    f.write("\n\n\n")
                    f.write("ROCAUC " + str(test_part_name) +
                            " " + str(roc_auc) + "\n")
                    f.write("AUCPR Inlier " + str(test_part_name) +
                            " " + str(auc_pr_inlier) + "\n")
                    f.write("AUCPR Outlier " + str(test_part_name) +
                            " " + str(auc_pr_outlier) + "\n")
                    f.write("F1 Inlier " + str(test_part_name) +
                            " " + str(np.nanmax(f1_scores_inlier)) + "\n")
                    f.write("F1 Outlier " + str(test_part_name) +
                            " " + str(np.nanmax(f1_scores_outlier)) + "\n")

                for cls in class_scores.keys():
                    tb_writer.add_scalar(
                        f"cls_scores {cls}", class_scores[cls], epoch)

                tb_writer.add_scalar("eval/ROC_AUC", roc_auc, epoch)
                tb_writer.add_scalar("eval/AUCPR_INLIER", auc_pr_inlier, epoch)
                tb_writer.add_scalar("eval/AUCPR_OUTLIER",
                                     auc_pr_outlier, epoch)
                tb_writer.add_scalar(
                    "eval/F1_INLIER", np.nanmax(f1_scores_inlier), epoch)
                tb_writer.add_scalar(
                    "eval/F1_OUTLIER", np.nanmax(f1_scores_outlier), epoch)

    model.train()
