import csv
import os

import numpy as np
from sklearn import metrics


class FinalScore:
    def __init__(self, new_folder):
        self.new_folder = new_folder

    def odir_metrics(self, gt_data, pr_data):
        th = 0.5
        gt = gt_data.flatten()
        pr = pr_data.flatten()
        kappa = metrics.cohen_kappa_score(gt, pr > th)
        f1 = metrics.f1_score(gt, pr > th, average="micro")
        auc = metrics.roc_auc_score(gt, pr)
        final_score = (kappa + f1 + auc) / 3.0
        return kappa, f1, auc, final_score

    def import_data(self, filepath):
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            pr_data = [[int(row[0])] + list(map(float, row[1:])) for row in reader]
        pr_data = np.array(pr_data)
        return pr_data

    def output(self):
        gt_data = self.import_data(
            os.path.join(self.new_folder, "odir_ground_truth.csv")
        )
        pr_data = self.import_data(
            os.path.join(self.new_folder, "odir_predictions.csv")
        )
        kappa, f1, auc, final_score = self.odir_metrics(gt_data[:, 1:], pr_data[:, 1:])
        print("Kappa score:", kappa)
        print("F-1 score:", f1)
        print("AUC value:", auc)
        print("Final Score:", final_score)
