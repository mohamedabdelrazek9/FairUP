import numpy as np
import pandas as pd

class Fairness(object):
    """
    Compute fairness metrics
    """
    def __init__(self, G, test_nodes_idx, targets, predictions, sens_attr, neptune_run,
            multiclass_pred=False, multiclass_sens=False):

        self.multiclass_pred = multiclass_pred
        self.multiclass_sens = multiclass_sens
        self.sens_attr = sens_attr        
        self.neptune_run = neptune_run
        self.neptune_run["sens_attr"] = self.sens_attr
        self.G = G
        self.test_nodes_idx = test_nodes_idx.cpu().detach().numpy()
        self.true_y = np.asarray(targets) # target variables
        self.pred_y = np.asarray(predictions) # prediction of the classifier
        self.sens_attr_array = self.G.nodes["user"].data[self.sens_attr].cpu().detach().numpy() # sensitive attribute values
        self.sens_attr_values = self.sens_attr_array[self.test_nodes_idx]

        if self.multiclass_pred and self.multiclass_sens: # Classifier: multiclass - Sens.attr: multiclass
            self.class_range = list(set(self.true_y))
            self.y_hat = []
            self.yneq_hat = []
            for y_hat_idx in self.class_range:
                self.y_hat.append(self.pred_y == y_hat_idx)
                self.yneq_hat.append(self.pred_y != y_hat_idx)
                
            self.sens_attr_range = list(set(self.sens_attr_values))
            self.s = []
            for s_idx in self.sens_attr_range:
                self.s.append(self.sens_attr_values == s_idx)

            self.y_s = []
            self.yneq_s = []
            for y_idx in self.class_range:
                self.y_s.append([])
                self.yneq_s.append([])
                for s_idx in self.sens_attr_range:
                    self.y_s[y_idx].append(np.bitwise_and(self.true_y == y_idx, self.s[s_idx]))
                    self.yneq_s[y_idx].append(np.bitwise_and(self.true_y != y_idx, self.s[s_idx]))
            self.y_s = np.array(self.y_s)
            self.yneq_s = np.array(self.yneq_s)

        elif self.multiclass_sens: # Classifier: binary - Sens.attr: multiclass
            self.sens_attr_range = list(set(self.sens_attr_values))
            self.s = []
            self.y1_s = []
            self.y0_s = []
            for s_idx in self.sens_attr_range:
                self.s.append(self.sens_attr_values == s_idx)
                self.y1_s.append(np.bitwise_and(self.true_y == 1, self.s[s_idx]))
                self.y0_s.append(np.bitwise_and(self.true_y == 0, self.s[s_idx]))
        
        else: # Classifier: binary - Sens.attr: binary
            self.s0 = self.sens_attr_values == 0
            self.s1 = self.sens_attr_values == 1
            self.y1_s0 = np.bitwise_and(self.true_y == 1, self.s0)
            self.y1_s1 = np.bitwise_and(self.true_y == 1, self.s1)
            self.y0_s0 = np.bitwise_and(self.true_y == 0, self.s0)
            self.y0_s1 = np.bitwise_and(self.true_y == 0, self.s1)

    
    def statistical_parity(self):
        if self.multiclass_pred and self.multiclass_sens: # Classifier: multiclass - Sens.attr: multiclass
            """
            P(y^=0|s=0) = P(y^=0|s=1) = ... = P(y^=0|s=N)
            [...]
            P(y^=M|s=0) = P(y^=M|s=1) = ... = P(y^=M|s=N)
            """
            stat_parity = []
            for y_hat_idx in self.class_range:
                stat_parity.append([])
                for s_idx in self.sens_attr_range:
                    stat_parity[y_hat_idx].append(
                        sum(np.bitwise_and(self.y_hat[y_hat_idx], self.s[s_idx])) /
                        sum(self.s[s_idx])
                    )
                    self.neptune_run["fairness/SP_y^" + str(y_hat_idx) + "_s" + str(s_idx)] = stat_parity[y_hat_idx][s_idx]
        elif self.multiclass_sens: # Classifier: binary - Sens.attr: multiclass
            ''' P(y^=1|s=0) = P(y^=1|s=1) = ... = P(y^=1|s=N) '''
            stat_parity_s = []
            for s_idx in self.sens_attr_range:
                stat_parity_s.append(sum(self.pred_y[self.s[s_idx]]) / sum(self.s[s_idx]))
                self.neptune_run["fairness/SP_s" + str(s_idx)] = stat_parity_s[s_idx]
        else: # Classifier: binary - Sens.attr: binary
            ''' P(y^=1|s=0) = P(y^=1|s=1) '''
            # stat_parity = abs(sum(self.pred_y[self.s0]) / sum(self.s0) - sum(self.pred_y[self.s1]) / sum(self.s1))
            stat_parity_s0 = sum(self.pred_y[self.s0]) / sum(self.s0)
            stat_parity_s1 = sum(self.pred_y[self.s1]) / sum(self.s1)
            stat_parity_diff = stat_parity_s0 - stat_parity_s1
            self.neptune_run["fairness/SP_s0"] = stat_parity_s0
            self.neptune_run["fairness/SP_s1"] = stat_parity_s1
            
            print("Statistical Parity Difference (SPD): {:.4f}".format(np.abs(stat_parity_diff)))
            self.neptune_run["fairness/SPD"] = stat_parity_diff

    
    def equal_opportunity(self):
        if self.multiclass_pred and self.multiclass_sens: # Classifier: multiclass - Sens.attr: multiclass
            """
            P(y^=0|y=0,s=0) = P(y^=0|y=0,s=1) = ... = P(y^=0|y=0,s=N)
            [...]
            P(y^=M|y=M,s=0) = P(y^=M|y=M,s=1) = ... = P(y^=M|y=M,s=N)
            """
            equal_opp = []
            for y_hat_idx in self.class_range:
                equal_opp.append([])
                for s_idx in self.sens_attr_range:
                    try:
                        equal_opp[y_hat_idx].append(
                            sum(np.bitwise_and(self.y_hat[y_hat_idx], self.y_s[y_hat_idx][s_idx])) /
                            sum(self.y_s[y_hat_idx][s_idx])
                        )
                    except ZeroDivisionError:
                        equal_opp[y_hat_idx].append(0)
                    self.neptune_run["fairness/EO_y" + str(y_hat_idx) + "_s" + str(s_idx)] = equal_opp[y_hat_idx][s_idx]
        elif self.multiclass_sens: # Classifier: binary - Sens.attr: multiclass
            ''' P(y^=1|y=1,s=0) = P(y^=1|y=1,s=1) = ... = P(y^=1|y=1,s=N) '''
            equal_opp_s = []
            for s_idx in self.sens_attr_range:
                equal_opp_s.append(sum(self.pred_y[self.y1_s[s_idx]]) / sum(self.y1_s[s_idx]))
                self.neptune_run["fairness/EO_s" + str(s_idx)] = equal_opp_s[s_idx]
        else: # Classifier: binary - Sens.attr: binary
            ''' P(y^=1|y=1,s=0) = P(y^=1|y=1,s=1) '''
            # equal_opp = abs(sum(self.pred_y[self.y1_s0]) / sum(self.y1_s0) - sum(self.pred_y[self.y1_s1]) / sum(self.y1_s1))
            equal_opp_s0 = sum(self.pred_y[self.y1_s0]) / sum(self.y1_s0)
            equal_opp_s1 = sum(self.pred_y[self.y1_s1]) / sum(self.y1_s1)
            equal_opp_diff = equal_opp_s0 - equal_opp_s1
            self.neptune_run["fairness/EO_s0"] = equal_opp_s0
            self.neptune_run["fairness/EO_s1"] = equal_opp_s1
            
            print("Equal Opportunity Difference (EOD): {:.4f}".format(np.abs(equal_opp_diff)))
            self.neptune_run["fairness/EOD"] = equal_opp_diff


    def overall_accuracy_equality(self):
        if self.multiclass_pred and self.multiclass_sens: # Classifier: multiclass - Sens.attr: multiclass
            ''' P(y^=0|y=0,s=0) + ... + P(y^=M|y=M,s=0) = ... = P(y^=0|y=0,s=N) + ... + P(y^=M|y=M,s=N) '''
            oae_s = []
            for s_idx in self.sens_attr_range:
                oae_temp = 0.0
                for y_hat_idx in self.class_range:
                    try:
                        oae_temp += (
                            sum(np.bitwise_and(self.y_hat[y_hat_idx], self.y_s[y_hat_idx][s_idx])) /
                            sum(self.y_s[y_hat_idx][s_idx])
                        )
                    except ZeroDivisionError:
                        oae_temp += 0.0
                oae_s.append(oae_temp)
                self.neptune_run["fairness/OAE_s" + str(s_idx)] = oae_s[s_idx]
        elif self.multiclass_sens: # Classifier: binary - Sens.attr: multiclass
            ''' P(y^=0|y=0,s=0) + P(y^=1|y=1,s=0) = ... = P(y^=0|y=0,s=N) + P(y^=1|y=1,s=N)'''
            oae_s = []
            for s_idx in self.sens_attr_range:
                oae_s.append(
                    np.count_nonzero(self.pred_y[self.y0_s[s_idx]]==0) / sum(self.y0_s[s_idx]) +
                    sum(self.pred_y[self.y1_s[s_idx]]) / sum(self.y1_s[s_idx])
                )
                self.neptune_run["fairness/OAE_s" + str(s_idx)] = oae_s[s_idx]
        else: # Classifier: binary - Sens.attr: binary
            ''' P(y^=0|y=0,s=0) + P(y^=1|y=1,s=0) = P(y^=0|y=0,s=1) + P(y^=1|y=1,s=1) '''
            oae_s0 = np.count_nonzero(self.pred_y[self.y0_s0]==0) / sum(self.y0_s0) + sum(self.pred_y[self.y1_s0]) / sum(self.y1_s0)
            oae_s1 = np.count_nonzero(self.pred_y[self.y0_s1]==0) / sum(self.y0_s1) + sum(self.pred_y[self.y1_s1]) / sum(self.y1_s1)
            # oae_diff = abs(oae_s0 - oae_s1)
            oae_diff = oae_s0 - oae_s1
            self.neptune_run["fairness/OAE_s0"] = oae_s0
            self.neptune_run["fairness/OAE_s1"] = oae_s1
            
            print("Overall Accuracy Equality Difference (OAED): {:.4f}".format(np.abs(oae_diff)))
            self.neptune_run["fairness/OAED"] = oae_diff


    def treatment_equality(self):
        if self.multiclass_pred and self.multiclass_sens: # Classifier: multiclass - Sens.attr: multiclass
            """
            P(y^=0|y/=0,s=0) / P(y^/=0|y=0,s=0) = ... = P(y^=0|y/=0,s=N) / P(y^/=0|y=0,s=N)
            [...]
            P(y^=M|y/=M,s=0) / P(y^/=M|y=M,s=0) = ... = P(y^=M|y/=M,s=N) / P(y^/M|y=M,s=N)
            """
            te_fp_fn = []
            te_fn_fp = []
            te = []
            for y_hat_idx in self.class_range:
                te_fp_fn.append([])
                te_fn_fp.append([])
                abs_te_fp_fn = 0.0
                abs_te_fn_fp = 0.0
                te.append([])
                for s_idx in self.sens_attr_range:
                    try:
                        te_fp_fn[y_hat_idx].append(
                            (sum(np.bitwise_and(self.y_hat[y_hat_idx], self.yneq_s[y_hat_idx][s_idx])) / sum(self.yneq_s[y_hat_idx][s_idx])) /
                            (sum(np.bitwise_and(self.yneq_hat[y_hat_idx], self.y_s[y_hat_idx][s_idx])) / sum(self.y_s[y_hat_idx][s_idx]))
                        )
                    except ZeroDivisionError:
                        te_fp_fn[y_hat_idx].append(0)
                    
                    try:
                        te_fn_fp[y_hat_idx].append(
                            (sum(np.bitwise_and(self.yneq_hat[y_hat_idx], self.y_s[y_hat_idx][s_idx])) / sum(self.y_s[y_hat_idx][s_idx])) /
                            (sum(np.bitwise_and(self.y_hat[y_hat_idx], self.yneq_s[y_hat_idx][s_idx])) / sum(self.yneq_s[y_hat_idx][s_idx]))
                        )
                    except ZeroDivisionError:
                        te_fn_fp[y_hat_idx].append(0)

                    abs_te_fp_fn += abs(te_fp_fn[y_hat_idx][s_idx])
                    abs_te_fn_fp += abs(te_fn_fp[y_hat_idx][s_idx])
            
                    if abs_te_fp_fn < abs_te_fn_fp:
                        te[y_hat_idx].append(te_fp_fn[y_hat_idx][s_idx])
                    else:
                        te[y_hat_idx].append(te_fn_fp[y_hat_idx][s_idx])

            for y_idx in self.class_range:
                for s_idx in self.sens_attr_range:
                    self.neptune_run["fairness/TE_y" + str(y_idx) + "_s" + str(s_idx)] = te[y_idx][s_idx]

        elif self.multiclass_sens: # Classifier: binary - Sens.attr: multiclass
            ''' P(y^=1|y=0,s=0) / P(y^=0|y=1,s=0) = ... = P(y^=1|y=0,s=N) / P(y^=0|y=1,s=N) '''
            te1_s = []
            te0_s = []
            abs_te1 = []
            abs_te0 = []
            for s_idx in self.sens_attr_range:
                te1_s.append(
                    (sum(self.pred_y[self.y0_s[s_idx]]) / sum(self.y0_s[s_idx])) /
                    (np.count_nonzero(self.pred_y[self.y1_s[s_idx]]==0) / sum(self.y1_s[s_idx]))
                )
                te0_s.append(
                    (np.count_nonzero(self.pred_y[self.y1_s[s_idx]]==0) / sum(self.y1_s[s_idx])) /
                    (sum(self.pred_y[self.y0_s[s_idx]]) / sum(self.y0_s[s_idx]))
                )
                abs_te1.append(abs(te1_s[s_idx]))
                abs_te0.append(abs(te0_s[s_idx]))
            
            if sum(abs_te1) < sum(abs_te0):
                te_s = te1_s
            else:
                te_s = te0_s

            #for i in self.sens_attr_range:
                self.neptune_run["fairness/TE_s" + str(i)] = te_s[i]            

        else: # Classifier: binary - Sens.attr: binary
            ''' P(y^=1|y=0,s=0) / P(y^=0|y=1,s=0) = P(y^=1|y=0,s=1) / P(y^=0|y=1,s=1) '''
            te1_s0 = (sum(self.pred_y[self.y0_s0]) / sum(self.y0_s0)) / (np.count_nonzero(self.pred_y[self.y1_s0]==0) / sum(self.y1_s0))
            te1_s1 = (sum(self.pred_y[self.y0_s1]) / sum(self.y0_s1)) / (np.count_nonzero(self.pred_y[self.y1_s1]==0) / sum(self.y1_s1))
            te_diff_1 = te1_s0 - te1_s1
            abs_ted_1 = abs(te_diff_1)
            ''' P(y^=0|y=1,s=0) / P(y^=1|y=0,s=0) = P(y^=0|y=1,s=1) / P(y^=1|y=0,s=1) '''
            te0_s0 = (np.count_nonzero(self.pred_y[self.y1_s0]==0) / sum(self.y1_s0)) / (sum(self.pred_y[self.y0_s0]) / sum(self.y0_s0))
            te0_s1 = (np.count_nonzero(self.pred_y[self.y1_s1]==0) / sum(self.y1_s1)) / (sum(self.pred_y[self.y0_s1]) / sum(self.y0_s1))
            te_diff_0 = te0_s0 - te0_s1
            abs_ted_0 = abs(te_diff_0)

            # te_diff = min(te_diff_1, te_diff_0)
            if abs_ted_0 < abs_ted_1:
                te_s0 = te0_s0
                te_s1 = te0_s1
                te_diff = te_diff_0
            else:
                te_s0 = te1_s0
                te_s1 = te1_s1
                te_diff = te_diff_1

            self.neptune_run["fairness/TE_s0"] = te_s0
            self.neptune_run["fairness/TE_s1"] = te_s1
            
            print("Treatment Equality Difference (TED): {:.4f}".format(np.abs(te_diff)))
            self.neptune_run["fairness/TED"] = te_diff


    def disparate_impact(self):
        #num_of_priv = sum(self.s0)
        #num_of_unpriv = sum(self.s1)

        #unpriv_ratio = sum(self.pred_y[self.]/num_of_unpriv 

        #stat_parity_s0 = sum(self.pred_y[self.s0]) / sum(self.s0)
        #stat_parity_s1 = sum(self.pred_y[self.s1]) / sum(self.s1)
        #stat_parity_diff = stat_parity_s0 - stat_parity_s1

        print('true_y:', self.true_y)
        print('pred_y:', self.pred_y)