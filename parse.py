import re
import sys

file_path = sys.argv[1]
def parse_metrics(file_path):
    # Regular expressions to match metrics
    epoch_re = re.compile(r"EPOCH:\s+(\d+)")
    # auc_re = re.compile(r"(\w+) AUC:\s+([\d.]+)")
    # acc_re = re.compile(r"(\w+) ACC:\s+([\d.]+)")
    rmse_re = re.compile(r"(\w+) RMSE:\s+([\d.]+)")
    
    # metrics = {
    #     "train_auc": [],
    #     "train_acc": [],
    #     "valid_auc": [],
    #     "valid_acc": [],
    #     "test_auc": [],
    #     "test_acc": []
    # }
    metrics = {
        "train_rmse": [],
        "test_rmse": [],
        "valid_rmse": []
    }

    with open(file_path, 'r') as file:
        for line in file:
            # Match for each metric
            epoch_match = epoch_re.search(line)
            # auc_match = auc_re.search(line)
            # acc_match = acc_re.search(line)
            rmse_match = rmse_re.search(line)

            # if auc_match:
            #     metric_type = auc_match.group(1).lower()  # train, valid, test
            #     auc_value = float(auc_match.group(2))
            #     metrics[f"{metric_type}_auc"].append(auc_value)
            
            # if acc_match:
            #     metric_type = acc_match.group(1).lower()
            #     acc_value = float(acc_match.group(2))
            #     metrics[f"{metric_type}_acc"].append(acc_value)

            if rmse_match:
                metric_type = rmse_match.group(1).lower()
                rmse_val = float(rmse_match.group(2))
                metrics[f"{metric_type}_rmse"].append(rmse_val)
    # Find the best metrics (based on highest AUC)

    best_train_rmse = min(metrics["train_rmse"])
    best_valid_rmse = min(metrics["valid_rmse"])
    best_test_rmse = min(metrics["test_rmse"])
    # best_train_auc = max(metrics["train_auc"])
    # best_train_acc = metrics["train_acc"][metrics["train_auc"].index(best_train_auc)]
    
    # best_valid_auc = max(metrics["valid_auc"])
    # best_valid_acc = metrics["valid_acc"][metrics["valid_auc"].index(best_valid_auc)]
    
    # best_test_auc = max(metrics["test_auc"])
    # best_test_acc = metrics["test_acc"][metrics["test_auc"].index(best_test_auc)]
    
    return {
        "best_train_rmse": best_train_rmse,
        "best_valid_rmse": best_valid_rmse,
        "best_test_rmse": best_test_rmse
    }
    # return {
    #     "best_train_auc": best_train_auc,
    #     "best_train_acc": best_train_acc,
    #     "best_valid_auc": best_valid_auc,
    #     "best_valid_acc": best_valid_acc,
    #     "best_test_auc": best_test_auc,
    #     "best_test_acc": best_test_acc
    # }

# Example usage
# file_path = 'mlpbace1.txt'  # Replace with your actual file path
best_metrics = parse_metrics(file_path)
print("Best Metrics:", best_metrics)
