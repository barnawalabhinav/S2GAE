import re
import matplotlib.pyplot as plt
import sys
file_path = sys.argv[1]
def parse_metrics(file_path):
    # Regular expressions to match metrics
    epoch_re = re.compile(r"EPOCH:\s+(\d+)")
    auc_re = re.compile(r"(\w+) AUC:\s+([\d.]+)")
    
    epochs = []
    train_auc = []
    test_auc = []

    with open(file_path, 'r') as file:
        current_epoch = None
        for line in file:
            epoch_match = epoch_re.search(line)
            auc_match = auc_re.search(line)

            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                epochs.append(current_epoch)

            if auc_match:
                metric_type = auc_match.group(1).lower()  # train or test
                auc_value = float(auc_match.group(2))
                if metric_type == "train":
                    train_auc.append(auc_value)
                elif metric_type == "test":
                    test_auc.append(auc_value)

    return epochs, train_auc, test_auc

def plot_auc1(epochs, train_auc_freeze,test_auc_freeze,train_auc_finetune,test_auc_finetune,train_auc_scratch,test_auc_scratch,num,name):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs[:num], train_auc_freeze[:num], label="Train AUC Freeze")
    # plt.plot(epochs[:100], test_auc_freeze[:100], label="Test AUC Freeze")

    plt.plot(epochs[:num], train_auc_finetune[:num], label="Train AUC Finetune")
    # plt.plot(epochs[:100], test_auc_finetune[:100], label="Test AUC Finetune")

    plt.plot(epochs[:num], train_auc_scratch[:num], label="Train AUC Scratch")
    # plt.plot(epochs[:100], test_auc_scratch[:100], label="Test AUC Scratch")
    
    plt.title("Epoch-wise TRAIN AUC "+ name)
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(name+"_TRAIN.png")

def plot_auc2(epochs, train_auc_freeze,test_auc_freeze,train_auc_finetune,test_auc_finetune,train_auc_scratch,test_auc_scratch,num,name):
    plt.figure(figsize=(10, 6))
    # plt.plot(epochs[:num], train_auc_freeze[:num], label="Train AUC Freeze")
    plt.plot(epochs[:num], test_auc_freeze[:num], label="Test AUC Freeze")

    # plt.plot(epochs[:num], train_auc_finetune[:num], label="Train AUC Finetune")
    plt.plot(epochs[:num], test_auc_finetune[:num], label="Test AUC Finetune")

    # plt.plot(epochs[:num], train_auc_scratch[:num], label="Train AUC Scratch")
    plt.plot(epochs[:num], test_auc_scratch[:num], label="Test AUC Scratch")
    
    plt.title("Epoch-wise TEST AUC "+ name)
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(name+"_TEST.png")

# Example usage
# file_path = 'your_metrics_file.txt'  # Replace with your actual file path
epochs, train_auc_freeze, test_auc_freeze = parse_metrics(file_path+"1.txt")
epochs, train_auc_finetune, test_auc_finetune = parse_metrics(file_path+"2.txt")
epochs, train_auc_scratch, test_auc_scratch = parse_metrics(file_path+"3.txt")
plot_auc1(epochs, train_auc_freeze,test_auc_freeze,train_auc_finetune,test_auc_finetune,train_auc_scratch,test_auc_scratch,200,file_path[3:])
plot_auc2(epochs, train_auc_freeze,test_auc_freeze,train_auc_finetune,test_auc_finetune,train_auc_scratch,test_auc_scratch,200,file_path[3:])
