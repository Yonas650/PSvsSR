import os
import torch
import scipy.io
import numpy as np
import mat73
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW  #changed to AdamW optimizer
from torch.nn import CrossEntropyLoss
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  #for TensorBoard visualization
from torcheval.metrics.functional import multiclass_f1_score  #import F1 score function

from models.Conformer import EEGConformer
from models.ATCNet import ATCNet
from models.Inception import EEGInception

torch.backends.cudnn.enabled = False  #disable cuDNN to avoid cudnnException warnings

models = [ATCNet, EEGConformer, EEGInception]
datasets = ["Delay.mat", "Thermal.mat", "Urgency.mat", "Vibration.mat"]

#model and dataset selection
model_choice = ATCNet
dataset_choice = "Vibration.mat"
label_choice = "PS"  #PS labeling

def create_directories():
    results_path = '/scratch/yma9130/PSvsSR/results_PS_Vibration_ATCNet/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        print(f"created directory: {results_path}", flush=True)
    else:
        print(f"directory already exists: {results_path}", flush=True)
    for subfolder in ['accuracy', 'cfx', 'curves', 'logs']:
        folder_path = os.path.join(results_path, subfolder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"created subdirectory: {folder_path}", flush=True)
        else:
            print(f"subdirectory already exists: {folder_path}", flush=True)

def process_data():
    directory_path = '/scratch/yma9130/PSvsSR/datasets/'
    data_path = os.path.join(directory_path, dataset_choice)
    print("Path:", data_path, flush=True)
    vibration_dict = mat73.loadmat(data_path)
    vibration_list = vibration_dict['data']
    print(f"loaded data for {len(vibration_list)} subjects.", flush=True)

    nSubs = len(vibration_list)
    nConds, nChannels, nSamples, _ = np.shape(vibration_list[0])
    print(f"data shape per subject: Conditions={nConds}, Channels={nChannels}, Time Points={nSamples}", flush=True)

    y_vibration = []
    if label_choice == "SR":
        y_SR_path = os.path.join(directory_path, f'y_SR_{dataset_choice}')
        y_SR_dict = mat73.loadmat(y_SR_path)
        print("loaded SR labels from:", y_SR_path, flush=True)
        y_SR_list = y_SR_dict['y_SR']

    vibration_tr_con = []
    for i in range(len(vibration_list)):
        vibration_arr = vibration_list[i]

        #print the number of trials per condition
        print(f"\nSubject {i} - Number of trials per condition:", flush=True)
        trials_per_condition = []
        for j, cond in enumerate(vibration_arr):
            trials = cond.shape[2]  #shape is (channels, time_points, trials)
            trials_per_condition.append(trials)
            print(f"  Condition {j}: {trials} trials", flush=True)

        #check for consistency across conditions
        unique_trials = set(trials_per_condition)
        if len(unique_trials) > 1:
            print(f"  **warning:** inconsistent number of trials across conditions for Subject {i}.", flush=True)
            #determine the minimum number of trials across conditions
            min_trials = min(trials_per_condition)
            print(f"  **info:** trimming all conditions to {min_trials} trials for consistency.", flush=True)
            #trim trials in each condition to min_trials
            for j in range(nConds):
                current_trials = vibration_arr[j].shape[2]
                if current_trials > min_trials:
                    vibration_arr[j] = vibration_arr[j][:, :, :min_trials]
                    print(f"    - Condition {j}: trimmed from {current_trials} to {min_trials} trials.", flush=True)
            #recalculate expected and actual trials
            expected_total_trials = sum([min_trials for _ in range(nConds)])
            vibration_temp = np.concatenate([np.array(cond) for cond in vibration_arr], axis=2)
            actual_total_trials = vibration_temp.shape[2]
            print(f"Subject {i} - After trimming and concatenation shape: {vibration_temp.shape}", flush=True)
            if expected_total_trials != actual_total_trials:
                print(f"  **error:** after trimming, mismatch in total trials for Subject {i}. Expected {expected_total_trials}, got {actual_total_trials}.", flush=True)
                # we'lldecide how to handle this error
                #for now, we'll skip this subject
                print(f"  **action:** skipping Subject {i} due to trial mismatch.", flush=True)
                continue
            else:
                print(f"  **check passed:** total trials match after trimming for Subject {i}.", flush=True)
        else:
            print(f"  all conditions have {unique_trials.pop()} trials for Subject {i}.", flush=True)
            expected_total_trials = sum(trials_per_condition)
            vibration_temp = np.concatenate([np.array(cond) for cond in vibration_arr], axis=2)
            actual_total_trials = vibration_temp.shape[2]
            print(f"Subject {i} - After concatenation shape: {vibration_temp.shape}", flush=True)
            if expected_total_trials != actual_total_trials:
                print(f"  **error:** mismatch in total trials for Subject {i}. Expected {expected_total_trials}, got {actual_total_trials}.", flush=True)
                print(f"  **action:** skipping Subject {i} due to trial mismatch.", flush=True)
                continue
            else:
                print(f"  **check passed:** total trials match for Subject {i}.", flush=True)

        #transposition step: (channels, time_points, trials) -> (trials, channels, time_points)
        vibration_temp = np.transpose(vibration_temp, (2, 0, 1))
        print(f"Subject {i} - After transposition shape: {vibration_temp.shape}", flush=True)

        #check for consistent time points
        if vibration_temp.shape[2] !=1250:
            raise ValueError(f"Subject {i} has inconsistent time points: {vibration_temp.shape[2]} instead of 1250.")

        

        if label_choice == "SR":
            y_temp = np.array(y_SR_list[i][0])
            y_temp = y_temp.reshape(-1)  #ensure it's a 1D array
            #trim y_temp if necessary to match the number of trials
            if y_temp.shape[0] > vibration_temp.shape[0]:
                y_temp = y_temp[:vibration_temp.shape[0]]
            elif y_temp.shape[0] < vibration_temp.shape[0]:
                #trim vibration_temp if labels are fewer
                vibration_temp = vibration_temp[:y_temp.shape[0], :, :]

            if dataset_choice == "Thermal.mat":
                neutral_indices = np.where(y_temp == 2)[0]  #find indices where label is 2
                y_temp = np.delete(y_temp, neutral_indices)
                vibration_temp = np.delete(vibration_temp, neutral_indices, axis=0)
                y_temp = np.where(y_temp > 2, y_temp - 1, y_temp)  #adjust labels greater than 2
            print(f"Subject {i} - Labels shape after removing neutral: {y_temp.shape}", flush=True)
        else:
            if dataset_choice == "Delay.mat":
                y_temp = np.concatenate([
                    0 * np.ones(np.shape(vibration_arr[j])[2]) if j == 0 else
                    1 * np.ones(np.shape(vibration_arr[j])[2])
                    for j in range(nConds)
                ])
            else:
                y_temp = np.concatenate([
                    (j) * np.ones(np.shape(vibration_arr[j])[2])
                    for j in range(nConds)
                ])
            y_temp = y_temp.reshape(-1)  #ensure it's a 1D array
            print(f"Subject {i} - Labels shape: {y_temp.shape}", flush=True)

        #append data and labels
        vibration_tr_con.append(vibration_temp)
        y_vibration.append(y_temp)
        print(f"Subject {i} - Appended data shape: {vibration_temp.shape}", flush=True)
        print(f"Subject {i} - Appended labels shape: {y_temp.shape}", flush=True)

    print(f"\nNumber of subjects: {nSubs}, Number of conditions: {nConds}, Number of channels: {nChannels}, Number of samples: {nSamples}", flush=True)
    nConds = 2 if dataset_choice == "Delay.mat" else 4 if dataset_choice == "Thermal.mat" else nConds
    print(f"Adjusted number of conditions: {nConds}", flush=True)

    return vibration_tr_con, y_vibration, nSubs, nConds, nChannels, nSamples

def apply_filter(X):
    from scipy.signal import butter, sosfiltfilt

    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], btype='band', output='sos')
        return sos

    fs = 250  #sampling frequency
    lowcut = 0.5
    highcut = 40.0

    sos = butter_bandpass(lowcut, highcut, fs, order=5)
    #vectorized filtering: apply the bandpass filter along the time axis (axis=2)
    X_filtered = sosfiltfilt(sos, X, axis=2)
    print(f"Data shape after filtering: {X_filtered.shape}", flush=True)
    return X_filtered

def scaleData(X_train_t, X_test_t):
    for ch in range(60):
        scaler = StandardScaler()
        scaler.fit(X_train_t[:, ch, :])
        X_train_t[:, ch, :] = scaler.transform(X_train_t[:, ch, :])
        X_test_t[:, ch, :] = scaler.transform(X_test_t[:, ch, :])
    #ensure arrays are contiguous
    X_train_t = X_train_t.copy()
    X_test_t = X_test_t.copy()
    return X_train_t, X_test_t

def execute_model(vibration_tr_con, y_vibration, nSubs, nConds, nChannels, nSamples):
    confx = np.zeros((nSubs, nConds, nConds))
    scores = []  #to store accuracy for each LOSO fold
    f1_scores = []  #to store F1 scores for each LOSO fold

    #initialize lists to hold per-subject metrics
    all_subject_train_losses = []
    all_subject_test_losses = []
    all_subject_train_accuracies = []
    all_subject_test_accuracies = []

    #hyperparameters
    bs_t = 32
    learning_rate = 1e-3
    epochs = 100

    writer = SummaryWriter(log_dir='/scratch/yma9130/PSvsSR/results_PS_Vibration_ATCNet/logs')
    print("initialized TensorBoard SummaryWriter.", flush=True)

    for sub in range(nSubs):
        print(f'\nSubject: #{sub}', flush=True)

        X_test_t = vibration_tr_con[sub]
        x_train_t = vibration_tr_con[:sub] + vibration_tr_con[(sub + 1):]

        Y_test_t = np.squeeze(y_vibration[sub])
        y_train_t = y_vibration[:sub] + y_vibration[(sub + 1):]

        X_train_t = np.concatenate(x_train_t, axis=0)
        Y_train_t = np.concatenate(y_train_t)
        Y_train_t = np.squeeze(Y_train_t)

        print(f"Total training samples: {X_train_t.shape[0]}, Total training labels: {Y_train_t.shape[0]}", flush=True)
        print(f"Total test samples: {X_test_t.shape[0]}, Total test labels: {Y_test_t.shape[0]}", flush=True)

        #ensure labels are 1D
        Y_train_t = Y_train_t.reshape(-1)
        Y_test_t = Y_test_t.reshape(-1)

        #check for zero training samples
        if X_train_t.shape[0] == 0 or Y_train_t.shape[0] == 0:
            print(f"Skipping Subject {sub} due to no training data.", flush=True)
            continue

        print(f"X_train_t shape before filtering: {X_train_t.shape}", flush=True)
        print(f"X_test_t shape before filtering: {X_test_t.shape}", flush=True)

        #apply bandpass filter
        X_train_t = apply_filter(X_train_t)
        X_test_t = apply_filter(X_test_t)

        print(f"X_train_t shape before scaling: {X_train_t.shape}", flush=True)
        print(f"X_test_t shape before scaling: {X_test_t.shape}", flush=True)

        #normalize data
        X_train_t, X_test_t = scaleData(X_train_t, X_test_t)

        #ensure arrays are contiguous
        X_train_t = np.ascontiguousarray(X_train_t)
        X_test_t = np.ascontiguousarray(X_test_t)

        print(f"X_train_t shape after scaling: {X_train_t.shape}", flush=True)
        print(f"X_test_t shape after scaling: {X_test_t.shape}", flush=True)

        #convert to tensors
        X_train_t = torch.tensor(X_train_t, dtype=torch.float32)
        Y_train_t = torch.tensor(Y_train_t, dtype=torch.long)
        X_test_t = torch.tensor(X_test_t, dtype=torch.float32)
        Y_test_t = torch.tensor(Y_test_t, dtype=torch.long)

        #data augmentation
        noise_factor = 0.005
        X_train_noisy = X_train_t + noise_factor * torch.randn_like(X_train_t)
        X_train_t = torch.cat([X_train_t, X_train_noisy], dim=0)
        Y_train_t = torch.cat([Y_train_t, Y_train_t], dim=0)

        print(f"X_train_t shape after augmentation: {X_train_t.shape}", flush=True)
        print(f"Y_train_t shape after augmentation: {Y_train_t.shape}", flush=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"using device: {device}", flush=True)

        #Create datasets
        train_dataset = TensorDataset(X_train_t, Y_train_t)
        test_dataset = TensorDataset(X_test_t, Y_test_t)

        #handle class imbalance if label is SR
        #if label_choice == "SR":
        #calculate class weights
        class_counts = torch.bincount(Y_train_t, minlength=nConds)  #ensure counts for all classes
        class_weights = 1.0 / (class_counts.float() + 1e-6)  #add small value to avoid division by zero
        class_weights = class_weights.to(device)

        train_loader = DataLoader(train_dataset, batch_size=bs_t, shuffle=True)
        criterion = CrossEntropyLoss(weight=class_weights)
        """else:
            train_loader = DataLoader(train_dataset, batch_size=bs_t, shuffle=True)
            criterion = CrossEntropyLoss()"""

        test_loader = DataLoader(test_dataset, batch_size=bs_t, shuffle=False)

        print(f"Train Loader created with {len(train_loader)} batches.")
        print(f"Test Loader created with {len(test_loader)} batches.", flush=True)

        #model-specific parameters
        if model_choice == ATCNet:
            model = ATCNet(n_outputs=nConds, n_chans=X_train_t.shape[1], n_times=X_train_t.shape[2],
                           conv_block_dropout=0.5, n_windows=5).to(device)
        elif model_choice == EEGInception:
            model = EEGInception(n_outputs=nConds, n_chans=X_train_t.shape[1], n_times=X_train_t.shape[2],
                                 sfreq=250, drop_prob=0.7).to(device)  #original(250,0.5)
        elif model_choice == EEGConformer:
            model = EEGConformer(n_outputs=nConds, n_chans=X_train_t.shape[1], n_times=X_train_t.shape[2],
                                 sfreq=250, filter_time_length=128).to(device)

        #weight initialization
        def weights_init(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        model.apply(weights_init)
        print("applied xavier uniform initialization to conv2d layers.", flush=True)

        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
        print("initialized AdamW optimizer and ReduceLROnPlateau scheduler.", flush=True)

        print("initialized CrossEntropyLoss criterion.", flush=True)

        #initialize lists to store losses and accuracies for this subject
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        for epoch in range(epochs):
            model.train()
            train_loss, correct, total = 0.0, 0, 0

            for inputs, labels in train_loader:
                optimizer.zero_grad()
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            #calculate average training loss and accuracy for this epoch
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = correct / total

            #evaluate on test set
            model.eval()
            with torch.no_grad():
                test_loss, correct, total = 0.0, 0, 0
                all_y_true, all_y_pred = [], []
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    test_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    all_y_true.extend(labels.cpu().numpy())
                    all_y_pred.extend(predicted.cpu().numpy())

                #calculate average test loss and accuracy for this epoch
                avg_test_loss = test_loss / len(test_loader)
                test_accuracy = correct / total

                #calculate F1 score for this epoch
                f1_score = multiclass_f1_score(
                    torch.tensor(all_y_pred),
                    torch.tensor(all_y_true),
                    num_classes=nConds,
                    average="macro"
                )

            #scheduler step
            scheduler.step(avg_test_loss)

            #append to the lists
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

            #tensorBoard logging
            writer.add_scalar(f'Loss/Train_Subject_{sub}', avg_train_loss, epoch)
            writer.add_scalar(f'Loss/Test_Subject_{sub}', avg_test_loss, epoch)
            writer.add_scalar(f'Accuracy/Train_Subject_{sub}', train_accuracy, epoch)
            writer.add_scalar(f'Accuracy/Test_Subject_{sub}', test_accuracy, epoch)
            writer.add_scalar(f'F1_Score/Test_Subject_{sub}', f1_score.item(), epoch)

            #print epoch metrics
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                  f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.4f}, F1 Score: {f1_score.item():.4f}", flush=True)

        #after training, evaluate on test set
        print(f"Training completed for Subject {sub}. Now evaluating on test set.", flush=True)
        model.eval()
        with torch.no_grad():
            all_y_true, all_y_pred = [], []
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                all_y_true.extend(labels.cpu().numpy())
                all_y_pred.extend(predicted.cpu().numpy())

        #calculate accuracy for this LOSO fold
        correct = sum(p == t for p, t in zip(all_y_pred, all_y_true))
        total = len(all_y_true)
        test_accuracy = correct / total
        acc = test_accuracy * 100  #since test_accuracy is already a fraction
        scores.append(acc)

        #calculate and print F1 score for this LOSO fold
        loso_f1_score = multiclass_f1_score(
            torch.tensor(all_y_pred),
            torch.tensor(all_y_true),
            num_classes=nConds,
            average="macro"
        )
        f1_scores.append(loso_f1_score.item())
        print(f"Subject {sub} - LOSO F1 Score: {loso_f1_score.item():.4f}", flush=True)

        #print accuracy in the desired format
        print(f"Subject {sub} - Accuracy: {acc:.2f}%", flush=True)

        confx[sub, :, :] = confusion_matrix(all_y_true, all_y_pred, labels=np.arange(nConds))
        print(f"Confusion matrix for Subject {sub}:\n{confx[sub, :, :]}", flush=True)

        #append per-subject metrics to the overall lists
        all_subject_train_losses.append(train_losses)
        all_subject_test_losses.append(test_losses)
        all_subject_train_accuracies.append(train_accuracies)
        all_subject_test_accuracies.append(test_accuracies)

    writer.close()  #close the TensorBoard SummaryWriter
    print("closed TensorBoard SummaryWriter.", flush=True)

    #compute average metrics across subjects
    avg_train_losses = np.mean(all_subject_train_losses, axis=0)
    avg_test_losses = np.mean(all_subject_test_losses, axis=0)
    avg_train_accuracies = np.mean(all_subject_train_accuracies, axis=0)
    avg_test_accuracies = np.mean(all_subject_test_accuracies, axis=0)

    return confx, scores, f1_scores, avg_train_losses, avg_test_losses, avg_train_accuracies, avg_test_accuracies, epochs, nSubs

def display_results(confx, scores, f1_scores, avg_train_losses, avg_test_losses, avg_train_accuracies, avg_test_accuracies, epochs, nSubs):
    cfx_ = np.squeeze(np.mean(confx, axis=0))
    cfx_ = 100 * (cfx_ / cfx_.sum(axis=1, keepdims=True))

    overall_accuracy = np.mean(scores)
    overall_f1_score = np.mean(f1_scores)
    print(f"\noverall accuracy: {overall_accuracy:.2f}%", flush=True)
    print(f"\noverall F1 score: {overall_f1_score:.4f}", flush=True)

    results_path = '/scratch/yma9130/PSvsSR/results_PS_Vibration_ATCNet/'
    file_name = model_choice.__name__ + "_" + dataset_choice.replace(".mat", "") + "_" + label_choice

    np.save(os.path.join(os.path.join(results_path, 'accuracy'), file_name + '.npy'), scores, allow_pickle=True)
    np.save(os.path.join(os.path.join(results_path, 'accuracy'), file_name + '_f1.npy'), f1_scores, allow_pickle=True)
    print(f"saved accuracy scores to {os.path.join(results_path, 'accuracy', file_name + '.npy')}", flush=True)
    print(f"saved F1 scores to {os.path.join(results_path, 'accuracy', file_name + '_f1.npy')}", flush=True)

    disp = ConfusionMatrixDisplay(confusion_matrix=cfx_)
    disp.plot()
    plt.show()
    plt.savefig(os.path.join(os.path.join(results_path, 'cfx'), file_name + '.pdf'))
    print(f"saved confusion matrix to {os.path.join(results_path, 'cfx', file_name + '.pdf')}", flush=True)

    plt.figure(figsize=(12, 5))

    #since the lists are averaged over all subjects, adjust the x-axis accordingly
    total_epochs = epochs

    #plot average training and testing losses
    plt.subplot(1, 2, 1)
    plt.plot(range(1, total_epochs + 1), avg_train_losses, label='Train Loss')
    plt.plot(range(1, total_epochs + 1), avg_test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Average Training and Testing Losses')

    #plot average training and testing accuracies
    plt.subplot(1, 2, 2)
    plt.plot(range(1, total_epochs + 1), avg_train_accuracies, label='Train Accuracy')
    plt.plot(range(1, total_epochs + 1), avg_test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Average Training and Testing Accuracies')

    plt.show()
    plt.savefig(os.path.join(os.path.join(results_path, 'curves'), file_name + '.pdf'))
    print(f"saved learning curves to {os.path.join(results_path, 'curves', file_name + '.pdf')}", flush=True)

def main():
    print(f'Model: {model_choice.__name__}, Dataset: {dataset_choice.replace(".mat", "")}, Label: {label_choice}', flush=True)

    create_directories()  #create necessary directories

    vibration_tr_con, y_vibration, nSubs, nConds, nChannels, nSamples = process_data()

    confx, scores, f1_scores, avg_train_losses, avg_test_losses, avg_train_accuracies, avg_test_accuracies, epochs, nSubs = execute_model(
        vibration_tr_con, y_vibration, nSubs, nConds, nChannels, nSamples
    )

    display_results(confx, scores, f1_scores, avg_train_losses, avg_test_losses, avg_train_accuracies, avg_test_accuracies, epochs, nSubs)

if __name__ == "__main__":
    main()
