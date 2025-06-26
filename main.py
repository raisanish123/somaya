
# STEP 1: IMPORTS & DEVICE SETUP
import os
from datetime import datetime
import numpy as np
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import glob

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

import time
start_time = time.time()

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === HYPERPARAMETERS ===
config = {
    "sequence_length": 20,              # How many time steps the model sees at once
    "future": 1,                        # How many steps into the future to predict
    "test_ratio": 0.25, # Train/test split
   
   #temporal settings
    "temporal_batch_size": 64,                   # For both train and test loaders
    "temporal_epochs": 2,                       # Number of training epochs
    "temporal_hidden_dim": 64,                   # Hidden representation size
    "temporal_num_heads": 4,                     # Attention heads
    "temporal_num_layers": 3,               # Transformer depth
    "temporal_dropout": 0.1,
        
    #spatial settings
    "spatial_batch_size": 64,                   # For both train and test loaders
    "spatial_epochs": 2,                       # Number of training epochs
    "spatial_hidden_dim": 32,                   # Hidden representation size
    "spatial_num_heads": 2,                     # Attention heads
    "spatial_num_layers": 1,                    # Transformer depth
    "spatial_dropout": 0.1,
    
    "num_rooms": 5                      # R1–R5
}

# Format: Year/Month/Day-Hour_Min_Sec (slashes replaced for folder safety)
timestamp = datetime.now().strftime("%Y%m%d%H%M")
RESULTS_DIR = os.path.join("results", f"Experiment_Results_{timestamp}")
os.makedirs(RESULTS_DIR, exist_ok=True)

summary_txt = os.path.join(RESULTS_DIR, "summary_metrics.txt")

# Save hyperparameters to summary file
with open(summary_txt, "w") as f:
    f.write(" HYPERPARAMETERS USED IN THIS RUN \n")
    for key, value in config.items():
        f.write(f"{key}: {value}\n")
    f.write("\n")

# Load specific room CSV files from a folder
DATA_DIR = "training_data"
csv_files = sorted(glob.glob(os.path.join(DATA_DIR,"*.csv")))
file_paths = {f"R{i+1}": csv_files[i] for i in range(min(5, len(csv_files)))}
room_list = list(file_paths.keys()) 


target_column = 'occupant_count [number]'
room_data = {}

for room, path in file_paths.items():
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.ffill().bfill()
    room_data[room] = df
# Tracking info
    print(f"{room} - Shape: {df.shape}")
    print(f"{room} - Columns: {df.columns.tolist()}")
    print(f"{room} - Null values per column:\n{df.isnull().sum()}")
    print(f"{room} - Unique occupant counts: {df[target_column].unique()}")
    print(f"{room} - Sample data:\n{df.head()}")

# STEP 3: SCALING
excluded_columns = ['timestamp', target_column]
room_scaled_data = {}
scalers = {}

for room, df in room_data.items():
    print(f"\n Scaling data for {room} ")

    # Separate features and target
    features = df.drop(columns=excluded_columns)
    target = df[target_column]

    # Print feature shape and preview
    print(f"{room} - Features shape: {features.shape}")
    print(f"{room} - Feature columns: {features.columns.tolist()}")
    print(f"{room} - First 5 rows of features:\n{features.head()}")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Print scaled feature stats
    print(f"{room} - Scaled features shape: {scaled_features.shape}")
    print(f"{room} - First 3 rows of scaled features:\n{scaled_features[:3]}")

    # Store scaled data
    #scaler = StandardScaler()
    #scaled_features = scaler.fit_transform(features)
    room_scaled_data[room] = {
        'X': scaled_features,
        'y': target.values,
        'timestamp': df['timestamp'].values
    }
    scalers[room] = scaler

    # Target stats
    print(f"{room} - Target (y) shape: {target.shape}")
    print(f"{room} - Unique target values: {np.unique(target.values)}")

    # Timestamp range
    print(f"{room} - Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# STEP 4: SEQUENCE CREATION
def create_sequences(X, y, timestamps, seq_len, future):
#Creates overlapping sequences of features and targets with a future
        #Args:
        #X: Feature matrix (scaled)
        #y: Target values
        #timestamps: Corresponding timestamps
        #seq_len: Length of each input sequence
        #future: How many steps into the future to predict
        #Returns:
        #X_seq: Input sequences of shape [num_sequences, seq_len, num_features]
        #y_seq: Corresponding future targets
        #t_seq: Timestamps for each target

    X_seq, y_seq, t_seq = [], [], []
    for i in range(len(X) - seq_len - future + 1):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len + future - 1])
        t_seq.append(timestamps[i+seq_len + future - 1])
    return np.array(X_seq), np.array(y_seq), np.array(t_seq)

# Set parameters
sequence_length = config["sequence_length"]
future = config["future"]
test_ratio = config["test_ratio"]
room_sequences = {}

# Loop through each room and create sequences
for room in file_paths.keys():
    data = room_scaled_data[room]
    X_seq, y_seq, t_seq = create_sequences(data['X'], data['y'], data['timestamp'], sequence_length, future)
    # Print shapes to understand the structure
    print(f"{room} - Total sequences created: {X_seq.shape[0]}")
    print(f"{room} - Input sequence shape: {X_seq.shape} (should be [samples, seq_len, features])")
    print(f"{room} - Target shape: {y_seq.shape}")
    print(f"{room} - Timestamp shape: {t_seq.shape}")

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=test_ratio, shuffle=True)

    # Print split stats
    print(f"{room} - Training samples: {X_train.shape[0]}")
    print(f"{room} - Testing samples: {X_test.shape[0]}")

    # Store results
    room_sequences[room] = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }
    
print(f"Data preprocessing complete. Results saved in: {RESULTS_DIR}") 

#step 4:
import torch
import torch.nn as nn

#  Temporal Transformer Encoder for One Room
class TemporalTransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout):
      #input_dim: number of features per time step
      #hidden_dim: size the Transformer will use internally
      #num_heads: how many attention heads for better learning
      #num_layers: how many Transformer layers to stack

        super().__init__() #super().__init__() calls the parent class nn.Module to initialize everything
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # This is a linear layer that changes the input size.
        #For example: if each time step has 23 features (input_dim=23), and you want to use hidden_dim=64,
        #this layer changes the shape from (batch, seq_len, 23) → (batch, seq_len, 64).

        #This sets up one Transformer block:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, #the size of each hidden vector
            nhead=num_heads,#number of attention heads
            dim_feedforward=hidden_dim * 2,#the internal layer size inside the transformer
            dropout=dropout,#randomly turn off 10% of the neurons during training to avoid overfitting
            batch_first=True
        )

        #This stacks multiple encoder layers together (like putting num_layers blocks on top of each other
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    #This is the function that runs when you pass data into the encoder
    def forward(self, x):  # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # change input size to match hidden_dim
        x = self.encoder(x)     # apply transformer encoder
        return x


#  Spatial- Model for Multiple Rooms
class MultiRoomTransformerModel(nn.Module):
    def __init__(self, input_dim, seq_len, temporal_hidden_dim, temporal_num_heads, temporal_num_layers, num_rooms,spatial_hidden_dim, spatial_num_heads,spatial_num_layers, temporal_dropout, spatial_dropout):
        super().__init__()

        # One encoder for each room to learn time patterns. These learn time patterns inside each room.
        self.room_encoders = nn.ModuleList([
            TemporalTransformerEncoder(input_dim, temporal_hidden_dim, temporal_num_heads, temporal_num_layers, temporal_dropout)
            for _ in range(num_rooms)
        ])

        #Prediction heads per room. Each room has a small neural network.
        #Takes the encoded sequence from the Transformer. Flattens it.
        #Turns it into a prediction for that room.
        self.temporal_heads = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(seq_len * temporal_hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            for _ in range(num_rooms)
        ])
        
        # Projection if temporal and spatial hidden dims differ
        if temporal_hidden_dim != spatial_hidden_dim:
            self.temporal_to_spatial = nn.Linear(temporal_hidden_dim, spatial_hidden_dim)
        else:
            self.temporal_to_spatial = nn.Identity()
            
        spatial_encoder_layer = nn.TransformerEncoderLayer(
            d_model=spatial_hidden_dim,
            nhead=spatial_num_heads,
            dim_feedforward=spatial_hidden_dim * 2,
            dropout=spatial_dropout,
            batch_first=True
        )
        # This part looks at all rooms together
        # It helps the model learn how rooms affect each other
  
        self.spatial_encoder = nn.TransformerEncoder(spatial_encoder_layer, num_layers=spatial_num_layers)
    
        # After combining room info, this layer makes the final prediction
        self.spatial_output_head = nn.Linear(spatial_hidden_dim, 1)

    def forward(self, x):  # x: (batch, num_rooms, seq_len, input_dim)
     #takes input x which is a batch of sensor data for all rooms.
        batch_size, num_rooms, seq_len, input_dim = x.shape
       


        room_contexts = []      # store one summary from each room to combine later
        temporal_predictions = []  # store predictions for each room

        for r in range(num_rooms):#Loops through each room and gets its data from the input.
           
            x_room = x[:, r] # get data for one room(batch, seq-length, features_num)
            encoded = self.room_encoders[r](x_room) #Passes that room's data through its Transformer to learn time patterns.
           
            #Takes the average over all time steps to get a summary vector for that room and saves it.
            pooled = encoded.mean(dim=1) # get summary for this room
           

            projected= self.temporal_to_spatial(pooled)
          
            room_contexts.append(projected) # (batch, spatial_hidden_dim)

            #Flattens the encoded output and uses the room’s small network to make a prediction
            temporal_pred = self.temporal_heads[r](encoded).squeeze(-1)
            temporal_predictions.append(temporal_pred)

        # Combine all room summaries into one big tensor
        context_tensor = torch.stack(room_contexts, dim=1)  #shape: (batch, num_rooms, hidden_dim)
        

        # Combine room info so model can understand how rooms affect each other
        # This is like letting the rooms know about each other
        fused = self.spatial_encoder(context_tensor)  # (batch, num_rooms, hidden_dim)

        # Make prediction for each room after combining info
        spatial_pred = self.spatial_output_head(fused).squeeze(-1)  # (batch, num_rooms)
        temporal_pred = torch.stack(temporal_predictions, dim=1)   # (batch, num_rooms)

        return temporal_pred, spatial_pred

from torch.utils.data import Dataset, DataLoader

class MultiRoomDataset(Dataset):
    def __init__(self, room_sequences, room_names, is_train=True):
        self.X = []
        self.y = []
        self.room_names = room_names

        # Determine which data to use based on is_train flag
        data_key_X = 'X_train' if is_train else 'X_test'
        data_key_y = 'y_train' if is_train else 'y_test'

        # Align all rooms and build combined input/output tensors
        num_samples = len(room_sequences[self.room_names[0]][data_key_X])

        for i in range(num_samples):
            rooms_input = []
            rooms_target = []

            for room in self.room_names:
                rooms_input.append(room_sequences[room][data_key_X][i])   # (seq_len, num_features)
                rooms_target.append(room_sequences[room][data_key_y][i])  # scalar

            self.X.append(np.stack(rooms_input))  # shape: (num_rooms, seq_len, num_features)
            self.y.append(np.array(rooms_target)) # shape: (num_rooms,)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = MultiRoomDataset(room_sequences,room_list, is_train=True)
train_loader = DataLoader(train_dataset, batch_size=config["spatial_batch_size"], shuffle=True)
test_dataset = MultiRoomDataset(room_sequences, room_list, is_train=False)
test_loader = DataLoader(test_dataset, batch_size=config["spatial_batch_size"], shuffle=False)# shuffle=False for consistent evaluation




# MODEL HYPERPARAMETERS
num_sensor_features = room_scaled_data['R1']['X'].shape[1]   # Total number of features per timestep it is ok to have only room1 because all rooms R1–R5 have the same number of sensor features

# We do NOT need to instantiate TemporalTransformerEncoder separately.
# The MultiRoomTransformerModel automatically creates one TemporalTransformerEncoder
# for each room inside its __init__ method using a loop:
#     self.room_encoders = nn.ModuleList([...])
# This means each room gets its own dedicated temporal encoder.
# These encoders are then used inside the model's forward pass, so we only need to
# instantiate the full multi_room_model once, and it already includes:
#   - Temporal encoders for each room
#   - A shared spatial encoder to combine room information
#   - Output prediction heads for temporal and spatial predictions

#INSTANTIATE MULTI-ROOM TRANSFORMER MODEL
multi_room_model = MultiRoomTransformerModel(
    input_dim=num_sensor_features,
    seq_len=config["sequence_length"],
    temporal_hidden_dim=config["temporal_hidden_dim"],
    temporal_num_heads=config["temporal_num_heads"],
    temporal_num_layers=config["temporal_num_layers"],
    num_rooms=config["num_rooms"],
    spatial_hidden_dim=config["spatial_hidden_dim"],
    spatial_num_heads=config["spatial_num_heads"],
    spatial_num_layers=config["spatial_num_layers"],
    temporal_dropout=config["temporal_dropout"],
    spatial_dropout=config["spatial_dropout"]
).to(device)

# LOAD A MINI-BATCH FROM TRAINING DATA
# Each batch contains:
# - input_batch: shape [batch_size, total_rooms, seq_len, num_features]
# - actual_occupancy_batch: shape [batch_size, total_rooms]
input_batch, actual_occupancy_batch = next(iter(train_loader))
input_batch = input_batch.to(device)
actual_occupancy_batch = actual_occupancy_batch.to(device)

# test the model: single forward pass without traiing
# We use torch.no_grad() to turn off gradient tracking.
# This is important because we're not training the model and just testing it.
# It saves memory and speeds up computation.
print("\n Running one forward pass to see output shapes")
with torch.no_grad():
  # Pass the input batch through the model.
    # This runs one forward pass on the multi-room Transformer model.
    # The model returns two sets of predictions:
    # - temporal_predictions: output from each room’s own temporal encoder
    # - spatial_predictions: output after combining all rooms using the spatial encoder
    temporal_predictions, spatial_predictions = multi_room_model(input_batch)
    # At this point, both outputs have shape: [batch_size, total_rooms]
    # Each element is the predicted occupancy count for a room in a sample.

#  confirm the output shapes match as expected
print("\nForward pass complete.")
print("Temporal predictions shape: ", temporal_predictions.shape)     # → (batch_size, total_rooms)
print("Spatial predictions shape:  ", spatial_predictions.shape)      # → (batch_size, total_rooms)
print("Actual occupancy shape:     ", actual_occupancy_batch.shape)   # → (batch_size, total_rooms)

# print sample predictions for all rooms
print("\n Sample predictions for first 5 data points in each room ")
for room_index in range(config["num_rooms"]):
    print(f"\n Room {room_index + 1} Predictions:")
    print("Index\tTemporal\tSpatial\t\tActual")
    for i in range(5):  # First 5 predictions from batch
        temp_pred = temporal_predictions[i, room_index].item()
        spat_pred = spatial_predictions[i, room_index].item()
        actual_val = actual_occupancy_batch[i, room_index].item()
        print(f"{i}\t{temp_pred:.2f}\t\t{spat_pred:.2f}\t\t{actual_val}")

# @title
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
temporal_train_losses = {}
temporal_test_losses = {}
temporal_train_accs = {}
temporal_test_accs = {}
temporal_preds = {}
temporal_actuals = {}

#trains the temporal transformer model for one room
def train_and_plot_single_room(room_id, input_seq_len, feature_dim, hidden_dim, num_heads, num_layers, epochs=config["temporal_epochs"]):
    print(f"\n Training Model for {room_id} =")

    # Load and prepare data
    # Convert numpy arrays to PyTorch tensors and move them to GPU if available
    X_train = torch.tensor(room_sequences[room_id]['X_train'], dtype=torch.float32).to(device)
    y_train = torch.tensor(room_sequences[room_id]['y_train'], dtype=torch.float32).to(device)
    X_test = torch.tensor(room_sequences[room_id]['X_test'], dtype=torch.float32).to(device)
    y_test = torch.tensor(room_sequences[room_id]['y_test'], dtype=torch.float32).to(device)

    
    print(f"{room_id} - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"{room_id} - X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    

    # Combine training data into a PyTorch DataLoader for batch training
    train_loader = DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=config["temporal_batch_size"], shuffle=True)

    # Build model
    # Transformer encoder for time-series data (learns patterns over time)
    encoder = TemporalTransformerEncoder(feature_dim, hidden_dim, num_heads, num_layers,config["temporal_dropout"] ).to(device)

    # Prediction head: fully connected layers that turn transformer output into a single number
    head = nn.Sequential(
        nn.Flatten(),# Flattens the (batch, seq_len, hidden_dim) into (batch, seq_len * hidden_dim)
        nn.Linear(input_seq_len * hidden_dim, 64),#First dense layer
        nn.ReLU(),  # Rectified Linear Unit
        nn.Linear(64, 1)   # Output one value (occupancy count)
    ).to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=0.001)
    criterion = nn.MSELoss()

    # Tracking
    train_losses, test_losses = [], []
    train_accuracies=[]
    test_accuracies = []
    all_test_preds = []

    for epoch in range(epochs):
        encoder.train()
        head.train()

        total_loss, correct = 0, 0
        total_samples = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            encoded = encoder(batch_X) # Pass through transformer
            preds = head(encoded).squeeze(-1) # Remove last dimension for (batch_size,)
            loss = criterion(preds, batch_y) # Compute loss
            loss.backward()
            optimizer.step()

            # Track training accuracy (rounded occupancy match)
            total_loss += loss.item()
            correct += (preds.round() == batch_y.round()).sum().item()
            total_samples += batch_y.size(0)

        # Save training metrics
        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct / total_samples
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        # Evaluate
        encoder.eval()
        head.eval()
        with torch.no_grad():
            encoded_test = encoder(X_test)
            preds_test = head(encoded_test).squeeze(-1)
            test_loss = criterion(preds_test, y_test).item()
            test_acc = (preds_test.round() == y_test.round()).sum().item() / len(y_test)

        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}, "
              f"Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")

    # Final predictions for plotting
    all_test_preds = preds_test.detach().cpu().numpy() #Stop tracking gradients (for safety and speed)
    y_test_np = y_test.detach().cpu().numpy()
    
    temporal_train_losses[room_id] = train_losses
    temporal_test_losses[room_id] = test_losses
    temporal_train_accs[room_id] = train_accuracies
    temporal_test_accs[room_id] = test_accuracies
    temporal_preds[room_id] = all_test_preds[:100]
    temporal_actuals[room_id] = y_test_np[:100]

    
    # 4. Final evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test_np, all_test_preds))
    mae = mean_absolute_error(y_test_np, all_test_preds)
    r2 = r2_score(y_test_np, all_test_preds)
    print(f"\nFinal Evaluation for {room_id}:")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE : {mae:.3f}")
    print(f"  R²  : {r2:.3f}")
    with open(summary_txt, "a") as f:
        f.write(f"{room_id}\tRMSE: {rmse:.4f}\tMAE: {mae:.4f}\tR²: {r2:.4f}\n")
with open(summary_txt, "a") as f:
    f.write("\n TEMPORAL (Single-Room) Transformer Evaluation \n")       
for room_id in file_paths.keys():
    train_and_plot_single_room(
        room_id=room_id,
        input_seq_len=config["sequence_length"],
        feature_dim=num_sensor_features,
        hidden_dim=config["temporal_hidden_dim"],
        num_heads=config["temporal_num_heads"],
        num_layers=config["temporal_num_layers"],
        epochs=config["temporal_epochs"]
    )
room_names = list(file_paths.keys())

def plot_temporal_loss_subplots(temporal_train_losses, temporal_test_losses, room_names, save_path):
    fig, axs = plt.subplots(3,2, figsize=(14, 10), sharey=True)
    axs = axs.flatten()
    fig.suptitle("Temporal Train/Test Loss per Room", fontsize=16)

    for i, room in enumerate(room_names):
        axs[i].plot(temporal_train_losses[room], label='Train Loss')
        axs[i].plot(temporal_test_losses[room], label='Test Loss')
        axs[i].set_title(room)
        axs[i].set_xlabel("Epoch")
        axs[i].grid(True)
        if i == 0:
            axs[i].set_ylabel("Loss")
        axs[i].legend()
    fig.delaxes(axs[-1])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()

def plot_temporal_accuracy_subplots(temporal_train_accs, temporal_test_accs, room_names, save_path):
    fig, axs = plt.subplots(3,2, figsize=(14, 10), sharey=True)
    axs = axs.flatten()
    fig.suptitle("Temporal Train/Test Accuracy per Room", fontsize=16)

    for i, room in enumerate(room_names):
        axs[i].plot(temporal_train_accs[room], label='Train Acc')
        axs[i].plot(temporal_test_accs[room], label='Test Acc')
        axs[i].set_title(room)
        axs[i].set_xlabel("Epoch")
        axs[i].grid(True)
        if i == 0:
            axs[i].set_ylabel("Accuracy")
        axs[i].legend()
    fig.delaxes(axs[-1])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()

def plot_temporal_actual_vs_predicted_subplots(temporal_preds, temporal_actuals, room_names, save_path):
    fig, axs = plt.subplots(3,2, figsize=(14,10), sharey=True)
    axs = axs.flatten()
    fig.suptitle("Temporal Actual vs Predicted (First 100 Samples)", fontsize=16)

    for i, room in enumerate(room_names):
        axs[i].plot(temporal_actuals[room][:100], label='Actual')
        axs[i].plot(temporal_preds[room][:100], label='Predicted')
        axs[i].set_title(room)
        axs[i].set_xlabel("Index")
        axs[i].grid(True)
        if i == 0:
            axs[i].set_ylabel("Occupancy")
        axs[i].legend()
    fig.delaxes(axs[-1])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()

plot_temporal_loss_subplots(
    temporal_train_losses,
    temporal_test_losses,
    room_names,
    os.path.join(RESULTS_DIR, "temporal_loss_all_rooms.png")
)

plot_temporal_accuracy_subplots(
    temporal_train_accs,
    temporal_test_accs,
    room_names,
    os.path.join(RESULTS_DIR, "temporal_accuracy_all_rooms.png")
    )

plot_temporal_actual_vs_predicted_subplots(
    temporal_preds,
    temporal_actuals,
    room_names,
    os.path.join(RESULTS_DIR, "temporal_actual_vs_predicted_all_rooms.png")
)

room_names = list(file_paths.keys())
spatial_train_losses_per_room = {room: [] for room in room_names}
spatial_test_losses_per_room = {room: [] for room in room_names}
spatial_train_accs_per_room = {room: [] for room in room_names}
spatial_test_accs_per_room = {room: [] for room in room_names}
spatial_preds_per_room = {}
spatial_actuals_per_room = {}

# Train the multi-room Transformer model using spatial predictions
def train_multiroom_model(model, train_loader, epochs, test_loader,lr=0.001):

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    #print("Optimizer:", optimizer)
    print("Loss Function:", criterion)

    # Lists to track metrics for plotting and analysis
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    # Training loop
    for epoch in range(epochs):
        model.train()   # Set model to training mode
        total_loss, correct, total_samples = 0, 0, 0


        # Go through each batch of data
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            _, spatial_preds = model(X_batch)  # Get predictions (ignore temporal_preds)
            loss = criterion(spatial_preds, y_batch)  # Compute loss based on spatial predictions
            loss.backward() #Computes gradients	After forward pass & loss
            optimizer.step()  # uppdate weights using gradients

            total_loss += loss.item()
            correct += (spatial_preds.round() == y_batch).sum().item() # Count correct predictions
            total_samples += y_batch.numel()

        # Compute average training loss and accuracy
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total_samples
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Evaluation
        model.eval()
        with torch.no_grad():
            all_preds, all_actuals = [], []
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                _, spatial_preds = model(X_batch)
                all_preds.append(spatial_preds.cpu())
                all_actuals.append(y_batch.cpu())

            # Combine all batches into full test set
            preds_all = torch.cat(all_preds)
            y_all = torch.cat(all_actuals)
            test_loss = criterion(preds_all, y_all).item()
            test_acc = (preds_all.round() == y_all.round()).sum().item() / y_all.numel()

        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f} | Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")

    # Save per-room tracking results (first 100)
    for i, room in enumerate(room_names):
        spatial_preds_per_room[room] = preds_all[:100, i].numpy()
        spatial_actuals_per_room[room] = y_all[:100, i].numpy()

    for room in room_names:
        spatial_train_losses_per_room[room] = train_losses
        spatial_test_losses_per_room[room] = test_losses
        spatial_train_accs_per_room[room] = train_accs
        spatial_test_accs_per_room[room] = test_accs

    return preds_all.numpy(), y_all.numpy()

# === TRAIN THE MODEL ===
preds_np, actuals_np = train_multiroom_model(
    model=multi_room_model,
    train_loader=train_loader,
    test_loader=test_loader,
    epochs=config["spatial_epochs"],
    lr=0.001
)

print("\n Final Evaluation Metrics Per Room (Spatial Model):")
with open(summary_txt, "a") as f:
    f.write("\n SPATIAL (Multi-Room) Transformer Evaluation \n")

    # Store all spatial results
    spatial_results = []

    for room_index in range(preds_np.shape[1]):
        y_true = actuals_np[:, room_index]
        y_pred = preds_np[:, room_index]

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f"\nRoom {room_index + 1}:")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  MAE : {mae:.3f}")
        print(f"  R²  : {r2:.3f}")
        room_id = f"R{room_index + 1}"
        spatial_results.append((room_index + 1, rmse, mae, r2))

    # Write in temporal-style format
    for room_id, rmse, mae, r2 in spatial_results:
        f.write(f"R{room_id}\tRMSE: {rmse:.4f}\tMAE: {mae:.4f}\tR²: {r2:.4f}\n")
   

def plot_multiroom_loss_subplots(train_losses_per_room, test_losses_per_room, room_names, save_path):
    fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharey=True)
    axs = axs.flatten()
    fig.suptitle("Spatial Train/Test Loss per Room", fontsize=16)

    for i, room in enumerate(room_names):
        axs[i].plot(train_losses_per_room[room], label='Train Loss')
        axs[i].plot(test_losses_per_room[room], label='Test Loss')
        axs[i].set_title(room)
        axs[i].set_xlabel("Epoch")
        axs[i].grid(True)
        if i == 0:
            axs[i].set_ylabel("Loss")
        axs[i].legend()
    fig.delaxes(axs[-1])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()


def plot_multiroom_accuracy_subplots(train_accs_per_room, test_accs_per_room, room_names, save_path):
    fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharey=True)
    axs = axs.flatten()
    fig.suptitle("Spatial Train/Test Accuracy per Room", fontsize=16)

    for i, room in enumerate(room_names):
        axs[i].plot(train_accs_per_room[room], label='Train Acc')
        axs[i].plot(test_accs_per_room[room], label='Test Acc')
        axs[i].set_title(room)
        axs[i].set_xlabel("Epoch")
        axs[i].grid(True)
        if i == 0:
            axs[i].set_ylabel("Accuracy")
        axs[i].legend()
    fig.delaxes(axs[-1])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()


def plot_multiroom_actual_vs_predicted_subplots(preds_per_room, actuals_per_room, room_names, save_path):
    fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharey=True)
    axs = axs.flatten()
    fig.suptitle("Spatial Actual vs Predicted (First 100 Samples)", fontsize=16)

    for i, room in enumerate(room_names):
        axs[i].plot(actuals_per_room[room], label='Actual')
        axs[i].plot(preds_per_room[room], label='Predicted')
        axs[i].set_title(room)
        axs[i].set_xlabel("Index")
        axs[i].grid(True)
        if i == 0:
            axs[i].set_ylabel("Occupancy")
        axs[i].legend()
    fig.delaxes(axs[-1])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()

# === CALL GROUPED PLOTS ===
plot_multiroom_loss_subplots(
    spatial_train_losses_per_room,
    spatial_test_losses_per_room,
    room_names,
    os.path.join(RESULTS_DIR, "multiroom_loss_all_rooms.png")
)

plot_multiroom_accuracy_subplots(
    spatial_train_accs_per_room,
    spatial_test_accs_per_room,
    room_names,
    os.path.join(RESULTS_DIR, "multiroom_accuracy_all_rooms.png")
)

plot_multiroom_actual_vs_predicted_subplots(
    spatial_preds_per_room,
    spatial_actuals_per_room,
    room_names,
    os.path.join(RESULTS_DIR, "multiroom_actual_vs_predicted_all_rooms.png")
)

# === GET ONE BATCH FROM TEST LOADER ===
multiroom_batch = next(iter(test_loader))  # or use train_loader
X_batch = multiroom_batch[0].to(device)    # shape: (batch_size, num_rooms, seq_len, input_dim)

# === Confirm input shape ===
print("X_batch shape:", X_batch.shape)  # Should be (batch, num_rooms, seq_len, input_dim)

# === PLOT ROOM DEPENDENCY HEATMAP ===
def plot_room_dependency_heatmap(model, input_batch, room_names=None, save=True, show=False):
    model.eval()
    with torch.no_grad():
        assert input_batch.ndim == 4, f"Expected shape (batch, rooms, seq_len, features), got {input_batch.shape}"
        batch_size, num_rooms, seq_len, input_dim = input_batch.shape

        # Encode each room using its temporal encoder
        room_contexts = []
        for r in range(num_rooms):
            x_room = input_batch[:, r]  # (batch, seq_len, input_dim)
            encoded = model.room_encoders[r](x_room)  # (batch, seq_len, hidden_dim)
            pooled = encoded.mean(dim=1)  # (batch, hidden_dim)
            room_contexts.append(pooled)

        room_contexts = [model.temporal_to_spatial(c) for c in room_contexts]

        # Stack into (batch, num_rooms, hidden_dim)
        context_tensor = torch.stack(room_contexts, dim=1)

        # Apply spatial encoder to get fused representation
        fused = model.spatial_encoder(context_tensor)  # (batch, num_rooms, hidden_dim)

        # Average across the batch → (num_rooms, hidden_dim)
        avg_representation = fused.mean(dim=0).cpu().numpy()

        # Compute correlation matrix → (num_rooms, num_rooms)
        corr_matrix = np.corrcoef(avg_representation)

        # Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
                    xticklabels=room_names, yticklabels=room_names)
        plt.title("Room Dependency Heatmap (Learned Spatial Representations)")
        plt.xlabel("Rooms")
        plt.ylabel("Rooms")

        if save:
            plt.savefig(os.path.join(RESULTS_DIR, "multiroom_room_dependency_heatmap.png"))
        if show:
            plt.show()
        else:
            plt.close()

# === CALL FUNCTION ===
plot_room_dependency_heatmap(multi_room_model, X_batch, room_names)


# === SAVE RUNTIME INFO ===
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_minutes = elapsed_time / 60

with open(summary_txt, "a") as f:
    f.write(f"\nTotal Runtime: {elapsed_time:.2f} seconds ({elapsed_minutes:.2f} minutes)\n")
