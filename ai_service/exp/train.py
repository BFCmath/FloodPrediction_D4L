# %% [code] {"execution":{"iopub.status.busy":"2025-11-21T05:47:40.582108Z","iopub.execute_input":"2025-11-21T05:47:40.582598Z","iopub.status.idle":"2025-11-21T05:47:40.586769Z","shell.execute_reply.started":"2025-11-21T05:47:40.582576Z","shell.execute_reply":"2025-11-21T05:47:40.585991Z"},"_kg_hide-output":true}
# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2025-11-23T15:15:04.035694Z","iopub.execute_input":"2025-11-23T15:15:04.036144Z","iopub.status.idle":"2025-11-23T15:15:08.935766Z","shell.execute_reply.started":"2025-11-23T15:15:04.036124Z","shell.execute_reply":"2025-11-23T15:15:08.934957Z"}}
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    """
    Đây là khối xây dựng cơ bản (Building Block). 
    Nó giống LSTM nhưng thay phép nhân ma trận bằng phép Conv2d 
    để giữ lại thông tin không gian (địa hình, vị trí nước).
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # Ghép đầu vào hiện tại và trạng thái ẩn (hidden state) cũ
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class FloodForecastModel(nn.Module):
    """
    Model chính: Encoder-Decoder ConvLSTM
    """
    def __init__(self, input_channels=4, hidden_channels=64, kernel_size=3):
        super(FloodForecastModel, self).__init__()
        
        # Input Channels = 4 bao gồm: 
        # 1. Rainfall (dynamic)
        # 2. Past Flood Depth (dynamic) - Optional (có thể bỏ nếu muốn test experiment 4)
        # 3. DEM (static)
        # 4. Manning (static)
        
        self.hidden_channels = hidden_channels
        self.conv_lstm = ConvLSTMCell(input_channels, hidden_channels, kernel_size, bias=True)
        
        # Lớp cuối cùng để chuyển từ Hidden State -> Water Depth (1 channel)
        self.final_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1) 

    def forward(self, dynamic_input, static_input, future_steps=4):
        """
        dynamic_input: (Batch, Time_in, Channels=2, Height, Width) -> Rain + Past Depth
        static_input:  (Batch, Channels=2, Height, Width) -> DEM + Manning
        future_steps:  Số frame cần dự đoán (mặc định 4)
        """
        b, t, c, h, w = dynamic_input.size()
        
        # Khởi tạo trạng thái ẩn (h, c)
        h, c = self.conv_lstm.init_hidden(b, (h, w))
        
        # --- ENCODER ---
        # Duyệt qua 12 khung hình quá khứ
        for time_step in range(t):
            # Lấy dynamic input tại thời điểm t
            x_t = dynamic_input[:, time_step, :, :, :]
            
            # Ghép dynamic với static (DEM, Manning lặp lại cho mỗi frame)
            # Input vào ConvLSTM sẽ có 4 channels
            combined_input = torch.cat([x_t, static_input], dim=1)
            
            h, c = self.conv_lstm(combined_input, (h, c))
        
        # --- DECODER ---
        outputs = []
        
        # Sau khi Encoder chạy xong, 'h' đang chứa thông tin tổng hợp của quá khứ
        # Ta dùng 'h' này để dự đoán tương lai.
        # Cách đơn giản nhất (Zero-input decoding): Giả sử input tương lai là 0 hoặc dùng lại h
        # Ở đây mình dùng chính output dự đoán làm input cho bước tiếp theo (Autoregressive)
        # Hoặc đơn giản hơn cho bản "Light": Chỉ chạy ConvLSTM tiếp tục với input rỗng/hoặc feature tĩnh
        
        # Chiến thuật Simple: Dùng trạng thái h hiện tại để predict, 
        # input cho bước tiếp theo chỉ là static features (vì ta ko biết mưa tương lai chính xác)
        
        current_prediction = self.final_conv(h) # Dự đoán frame t+1 đầu tiên
        
        for _ in range(future_steps):
            # Lưu dự đoán
            outputs.append(current_prediction)
            
            # Chuẩn bị input cho bước tiếp theo
            # Ta lấy chính dự đoán vừa rồi làm input giả lập cho "Past Depth"
            # Rain tương lai không biết -> giả sử = 0 hoặc giữ nguyên rain cuối (ở đây mình set 0 cho đơn giản)
            dummy_rain = torch.zeros_like(current_prediction) 
            
            # Ghép: [Dummy Rain, Predicted Depth, DEM, Manning]
            next_input = torch.cat([dummy_rain, current_prediction, static_input], dim=1)
            
            # Chạy ConvLSTM bước tiếp theo
            h, c = self.conv_lstm(next_input, (h, c))
            
            # Tạo dự đoán mới
            current_prediction = self.final_conv(h)

        # Xếp chồng các output theo chiều thời gian
        # Kết quả: (Batch, Future_Steps, 1, Height, Width)
        outputs = torch.stack(outputs, dim=1)
        return outputs

# --- TEST CODE (Chạy thử để check lỗi shape) ---
if __name__ == "__main__":
    # Giả lập dữ liệu
    # Batch = 2, Time_in = 12, H=128, W=128 (Lưu ý: Train nên crop ảnh nhỏ, 1073x1073 sẽ OOM)
    
    # Dynamic: Rain (1) + Past Depth (1) = 2 channels
    dummy_dynamic = torch.randn(2, 12, 2, 128, 128) 
    
    # Static: DEM (1) + Manning (1) = 2 channels
    dummy_static = torch.randn(2, 2, 128, 128)
    
    # Khởi tạo model
    model = FloodForecastModel(input_channels=4, hidden_channels=32, kernel_size=3)
    
    # Chạy model
    output = model(dummy_dynamic, dummy_static, future_steps=4)
    
    print(f"Input Dynamic Shape: {dummy_dynamic.shape}")
    print(f"Input Static Shape:  {dummy_static.shape}")
    print(f"Output Shape:        {output.shape}")
    
    # Check output dimension logic
    # Expect: (2, 4, 1, 128, 128)

# %% [code] {"execution":{"iopub.status.busy":"2025-11-23T15:15:08.937272Z","iopub.execute_input":"2025-11-23T15:15:08.937665Z","iopub.status.idle":"2025-11-23T15:15:14.587531Z","shell.execute_reply.started":"2025-11-23T15:15:08.937643Z","shell.execute_reply":"2025-11-23T15:15:14.586855Z"}}
!pip install rasterio

# %% [code] {"execution":{"iopub.status.busy":"2025-11-23T15:17:38.400357Z","iopub.execute_input":"2025-11-23T15:17:38.401033Z","iopub.status.idle":"2025-11-23T15:17:38.418555Z","shell.execute_reply.started":"2025-11-23T15:17:38.40101Z","shell.execute_reply":"2025-11-23T15:17:38.417858Z"}}
import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np

class FloodCastBenchDataset(Dataset):
    def __init__(self, root_dir, resolution='30m', seq_len_in=12, seq_len_out=4, patch_size=128, mode='train', val_ratio=0.2):
        """
        Phiên bản High-Fidelity: Sử dụng toàn bộ dữ liệu 5 phút/frame.
        """
        self.root_dir = root_dir
        self.seq_len_in = seq_len_in   
        self.seq_len_out = seq_len_out 
        self.patch_size = patch_size
        self.mode = mode
        
        # 1. Setup Paths
        self.input_dir = os.path.join(root_dir, 'Relevant data') # Kaggle path thường ko có underscore
        self.output_dir = os.path.join(root_dir, 'High-fidelity flood forecasting', resolution, 'Australia')
        
        # 2. Load Static Files (DEM & Manning)
        dem_path = os.path.join(self.input_dir, 'DEM', 'Australia_DEM.tif')
        manning_path = os.path.join(self.input_dir, 'Land use and land cover', 'Australia.tif')
        
        self.dem = self._read_tif(dem_path)
        self.manning = self._read_tif(manning_path)
        
        # 3. Indexing Files
        # Load toàn bộ list file Rain
        self.rain_files = sorted([
            os.path.join(self.input_dir, 'Rainfall/Australia flood', f) 
            for f in os.listdir(os.path.join(self.input_dir, 'Rainfall/Australia flood')) if f.endswith('.tif')
        ])
        
        # Load toàn bộ list file Flood (Output)
        self.flood_files_raw = sorted([
            f for f in os.listdir(self.output_dir) if f.endswith('.tif')
        ])
        print(self.output_dir)
        # 4. Xây dựng Dictionary: Time -> File Path
        # Để tra cứu cực nhanh
        self.flood_map = {} 
        for f in self.flood_files_raw:
            try:
                # Xử lý tên file: bỏ .tif, lấy số
                t_str = f.replace('.tif', '')
                # Một số file có prefix, ví dụ "frame_1800.tif", cần handle nếu có
                # Ở đây giả định chỉ là số
                time_sec = int(t_str)
                self.flood_map[time_sec] = os.path.join(self.output_dir, f)
            except ValueError:
                continue # Bỏ qua file không đúng định dạng số
        
        # Lấy danh sách các mốc thời gian có Flood data, sort tăng dần
        self.valid_times = sorted(list(self.flood_map.keys()))
        
        # --- 5. Tạo danh sách Samples hợp lệ ---
        # Một sample hợp lệ phải có đủ chuỗi quá khứ (in) và tương lai (out)
        # Và phải có rain tương ứng
        
        self.samples = [] # List chứa time_start của mỗi sample
        
        # Khoảng cách giữa các frame output (giả sử 5 phút = 300s)
        # Cần tự động detect step size nếu chưa biết
        if len(self.valid_times) > 1:
            self.dt = self.valid_times[1] - self.valid_times[0] # Thường là 300 (5 phút) hoặc 30 (30s)
        else:
            self.dt = 300 
            
        print(f"detected time step: {self.dt} seconds")

        for t in self.valid_times:
            # Kiểm tra xem có đủ sequence không
            # Sequence input: [t, t+dt, ..., t + (seq_in-1)*dt]
            # Sequence output: ... tiếp theo
            
            # Logic: t là thời điểm bắt đầu của chuỗi INPUT
            # Cần check sự tồn tại của toàn bộ chuỗi In + Out
            is_valid = True
            total_steps = seq_len_in + seq_len_out
            
            for i in range(total_steps):
                check_time = t + i * self.dt
                if check_time not in self.flood_map:
                    is_valid = False
                    break
                
                # Check thêm: Có file Rain tương ứng không?
                # Rain index = time // 1800
                rain_idx = check_time // 1800
                if rain_idx < 0 or rain_idx >= len(self.rain_files):
                    is_valid = False
                    break
            
            if is_valid:
                self.samples.append(t)

        # --- 6. TIME SPLIT (80/20) ---
        total_samples = len(self.samples)
        split_idx = int(total_samples * (1 - val_ratio))
        
        if self.mode == 'train':
            self.samples = self.samples[:split_idx]
            print(f"[Dataset-TRAIN] Samples: {len(self.samples)} (Step {self.dt}s)")
        else:
            self.samples = self.samples[split_idx:]
            print(f"[Dataset-VALID] Samples: {len(self.samples)} (Step {self.dt}s)")

    def _read_tif(self, path):
        with rasterio.open(path) as src:
            data = src.read(1)
            data = np.nan_to_num(data, nan=0.0)
            data = np.maximum(data, 0) 
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start_time = self.samples[idx]
        
        rain_seq = []
        depth_seq = []
        
        total_steps = self.seq_len_in + self.seq_len_out
        
        for i in range(total_steps):
            curr_time = start_time + i * self.dt
            
            # 1. Lấy Flood (Depth) - High Fidelity
            depth_path = self.flood_map[curr_time]
            depth_data = self._read_tif(depth_path)
            depth_seq.append(depth_data)
            
            # 2. Lấy Rain - Upsampling (Lặp lại)
            # Thời gian t -> Rain frame thứ (t // 1800)
            rain_idx = int(curr_time // 1800)
            rain_path = self.rain_files[rain_idx]
            rain_data = self._read_tif(rain_path)
            rain_seq.append(rain_data)
            
        rain_seq = np.array(rain_seq)
        depth_seq = np.array(depth_seq)
        
        # Chia Input/Target
        in_rain = rain_seq[:self.seq_len_in]      # (12, H, W)
        in_depth = depth_seq[:self.seq_len_in]    # (12, H, W)
        target_depth = depth_seq[self.seq_len_in:] # (4, H, W)
        
        # --- CROP & NORMALIZE ---
        if self.mode == 'train':
            H, W = self.dem.shape
            top = np.random.randint(0, H - self.patch_size)
            left = np.random.randint(0, W - self.patch_size)
            
            def crop(arr, is_3d=True):
                if is_3d: return arr[..., top:top+self.patch_size, left:left+self.patch_size]
                return arr[top:top+self.patch_size, left:left+self.patch_size]

            c_rain = crop(in_rain)
            c_in_depth = crop(in_depth)
            c_target = crop(target_depth)
            c_dem = crop(self.dem, is_3d=False)
            c_man = crop(self.manning, is_3d=False)
        else:
            # Valid -> Full size
            c_rain = in_rain
            c_in_depth = in_depth
            c_target = target_depth
            c_dem = self.dem
            c_man = self.manning

        # Normalize
        c_rain = c_rain / 50.0
        c_in_depth = c_in_depth / 5.0
        c_target = c_target / 5.0
        c_dem = (c_dem - np.min(c_dem)) / (np.max(c_dem) - np.min(c_dem) + 1e-6)
        
        # To Tensor
        inp_dynamic = torch.stack([torch.from_numpy(c_rain), torch.from_numpy(c_in_depth)], dim=1).float()
        inp_static = torch.stack([torch.from_numpy(c_dem), torch.from_numpy(c_man)]).float()
        target = torch.from_numpy(c_target).float().unsqueeze(1)
        
        return inp_dynamic, inp_static, target

# %% [code] {"execution":{"iopub.status.busy":"2025-11-23T15:15:15.266765Z","iopub.execute_input":"2025-11-23T15:15:15.267281Z","iopub.status.idle":"2025-11-23T15:15:15.273497Z","shell.execute_reply.started":"2025-11-23T15:15:15.267253Z","shell.execute_reply":"2025-11-23T15:15:15.27299Z"}}
import torch
import numpy as np

def compute_metrics(pred, target, thresholds=[0.01, 0.05]):
    """
    pred, target: Tensor (Batch, Time, 1, H, W)
    thresholds: list các ngưỡng độ sâu (mét). Vd: 0.01m (1cm), 0.05m (5cm)
    """
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    
    metrics = {}
    
    # 1. RMSE (Root Mean Squared Error)
    mse = np.mean((pred - target) ** 2)
    metrics['RMSE'] = np.sqrt(mse)
    
    # 2. NSE (Nash-Sutcliffe Efficiency)
    # NSE = 1 - (Sum(Obs - Sim)^2 / Sum(Obs - Mean_Obs)^2)
    numerator = np.sum((target - pred) ** 2)
    denominator = np.sum((target - np.mean(target)) ** 2) + 1e-6
    metrics['NSE'] = 1 - (numerator / denominator)
    
    # 3. Classification Metrics (CSI, F1) theo threshold
    for t in thresholds:
        # Binary masks
        pred_mask = (pred >= t)
        target_mask = (target >= t)
        
        TP = np.sum(pred_mask & target_mask)
        FP = np.sum(pred_mask & ~target_mask) # Báo động giả
        FN = np.sum(~pred_mask & target_mask) # Bỏ sót lũ
        
        # CSI = TP / (TP + FP + FN)
        csi = TP / (TP + FP + FN + 1e-6)
        metrics[f'CSI_{int(t*100)}cm'] = csi
        
        # F1 Score
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        metrics[f'F1_{int(t*100)}cm'] = f1
        
    return metrics

# %% [code] {"execution":{"iopub.status.busy":"2025-11-23T15:15:15.275213Z","iopub.execute_input":"2025-11-23T15:15:15.275625Z","iopub.status.idle":"2025-11-23T15:15:20.9474Z","shell.execute_reply.started":"2025-11-23T15:15:15.275606Z","shell.execute_reply":"2025-11-23T15:15:20.94647Z"}}
!pip install dataset

# %% [code] {"execution":{"iopub.status.busy":"2025-11-23T15:15:20.948845Z","iopub.execute_input":"2025-11-23T15:15:20.949558Z","iopub.status.idle":"2025-11-23T15:15:40.32497Z","shell.execute_reply.started":"2025-11-23T15:15:20.949529Z","shell.execute_reply":"2025-11-23T15:15:40.323865Z"}}
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from tqdm import tqdm

# --- IMPORT MODULES ---
# Đảm bảo bạn đã lưu các file này cùng thư mục
# from dataset import FloodCastBenchDataset # Class Dataset mới bạn vừa sửa
# from model import FloodForecastModel 
# from metrics import compute_metrics # Class Metrics mới đã fix lỗi chia 0

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Config cho Data High-Fidelity (5 phút/frame)
SEQ_LEN_IN = 12  # Nhìn 12 frame quá khứ
SEQ_LEN_OUT = 6  # Dự đoán 6 frame tương lai (Nếu step=5p -> Dự báo 30p tới)
                 # Nếu muốn dự báo 2h, cần tăng lên 24 (nhưng sẽ rất nặng)

PATCH_SIZE = 128 # Giữ 128 để nhẹ VRAM. 512 rất dễ OOM với ConvLSTM.
BATCH_SIZE = 16   # Batch cho Train (Patch nhỏ)
LR = 1e-4        # Giảm LR chút vì data dày đặc, cần học kỹ hơn
NUM_EPOCHS = 15
PATIENCE = 5     # Early Stopping
DATA_PATH = '/kaggle/input/d4l-data/FloodCastBench'
SAVE_DIR = './results'

# Tạo thư mục
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(f"{SAVE_DIR}/plots", exist_ok=True)

# --- CUSTOM LOSS FUNCTION ---
class WeightedMSELoss(nn.Module):
    """Phạt nặng hơn nếu dự đoán sai ở vùng có nước"""
    def __init__(self, weight=10.0):
        super().__init__()
        self.weight = weight
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        loss = self.mse(pred, target)
        # Mask: Những chỗ thực tế có nước (> 1cm)
        wet_mask = target > 0.01 
        
        # Tạo matrix trọng số
        weights = torch.ones_like(loss)
        weights[wet_mask] = self.weight # Nhân 10 lần loss tại vùng nước
        
        return torch.mean(loss * weights)

# --- UTILS ---
class EarlyStopping:
    def __init__(self, patience=5, path='best_model.pth'):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.path)
        elif val_loss > self.best_loss:
            self.counter += 1
            # print(f'   [EarlyStopping] Counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.path)
            self.counter = 0
            # print(f'   [CheckPoint] Model saved (Loss: {val_loss:.4f})')

def plot_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(15, 5))
    
    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train')
    plt.plot(epochs, history['val_loss'], label='Valid')
    plt.title('Weighted Loss')
    plt.legend()
    
    # CSI
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['csi_1cm'], label='CSI 1cm', color='orange')
    plt.plot(epochs, history['csi_5cm'], label='CSI 5cm', color='red')
    plt.title('CSI Score (Higher is Better)')
    plt.legend()

    # RMSE
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['rmse'], label='RMSE', color='green')
    plt.title('RMSE (Meters)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/history.png')
    plt.close()

def save_sample_prediction(model, loader, epoch):
    model.eval()
    with torch.no_grad():
        # Lấy 1 batch
        dyn, stat, target = next(iter(loader))
        dyn, stat = dyn.to(DEVICE), stat.to(DEVICE)
        
        # Predict
        output = model(dyn, stat, future_steps=SEQ_LEN_OUT)
        
        # Lấy frame cuối cùng của sample đầu tiên
        # output shape: (Batch, Time, 1, H, W)
        pred_img = output[0, -1, 0].cpu().numpy()
        target_img = target[0, -1, 0].cpu().numpy()
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(target_img, cmap='Blues', vmin=0, vmax=1.0)
        plt.title(f'Ground Truth (t+{SEQ_LEN_OUT})')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(pred_img, cmap='Blues', vmin=0, vmax=1.0)
        plt.title(f'Prediction (Epoch {epoch})')
        plt.colorbar()
        
        plt.savefig(f'{SAVE_DIR}/plots/epoch_{epoch}.png')
        plt.close()

# --- MAIN TRAINING LOOP ---
def train():
    print(f"=== START TRAINING on {DEVICE} ===")
    print(f"Config: In={SEQ_LEN_IN}, Out={SEQ_LEN_OUT}, Patch={PATCH_SIZE}")
    
    # 1. Data Setup
    # Train: Cắt nhỏ (Patch) để học nhanh + data augmentation
    train_ds = FloodCastBenchDataset(
        DATA_PATH, 
        resolution='30m', 
        seq_len_in=SEQ_LEN_IN, 
        seq_len_out=SEQ_LEN_OUT, 
        patch_size=PATCH_SIZE, 
        mode='train'
    )
    
    # Valid: Full ảnh (mode='valid') để đánh giá tổng thể
    valid_ds = FloodCastBenchDataset(
        DATA_PATH, 
        resolution='30m', 
        seq_len_in=SEQ_LEN_IN, 
        seq_len_out=SEQ_LEN_OUT, 
        patch_size=PATCH_SIZE, 
        mode='valid'
    )
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    # QUAN TRỌNG: Valid batch_size = 1 vì ảnh Full Size rất nặng
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=2)
    
    # 2. Model Setup
    # Input channels = 4 (Rain, PastDepth, DEM, Manning)
    model = FloodForecastModel(input_channels=4, hidden_channels=64, kernel_size=3).to(DEVICE)
    
    # Dùng Weighted Loss thay vì MSE thường
    criterion = WeightedMSELoss(weight=20.0).to(DEVICE) 
    optimizer = optim.Adam(model.parameters(), lr=LR)
    early_stopping = EarlyStopping(patience=PATIENCE, path=f'{SAVE_DIR}/best_model.pth')
    
    history = {'train_loss': [], 'val_loss': [], 'rmse': [], 'csi_1cm': [], 'csi_5cm': []}

    # 3. Loop
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # --- TRAIN ---
        model.train()
        train_loss = 0
        for dyn, stat, target in train_loader:
            dyn, stat, target = dyn.to(DEVICE), stat.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(dyn, stat, future_steps=SEQ_LEN_OUT)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALID ---
        model.eval()
        val_loss = 0
        metrics_sum = {'RMSE': 0, 'CSI_1cm': 0, 'CSI_5cm': 0, 'NSE': 0}
        
        with torch.no_grad():
            for dyn, stat, target in valid_loader:
                dyn, stat, target = dyn.to(DEVICE), stat.to(DEVICE), target.to(DEVICE)
                output = model(dyn, stat, future_steps=SEQ_LEN_OUT)
                
                loss = criterion(output, target)
                val_loss += loss.item()
                
                # Tính metrics
                batch_metrics = compute_metrics(output, target)
                for k in metrics_sum:
                    metrics_sum[k] += batch_metrics.get(k, 0)

        avg_val_loss = val_loss / len(valid_loader)
        
        # Average Metrics
        for k in metrics_sum:
            metrics_sum[k] /= len(valid_loader)
            
        # --- REPORT ---
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['rmse'].append(metrics_sum['RMSE'])
        history['csi_1cm'].append(metrics_sum['CSI_1cm'])
        history['csi_5cm'].append(metrics_sum['CSI_5cm'])
        
        print(f"Epoch {epoch+1} | "
              f"Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | "
              f"CSI 1cm: {metrics_sum['CSI_1cm']:.3f} | "
              f"CSI 5cm: {metrics_sum['CSI_5cm']:.3f} | "
              f"Time: {time.time() - start_time:.0f}s")
        
        plot_history(history)
        save_sample_prediction(model, valid_loader, epoch+1)
        
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("🛑 Early Stopping Triggered!")
            break

if __name__ == "__main__":
    train()

# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]
