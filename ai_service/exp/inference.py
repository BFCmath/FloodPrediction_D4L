# %% [code] {"execution":{"iopub.status.busy":"2025-11-26T03:24:54.498768Z","iopub.execute_input":"2025-11-26T03:24:54.499012Z","iopub.status.idle":"2025-11-26T03:25:01.484922Z","shell.execute_reply.started":"2025-11-26T03:24:54.498992Z","shell.execute_reply":"2025-11-26T03:25:01.484127Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2025-11-26T03:25:01.48631Z","iopub.execute_input":"2025-11-26T03:25:01.486669Z","iopub.status.idle":"2025-11-26T03:25:08.483126Z","shell.execute_reply.started":"2025-11-26T03:25:01.486652Z","shell.execute_reply":"2025-11-26T03:25:08.482483Z"}}
!pip install rasterio

# %% [code] {"execution":{"iopub.status.busy":"2025-11-26T03:25:08.484011Z","iopub.execute_input":"2025-11-26T03:25:08.484275Z","iopub.status.idle":"2025-11-26T03:25:09.338239Z","shell.execute_reply.started":"2025-11-26T03:25:08.484239Z","shell.execute_reply":"2025-11-26T03:25:09.337721Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2025-11-26T03:25:09.338989Z","iopub.execute_input":"2025-11-26T03:25:09.339393Z","iopub.status.idle":"2025-11-26T04:08:27.139697Z","shell.execute_reply.started":"2025-11-26T03:25:09.339374Z","shell.execute_reply":"2025-11-26T04:08:27.138917Z"}}
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# --- 1. CẤU HÌNH CÁC MỐC THỜI GIAN ---
# Time step của dữ liệu là 5 phút (300s)
TIME_STEP_MINUTES = 5

# Dictionary các mốc cần check: {Tên: Số step}
HORIZONS = {
    '5 min': 5 // TIME_STEP_MINUTES,   # 6 steps
    '10 min': 10 // TIME_STEP_MINUTES, # 24 steps
    '30 min': 30 // TIME_STEP_MINUTES, # 48 steps
    '60 min': 60 // TIME_STEP_MINUTES, # 72 steps
    '120 min': 120 // TIME_STEP_MINUTES,# 144 steps
}

MAX_STEPS = max(HORIZONS.values()) # 288 steps
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = '/kaggle/input/convlstm/results/best_model.pth' # Đường dẫn file model đã lưu

# --- 2. HÀM TÍNH METRIC (TÁI SỬ DỤNG) ---
def compute_metrics_single_step(pred, target, thresholds=[0.01, 0.05]):
    """Tính metric cho 1 khung hình cụ thể tại thời điểm t"""
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    
    metrics = {}
    mse = np.mean((pred - target) ** 2)
    metrics['RMSE'] = np.sqrt(mse)
    
    for t in thresholds:
        pred_mask = (pred >= t)
        target_mask = (target >= t)
        TP = np.sum(pred_mask & target_mask)
        FP = np.sum(pred_mask & ~target_mask)
        FN = np.sum(~pred_mask & target_mask)
        csi = TP / (TP + FP + FN + 1e-6)
        metrics[f'CSI_{int(t*100)}cm'] = csi
    return metrics

# --- 3. MAIN EVALUATION SCRIPT ---
def evaluate_long_term():
    print(f"=== ĐANG CHUẨN BỊ DỮ LIỆU KIỂM TRA DÀI HẠN (MAX {MAX_STEPS} STEPS) ===")
    
    # Load lại Dataset với seq_len_out đủ dài (288) để có Ground Truth so sánh
    # Lưu ý: Việc này có thể làm giảm số lượng sample (do nhiều chuỗi không đủ độ dài 24h)
    valid_ds = FloodCastBenchDataset(
        '/kaggle/input/d4l-data/FloodCastBench', # Sửa lại path data của bạn nếu cần
        resolution='30m', 
        seq_len_in=12, 
        seq_len_out=MAX_STEPS, # Quan trọng: Load đủ 24h tương lai
        patch_size=128,        # Có thể tăng lên nếu VRAM cho phép, hoặc để nguyên
        mode='valid'
    )
    
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=2)
    print(f"-> Số lượng mẫu Valid đủ điều kiện test 24h: {len(valid_loader)}")

    # Load Model
    print(f"=== LOADING MODEL TỪ {MODEL_PATH} ===")
    model = FloodForecastModel(input_channels=4, hidden_channels=64, kernel_size=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Lưu trữ kết quả tổng hợp
    results = {h_name: {'RMSE': [], 'CSI_1cm': [], 'CSI_5cm': []} for h_name in HORIZONS}

    print("=== BẮT ĐẦU CHẠY INFERENCE ===")
    with torch.no_grad():
        for i, (dyn, stat, target) in enumerate(tqdm(valid_loader)):
            dyn, stat, target = dyn.to(DEVICE), stat.to(DEVICE), target.to(DEVICE)
            
            # Chạy model autoregressive tới 288 bước
            # Model sẽ tự lấy output t làm input cho t+1
            predictions = model(dyn, stat, future_steps=MAX_STEPS) 
            # Output shape: (Batch, 288, 1, H, W)

            # Check từng mốc thời gian
            for h_name, step_idx in HORIZONS.items():
                # step_idx tính từ 1, index mảng tính từ 0 -> lấy step_idx - 1
                idx = step_idx - 1
                
                # Lấy frame tại thời điểm đó
                pred_t = predictions[:, idx, :, :, :]
                target_t = target[:, idx, :, :, :]
                
                # Tính metric
                m = compute_metrics_single_step(pred_t, target_t)
                
                results[h_name]['RMSE'].append(m['RMSE'])
                results[h_name]['CSI_1cm'].append(m['CSI_1cm'])
                results[h_name]['CSI_5cm'].append(m['CSI_5cm'])

            # (Optional) Visualize 1 mẫu đầu tiên để kiểm tra
            if i == 0:
                visualize_long_term(predictions, target, HORIZONS)

    # --- 4. BÁO CÁO KẾT QUẢ ---
    print("\n" + "="*60)
    print(f"{'HORIZON':<15} | {'RMSE (m)':<10} | {'CSI 1cm':<10} | {'CSI 5cm':<10}")
    print("-" * 60)
    
    final_report = {}
    for h_name in HORIZONS:
        rmse_avg = np.mean(results[h_name]['RMSE'])
        csi1_avg = np.mean(results[h_name]['CSI_1cm'])
        csi5_avg = np.mean(results[h_name]['CSI_5cm'])
        
        print(f"{h_name:<15} | {rmse_avg:.4f}     | {csi1_avg:.4f}     | {csi5_avg:.4f}")
        final_report[h_name] = (rmse_avg, csi1_avg, csi5_avg)
    print("="*60)

def visualize_long_term(preds, targets, horizons):
    """Vẽ ảnh so sánh tại các mốc thời gian"""
    # Chọn các mốc chính để vẽ cho đỡ rối: 30p, 6h, 12h, 24h
    plot_points = ['5 min', '10 min', '30 min', '60 min', '120 min']
    
    plt.figure(figsize=(12, 6))
    plt.suptitle("Long-term Flood Forecast Visualization", fontsize=16)
    
    cols = len(plot_points)
    for idx, h_name in enumerate(plot_points):
        step = horizons[h_name] - 1
        
        # Lấy ảnh về CPU
        p = preds[0, step, 0].detach().cpu().numpy()
        t = targets[0, step, 0].detach().cpu().numpy()
        
        # Ground Truth (Hàng trên)
        plt.subplot(2, cols, idx + 1)
        plt.imshow(t, cmap='Blues', vmin=0, vmax=1.0)
        plt.axis('off')
        if idx == 0: plt.ylabel("Ground Truth", fontsize=12)
        plt.title(f"{h_name}")
        
        # Prediction (Hàng dưới)
        plt.subplot(2, cols, idx + 1 + cols)
        plt.imshow(p, cmap='Blues', vmin=0, vmax=1.0)
        plt.axis('off')
        if idx == 0: plt.ylabel("Prediction", fontsize=12)

    plt.tight_layout()
    import os
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/long_term_eval.png")
    plt.show()

if __name__ == "__main__":
    # Đảm bảo bạn đã import các class FloodForecastModel, FloodCastBenchDataset ở trên
    evaluate_long_term()

# %% [code]
