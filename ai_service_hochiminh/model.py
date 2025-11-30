import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
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
    def __init__(self, input_channels=4, hidden_channels=64, kernel_size=3):
        super(FloodForecastModel, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv_lstm = ConvLSTMCell(input_channels, hidden_channels, kernel_size, bias=True)
        self.final_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1) 

    def forward(self, dynamic_input, static_input, future_steps=4):
        """
        dynamic_input: (Batch, Time_in, Channels=2, Height, Width) -> Rain + Past Depth
        static_input:  (Batch, Channels=2, Height, Width) -> DEM + Manning
        """
        b, t, c, h, w = dynamic_input.size()
        h_state, c_state = self.conv_lstm.init_hidden(b, (h, w))
        
        # ENCODER
        for time_step in range(t):
            x_t = dynamic_input[:, time_step, :, :, :]
            combined_input = torch.cat([x_t, static_input], dim=1)
            h_state, c_state = self.conv_lstm(combined_input, (h_state, c_state))
        
        # DECODER
        outputs = []
        current_prediction = self.final_conv(h_state) 
        
        for _ in range(future_steps):
            outputs.append(current_prediction)
            dummy_rain = torch.zeros_like(current_prediction) 
            next_input = torch.cat([dummy_rain, current_prediction, static_input], dim=1)
            h_state, c_state = self.conv_lstm(next_input, (h_state, c_state))
            current_prediction = self.final_conv(h_state)

        outputs = torch.stack(outputs, dim=1)
        return outputs



