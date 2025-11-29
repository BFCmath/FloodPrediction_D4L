Do không có data của việt nam, nên mình sẽ chọn Australia (30m) nhé

Input:
DEM
Cái này hiểu đơn giản là file địa hình
Nó chứa 2 file: tif và tfw

Tif hiểu nhanh là 1 cái ảnh HxW, với các pixel là độ cao z - elevation in meters (real-world ground height above sea level)
File tfw giúp cái ảnh về scale về GPS thực tế

> each DEM pixel is the height of the terrain at that 30m × 30m ground cell.

Rainfall 
Unit: mm/hour
Each pixel value = rainfall intensity at that location in millimeters per hour
Khoảng cách giữa 2 file rainfall là 30 minutes

Initial conditions = High-fidelity_flood_forecasting/.../0.tif.

Land_use_and_land_cover  -Each pixel stores Manning’s n coefficient

Output resolutions 30m vs 60m — what is the difference?
Folder
Pixel size
Grid size
Detailed level
30m
30m × 30m
1073×1073
High detail
60m
60m × 60m
536×536
Coarse

Flood time (t=0, t=1800…) — compared to what?
t=0 flood depth corresponds to rainfall frame rainfall_000.tif
Khoảng cách giữa rainfall là 30p, còn của output là 5p


Each pixel in output

Each pixel represents the Water Depth relative to the land surface.


Input:
DEM (terrain height in meters)
Manning map (roughness/friction)
Rainfall time series (mm/hour) (last 12 frames) (last 6 hours)

(or can even try 24 frames…)

Output:
Predict the next 4*6=24 frames (next 2 hours)

Có thể dài quá thì đo metric trên next 5 minutes, next 10 minutes, next 30 minutes, next 1 hour and next 2 hours
Tức là predict the 1st frame, 2nd frame, 6th frames, 12th frames and 24th frames

Metric:

1. RMSE (Root Mean Squared Error)
2. NSE (Nash–Sutcliffe Efficiency) - Widely used in hydraulic modeling.
3. CSI (Critical Success Index) - Checks if you predicted the correct area to be flooded. It penalizes "False Alarms" (predicting flood where there is none) and "Misses" (failing to warn).
4. F1 score (đo luôn precision vs recall)



Để tính đc cái CSI vs F1, thì mình set threshold cho cái output nhé, theo tui research thì thường ngta để 1cm hoặc 5cm, nên mình tính CSI_1cm vs CSI_5cm luôn nhé


# TASK
Đọc hiểu phần  trên, và nam ro input và output là gi