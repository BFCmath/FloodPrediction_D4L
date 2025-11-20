@echo off
REM ============================================================================
REM Run All FloodCastBench Visualization Scripts
REM ============================================================================

echo.
echo ========================================================================
echo FloodCastBench Data Visualization Suite
echo ========================================================================
echo.

REM Create outputs directory if it doesn't exist
if not exist "outputs" mkdir outputs

echo [1/6] Visualizing DEM (Digital Elevation Model)...
python 01_visualize_dem.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to run 01_visualize_dem.py
    pause
    exit /b 1
)
echo.

echo [2/6] Visualizing Rainfall Time Series...
python 02_visualize_rainfall.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to run 02_visualize_rainfall.py
    pause
    exit /b 1
)
echo.

echo [3/6] Visualizing Manning's Roughness Coefficient...
python 03_visualize_manning.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to run 03_visualize_manning.py
    pause
    exit /b 1
)
echo.

echo [4/6] Visualizing Initial Conditions...
python 04_visualize_initial_conditions.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to run 04_visualize_initial_conditions.py
    pause
    exit /b 1
)
echo.

echo [5/6] Visualizing Flood Simulation Output...
python 05_visualize_flood_output.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to run 05_visualize_flood_output.py
    pause
    exit /b 1
)
echo.

echo [6/6] Converting Coordinates to Google Maps...
python 06_convert_to_google_maps.py
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Script 06 may require additional packages (pyproj, folium)
    echo Install with: pip install pyproj folium rasterio
)
echo.

echo ========================================================================
echo ALL VISUALIZATIONS COMPLETE!
echo ========================================================================
echo.
echo Generated files are in the 'outputs' folder:
echo   - 01_dem_visualization.png
echo   - 02_rainfall_visualization.png
echo   - 03_manning_visualization.png
echo   - 04_initial_conditions_visualization.png
echo   - 05_flood_output_30m_visualization.png
echo   - 06_google_map_interactive.html  ^<-- Open in browser!
echo   - 06_dem_coverage.kml             ^<-- Open in Google Earth!
echo   - 06_dem_coverage.geojson
echo.
echo Opening outputs folder...
start "" "outputs"
echo.
pause
