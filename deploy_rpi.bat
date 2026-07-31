@echo off
cd /d "D:\Esiot_project"
echo Deploying to RPi 5...
echo.

pscp -pw 123 -hostkey "SHA256:nwMhw+R7h0wn6iSC2zjGhJpJ4//V5pO9ydtB1wVryq8" SHASHANK/scripts/rpi_predict.py shashank@192.168.29.66:/home/shashank/Desktop/esiot/rpi_predict.py
if errorlevel 1 (
  echo FAILED - RPi might be offline. Try again when connected.
  pause
  exit /b 1
)
echo rpi_predict.py updated!

pscp -pw 123 -hostkey "SHA256:nwMhw+R7h0wn6iSC2zjGhJpJ4//V5pO9ydtB1wVryq8" SHASHANK/scripts/dashboard.py shashank@192.168.29.66:/home/shashank/Desktop/esiot/dashboard.py
if errorlevel 1 (echo dashboard.py failed) else (echo dashboard.py updated!)

echo.
echo Done! SSH into RPi and run:
echo   pkill -f rpi_predict
echo   cd ~/Desktop/esiot ^&^& python rpi_predict.py
echo.
pause
