@echo off
rem Train the time-branch separator (Windows). Resumes from the latest
rem compatible checkpoint in .\ckpts. Set FRESH=1 to start a new run.
rem   set DATA_DIRS=D:\data\train1,D:\data\train2
rem   set DATA_WEIGHTS=3,1
rem   scripts\train.bat
setlocal
if "%DATA_DIRS%"=="" set DATA_DIRS=train1,train2
if "%DATA_WEIGHTS%"=="" set DATA_WEIGHTS=3,1
set EXTRA=
if "%FRESH%"=="1" set EXTRA=--fresh
pushd "%~dp0..\main"
python mvsep.py --train ^
  --data_dirs "%DATA_DIRS%" --data_weights "%DATA_WEIGHTS%" ^
  --model_dim 256 --depth 16 --heads 8 ^
  --segment_seconds 8 --inference_overlap_seconds 2 ^
  --batch_size 1 --grad_accumulation 1 --num_workers 8 ^
  --dataset_size 50000 --remix_probability 0.5 ^
  --checkpoint_steps 8000 --weight_decay 0.0 --grad_clip 1.0 ^
  --ema_decay 0.999 --precision bf16 --ckpt %EXTRA% %*
popd
endlocal
