@echo off
:: ResNet Stage2 Training Batch Script
:: Automatically sets environment variables and starts training

echo =================================================
echo ResNet Stage2 Behavior Classification Training
echo =================================================

:: Set OpenMP environment variables
set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=1

echo Environment variables set:
echo   KMP_DUPLICATE_LIB_OK=%KMP_DUPLICATE_LIB_OK%
echo   OMP_NUM_THREADS=%OMP_NUM_THREADS%
echo   MKL_NUM_THREADS=%MKL_NUM_THREADS%
echo.

:: Check if arguments are provided
if "%1"=="" (
    echo Starting interactive training...
    python run_resnet_training.py
) else if "%1"=="test" (
    echo Running quick test...
    python run_resnet_training.py test
) else if "%1"=="debug" (
    echo Running debug mode...
    python train_resnet_stage2.py --backbone resnet18 --batch_size 4 --epochs 3 --frame_interval 10
) else if "%1"=="resnet18" (
    echo Running ResNet18 training...
    python train_resnet_stage2.py --backbone resnet18 --batch_size 16 --epochs 50
) else if "%1"=="resnet18_frozen" (
    echo Running ResNet18 frozen training...
    python train_resnet_stage2.py --backbone resnet18 --freeze_backbone --batch_size 24 --lr 1e-3
) else if "%1"=="resnet50" (
    echo Running ResNet50 training...
    python train_resnet_stage2.py --backbone resnet50 --visual_dim 512 --batch_size 8 --lr 5e-5
) else (
    echo Running custom config: %1
    python run_resnet_training.py %1
)

echo.
echo Training completed!
pause