# python main.py --cfg configs/mimic4/mimic4_outcome_gru_ep100_kf0_bs64_hid128.yaml --train --cuda 3

path='configs/mimic4/'
# 路径下的所有文件
files=$(ls $path)
for file in $files
do
    echo $file
    python main.py --cfg $path$file --train --cuda 3
done