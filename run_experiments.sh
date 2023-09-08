python ./experiments/prepare_date.py -i $1 -o $2;

python ./experiments/train.py --train_dir $2/all/train --valid_dir $2/all/valid --experiments_dir ./results --experiment_name all;
python ./experiments/eval.py --test_dir $2/all/test --model_path ./results/all/models/all --output_dir ./results/all;
python ./experiments/eval.py --test_dir $2/all/valid --model_path ./results/all/models/all --output_dir ./results/all;
python ./experiments/eval.py --test_dir $2/all/train --model_path ./results/all/models/all --output_dir ./results/all;

python ./experiments/train.py --train_dir $2/fed/train --valid_dir $2/fed/valid --experiments_dir ./results --experiment_name fed;
python ./experiments/eval.py --test_dir $2/fed/test --model_path ./results/fed/models/fed --output_dir ./results/fed;
python ./experiments/eval.py --test_dir $2/fed/valid --model_path ./results/fed/models/fed --output_dir ./results/fed;
python ./experiments/eval.py --test_dir $2/fed/train --model_path ./results/fed/models/fed --output_dir ./results/fed;

python ./experiments/train.py --train_dir $2/unfed/train --valid_dir $2/unfed/valid --experiments_dir ./results --experiment_name unfed;
python ./experiments/eval.py --test_dir $2/unfed/test --model_path ./results/unfed/models/unfed --output_dir ./results/unfed;
python ./experiments/eval.py --test_dir $2/unfed/valid --model_path ./results/unfed/models/unfed --output_dir ./results/unfed;
python ./experiments/eval.py --test_dir $2/unfed/train --model_path ./results/unfed/models/unfed --output_dir ./results/unfed;