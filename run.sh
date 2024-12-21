python main.py --name baron_mouse --n_runs 3 --sym False
python main.py --name pancreas --n_runs 3
python main.py --name mouse_bladder --n_runs 3
python main.py --name zeisel --n_runs 3 --sym False
python main.py --name worm_neuron --n_runs 3 --sym False
python main.py --name baron_human --n_runs 3
for drop_rate in 0.2 0.4 0.8
do
python main.py --name baron_mouse --drop_rate $drop_rate --n_runs 3 --gamma 1
done
for drop_rate in 0.2 0.4 0.8
do
python main.py --name pancreas --drop_rate $drop_rate --n_runs 3 --gamma 1
done
for drop_rate in 0.2 0.4 0.8
do
python main.py --name mouse_bladder --drop_rate $drop_rate --n_runs 3 --gamma 1
done
for drop_rate in 0.2 0.4 0.8
do
python main.py --name zeisel --drop_rate $drop_rate --n_runs 3 --gamma 1
done
for drop_rate in 0.2 0.4 0.8
do
python main.py --name worm_neuron --drop_rate $drop_rate --n_runs 3 --gamma 1
done
for drop_rate in 0.2 0.4 0.8
do
python main.py --name baron_human --drop_rate $drop_rate --n_runs 3 --gamma 1
done