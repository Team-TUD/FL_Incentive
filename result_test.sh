python3  federated_learning_inputation.py 0.0 [] > result_inputation_0_0.txt
python3  federated_learning_reputation.py 0.0 [] > result_reputation_0_0.txt
python3  federated_learning_shapley.py 0.0 []  > result_shapley_0_0.txt
for attacker in [1] [1,2] [1,2,3]
do
  for noise in 0.1 0.3 0.5 0.7 1
  do
    echo $noise $attacker ${attacker: -2:1} result_inputation_$noise_${attacker: -2:1}.txt
    python3  ./federated_learning_influence.py $noise $attacker > result_inputation_$noise_${attacker: -2:1}.txt
    python3  ./federated_learning_reputation.py $noise $attacker > result_reputation_$noise_${attacker: -2:1}.txt
    python3  ./federated_learning_shapley.py $noise $attacker  > result_shapley_$noise_${attacker: -2:1}.txt
  done
done
