

% num training examples
int: examples = 25;

%num features
int: features = 4;

% training examples
array[1..examples, 1..features] of -10..10: trainingSet = [|6,3,4,-3|3,2,-2,1|1,6,1,1|3,1,7,1|1,7,7,2|3,1,7,1|6,2,-2,4|0,1,1,1|5,1,-3,1|2,7,-2,1|8,4,1,-6|-8,4,6,-9|8,-4,-6,-6|-2,1,4,1|-4,4,1,-7|-3,2,0,5|0,0,0,1|4,0,0,1|-5,-1,0,1|8,2,0,1|-3,3,0,1|4,0,0,1|2,-7,0,3|0,0,0,1|4,3,6,7|];

% labels
array[1..examples] of {0,1}: labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0,0];


% num neurons in hidden layer
int: layerSize = 3;

% first bias
array[1..layerSize] of var -50 .. 50: bias1;

% first weights
array[1..features, 1..layerSize] of var -50 .. 50: w1;

% second bias
var -50 .. 50: bias2;

% second weights
array[1..layerSize] of var -50 .. 50: w2;


% values of hidden neurons
set of int: hidden_range = -200..200;
array[1..layerSize, 1..examples] of var hidden_range: hidden;

% hidden values equals features * weights + first bias under relu
constraint forall(l in 1..layerSize, e in 1..examples)(
  let { var hidden_range: result = sum(f in 1..features)(trainingSet[e,f]*w1[f,l]) + bias1[l]} in
    if result > 0 then hidden[l,e] = result else hidden[l,e] = 0 endif % relu
) :: bounds;

% outputs
set of int: output_range = -200..200;
array[1..examples] of var output_range: outputs;

% outputs (before activation) equals hidden values * second weights + second bias
constraint forall(e in 1..examples)(outputs[e] = sum(l in 1..layerSize)(hidden[l,e]*w2[l]) + bias2) :: bounds;

% predictions are the outputs under a binary activation
array[1..examples] of var {0,1}: predictions;
constraint forall(e in 1..examples)(predictions[e] = bool2int(outputs[e] > 0)) :: bounds;

% cost function sum of incorrect predictions
var 0..examples: cost;
constraint cost = sum(e in 1..examples)(bool2int(predictions[e] != labels[e])) :: bounds;


% symmetry break
constraint symmetry_breaking_constraint(exists(n in 1..features)(forall(i,j in 1..layerSize where i = j-1)(w1[n,i] <= w1[n,j]))) :: bounds;
constraint symmetry_breaking_constraint(exists(n in 1..features)(exists(i,j in 1..layerSize where i != j)(w1[n,i] < w1[n,j]))) :: bounds;
constraint symmetry_breaking_constraint(forall(i,j in 1..features where i = j-1)(min(v in 1..layerSize)(w1[i,v]) < min(v in 1..layerSize)(w1[j,v]))) :: bounds;

solve 
  :: seq_search([
  int_search(bias1,first_fail, indomain_median, complete),
  int_search([w1[f,l] | f in 1..features, l in 1..layerSize],first_fail, indomain_median, complete),
  int_search(w2,first_fail, indomain_median, complete)
  ])
  minimize cost;

%output[show(labels)];
%output["\n"];
%output[show(outputs)];
%output["\n"];
%output[show(cost)];