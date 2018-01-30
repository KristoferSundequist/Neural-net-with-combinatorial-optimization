% num training examples
int: examples = 53;

%num features
int: features = 4;

% training examples
%array[1..examples, 1..features] of -10..10: trainingSet = [|6,3,4,-3|3,2,-2,1|1,6,1,1|3,1,7,1|1,7,7,2|3,1,7,1|6,2,-2,4|0,1,1,1|5,1,-3,1|2,7,-2,1|8,4,1,-6|-8,4,6,-9|8,-4,-6,-6|-2,1,4,1|-4,4,1,-7|-3,2,0,5|0,0,0,1|4,0,0,1|-5,-1,0,1|8,2,0,1|-3,3,0,1|4,0,0,1|2,-7,0,3|0,0,0,1|4,3,6,7|7,-3,3,5|];

array[1..examples, 1..features] of -10..10: trainingSet = [|6,3,4,-3|3,2,-2,1|1,6,1,1|3,1,7,1|1,7,7,2|3,1,7,1|6,2,-2,4|0,1,1,1|5,1,-3,1|2,7,-2,1|8,4,1,-6|-8,4,6,-9|8,-4,-6,-6|-2,1,4,1|-4,4,1,-7|-3,2,0,5|0,0,0,1|4,0,0,1|-5,-1,0,1|8,2,0,1|-3,3,0,1|4,0,0,1|2,-7,0,3|0,0,0,1|4,3,6,7|7,-3,3,5|3,2,7,1|5,3,2,1|5,5,5,5|1,-7,0,1|0,2,3,1|1,-3,6,-7|1,2,-5,-3|1,3,7,-4|10,8,9,-10|3,-5,2,-5|9,-6,-1,-5|-9,-10,-10,3|-1,-2,-6,4|-5,-3,1,-5|6,0,6,0|-9,1,10,1|1,-2,9,10|4,-7,-7,9|-7,10,3,5|-7,6,-7,10|4,2,-8,-4|8,10,7,8|-7,5,5,9|-1,6,8,-3|8,1,-2,2|7,5,-5,10|9,-2,-6,-8|];

% labels
%array[1..examples] of {0,1}: labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0,0];
%array[1..examples] of {0,1}: labels = [1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,1];

array[1..examples] of var {0,1}: labels;
%constraint forall(i in 1..examples)(labels[i] = if trainingSet[i,1]+trainingSet[i,2]+trainingSet[i,3]+trainingSet[i,4] >= 10 then 1 else 0 endif);
constraint forall(i in 1..examples)(labels[i] = if trainingSet[i,1]*trainingSet[i,1] + trainingSet[i,2]*trainingSet[i,2] + trainingSet[i,1]*trainingSet[i,2] - trainingSet[i,3]*trainingSet[i,4] >= 30 then 1 else 0 endif);

% num neurons in hidden layer
int: layerSize = 5;

% first bias
array[1..layerSize] of var -5 .. 5: bias1;

% first weights
array[1..features, 1..layerSize] of var -2 .. 2: w1;

% second bias
var -10 .. 10: bias2;

% second weights
array[1..layerSize] of var -3 .. 3: w2;


% values of hidden neurons
set of int: hidden_range = -100..100;
array[1..layerSize, 1..examples] of var hidden_range: hidden;

% hidden values equals features * weights + first bias under relu
constraint forall(l in 1..layerSize, e in 1..examples)(
  let { var hidden_range: result = sum(f in 1..features)(trainingSet[e,f]*w1[f,l]) + bias1[l]} in
    if result > 0 then hidden[l,e] = result else hidden[l,e] = 0 endif % relu
) :: bounds;

% outputs
set of int: output_range = -40..40;
array[1..examples] of var output_range: outputs;

% outputs (before activation) equals hidden values * second weights + second bias
constraint forall(e in 1..examples)(outputs[e] = sum(l in 1..layerSize)(hidden[l,e]*w2[l]) + bias2) :: bounds;

% redundant
constraint forall(e in 1..examples)(if labels[e] == 1 then outputs[e] > -10 else outputs[e] < 10 endif) :: bounds;

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
  int_search(bias1,first_fail, indomain_middle, complete),
  int_search([w1[f,l] | f in 1..features, l in 1..layerSize],first_fail, indomain_median, complete),
  int_search(w2,first_fail, indomain_median, complete)
  ])
  minimize cost;

%output[show(labels)];
%output["\n"];
%output[show(outputs)];
%output["\n"];
%output[show(cost)];