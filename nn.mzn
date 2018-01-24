% abs error function
function var int: absError(var int: prediction, var int: label) = abs(prediction-label);

% squared error function
function var int: squaredError(var int: prediction, var int: label) = let {var int: err = prediction-label} in err*err;

% num training examples
int: examples = 23;

%num features
int: features = 4;

% training examples
array[1..examples, 1..features] of int: trainingSet = [|6,3,4,-3|3,2,-2,1|1,6,1,1|3,1,7,1|1,7,7,2|3,1,7,1|6,2,-2,4|0,1,1,1|5,1,-3,1|2,7,-2,1|8,4,1,-6|-8,4,6,-9|8,-4,-6,-6|-2,1,4,1|-4,4,1,-7|-3,2,0,5|0,0,0,1|4,0,0,1|-5,-1,0,1|8,2,0,1|-3,3,0,1|4,0,0,1|2,-7,0,3|];

% labels
array[1..examples] of var int: labels;

% function to approximate
constraint forall(i in 1..examples)(labels[i] = trainingSet[i,1]*trainingSet[i,1] + trainingSet[i,2]*trainingSet[i,2] + trainingSet[i,1]*trainingSet[i,2] - trainingSet[i,3]*trainingSet[i,4]);

% num neurons in hidden layer
int: layerSize = 10;

% first bias
array[1..layerSize] of var -15 .. 15: bias1;

% first weights
array[1..features, 1..layerSize] of var -10 .. 10: w1;

% second bias
var -15 .. 15: bias2;

% second weights
array[1..layerSize] of var -15 .. 15: w2;


% values of hidden neurons
array[1..layerSize, 1..examples] of var -20..20: hidden;

% hidden values equals features * weights + first bias under relu
constraint forall(l in 1..layerSize, e in 1..examples)(
  let { var int: result = sum(f in 1..features)(trainingSet[e,f]*w1[f,l]) + bias1[l]} in
    if result > 0 then hidden[l,e] = result else hidden[l,e] = 0 endif % relu
);

% outputs
array[1..examples] of var int: outputs;

% outputs equals hidden values * second weights + second bias
constraint forall(e in 1..examples)(outputs[e] = sum(l in 1..layerSize)(hidden[l,e]*w2[l]) + bias2);

% symmetry break
constraint symmetry_breaking_constraint(exists(n in 1..features)(forall(i,j in 1..layerSize where i = j-1)(w1[n,i] <= w1[n,j])));
constraint symmetry_breaking_constraint(exists(n in 1..features)(exists(i,j in 1..layerSize where i != j)(w1[n,i] < w1[n,j])));
constraint symmetry_breaking_constraint(forall(i,j in 1..features where i = j-1)(min(v in 1..layerSize)(w1[i,v]) < min(v in 1..layerSize)(w1[j,v])));

% cost function is sum of squared distance between outputs and labels
var 0..100000: cost;
constraint cost = sum(e in 1..examples)(squaredError(outputs[e], labels[e]));

solve minimize cost;

%output[show(labels)];
%output["\n"];
%output[show(outputs)];
%output["\n"];
%output[show(cost)];