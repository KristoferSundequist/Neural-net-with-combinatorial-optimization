function var int: absError(var int: prediction, var int: label) = abs(prediction-label);

int: examples = 18;
int: features = 2;
array[1..examples, 1..features] of int: trainingSet = [|6,3|3,2|1,6|3,1|1,7|3,1|6,2|0,1|5,1|2,7|8,4|-8,4|8,-4|-2,1|-4,4|-3,2|0,0|4,0|];
array[1..examples] of var int: labels;
% function to approx
%constraint forall(i in 1..examples)(labels[i] = trainingSet[i,1]*trainingSet[i,1] + trainingSet[i,2]*trainingSet[i,2]*trainingSet[i,2]);
constraint forall(i in 1..examples)(labels[i] = trainingSet[i,1]*trainingSet[i,1] + trainingSet[i,2]*trainingSet[i,2] + trainingSet[i,1]*trainingSet[i,2]);

int: varBounds = 10;

int: layerSize = 3;
array[1..layerSize] of var -varBounds .. varBounds: bias1;
array[1..features, 1..layerSize] of var -varBounds .. varBounds: w1;

var -varBounds .. varBounds: bias2;
array[1..layerSize] of var -varBounds .. varBounds: w2;


% hidden
array[1..layerSize, 1..examples] of var 0..10: hidden;
constraint forall(l in 1..layerSize, e in 1..examples)(
  let { var int: result = sum(f in 1..features)(trainingSet[e,f]*w1[f,l]) + bias1[l]} in
    if result > 0 then hidden[l,e] = result else hidden[l,e] = 0 endif % relu
);

% outputs
array[1..examples] of var int: outputs;
constraint forall(e in 1..examples)(outputs[e] = sum(l in 1..layerSize)(hidden[l,e]*w2[l]) + bias2);

% symmetry break
constraint exists(n in 1..features)(forall(i,j in 1..layerSize where i = j-1)(w1[n,i] <= w1[n,j]));

constraint exists(n in 1..features)(exists(i,j in 1..layerSize where i != j)(w1[n,i] < w1[n,j]));

constraint forall(i,j in 1..features where i = j-1)(min(v in 1..layerSize)(w1[i,v]) < min(v in 1..layerSize)(w1[j,v]));

% extra prop
int: bound = 10;
constraint forall(i in 1..examples)(absError(outputs[i], labels[i]) < bound); 

var int: cost;
constraint cost = sum(e in 1..examples)(absError(outputs[e], labels[e]));
%solve satisfy;

solve minimize cost;