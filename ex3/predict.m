function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
M = zeros(size(X, 1), 1);  %matrix that contains the max value of h(x)...
                           %... in each training sample, added by naufal

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Added by naufal

% Now you will implement feedforward propagation for the neural network. 
% You will need to complete the code in predict.m to return the neural network's prediction. 
% You should implement the feedforward computation that computes  for every example  
% and returns the associated predictions. Similar to the one-vs-all classication strategy, 
% the prediction from the neural network will be the label that has the largest output .

% Implementation Note: The matrix X contains the examples in rows. When you complete the code in predict.m, 
% you will need to add the column of 1's to the matrix. The matrices Theta1 and Theta2 contain the parameters 
% for each unit in rows. Specically, the first row of Theta1 corresponds to the first hidden unit in the second layer. 
% In MATLAB, when you compute z(2)=theta(1) * a(1), be sure that you index (and if necessary, transpose) X correctly 
% so that you get a(1) as a column vector.


%calculate 10 outputs
a1=[ones(m, 1) X];  % Add ones to the X data matrix
z1=a1*Theta1';
a2=sigmoid(z1);

a2=[ones(m, 1) a2];  % Add ones to the X data matrix
z2=a2*Theta2';
a3=sigmoid(z2);


%pick index with max value for each training sample
[M,p] = max(a3,[],2);


% =========================================================================


end
