function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
             
   %added by naufal : Theta1 = hidden_layer_size x input_layer_size + 1 (25 x 401) matrix
   %                : Theta2 = num_labels x hidden_layer_size + 1 (10 x 26) matrix


% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% From here and below added by naufal

% Part 1
    
%calculate outputs
a1=[ones(m, 1) X];  % Add ones to the X data matrix , a1 = 5000x401
z2=a1*Theta1';      % z2 = (5000x401) x (25x401)' = 5000x25
a2=sigmoid(z2);     % a2 = 5000x25

a2=[ones(m, 1) a2];  % Add ones to the X data matrix, 5000x26
z3=a2*Theta2';       % z3 = (5000x26) x (10x26)' = 5000x10
a3=sigmoid(z3);      % a3 = 5000x10

%recode y into m x K matrix
ybin = zeros(m,num_labels);
for i=1:m
    ybin(i,1:num_labels) = [1:num_labels] == y(i);  %ybin = 5000x10
end

%calculate cost(unregularized)
J=(-1/m)*sum(sum(ybin.*log(a3)+(1-ybin).*(log(1-a3))));

%adding regularization algorithm
J=J + (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));


% Part 2 (backpropagation)

%VECTOR IMPLEMENTATION
delta3 = (a3-ybin)';    %delta3 = 10x5000
delta2 = (delta3'*Theta2(:,2:end).*sigmoidGradient(z2))'; %delta2 = 25x5000
Theta1_grad = (delta2*a1)/m;  %(25x5000) x (5000x401) = 25x401 -> MATCH!! 
Theta2_grad = (delta3*a2)/m;  %(10x5000) x (5000x26) = 10x26 -> MATCH!!

% Part 3 (regularization)
Theta1_grad_regular = (lambda/m) * Theta1(:,2:end);
Theta2_grad_regular = (lambda/m) * Theta2(:,2:end);

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + Theta1_grad_regular;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + Theta2_grad_regular;


%Part 2 LOOP IMPLEMENTATION (method instructed in video)
%gradient error is high (1.88e-05), cannot pass online grader,
%not recommended by moderator in the forum.
%
%for t=1:m
%  %assign a_1 as x(t) transpose
%  a_1=X(t,:)'; %a_1 input_layer_sizex1 (400x1) column vector
%    
%  %step1 : forward propragation
%  a_1=[1; a_1];   % Add bias unit, a_1 is (input_layer_size + 1) x 1 or 401x1 column vector
%  z_1=Theta1*a_1;  % z_1 = (25x401) x (401x1) = 25x1 column vector
%  a_2=sigmoid(z_1);  % a_2 25x1 column vector
%
%  a_2=[1; a_2];  % Add bias unit, a_2 is (hidden_layer_size + 1) x 1 or 26x1 column vector
%  z_2=Theta2*a_2;  % z_2 = (10x26) x (26x1) = 10x1 column vector
%  a_3=sigmoid(z_2);  % a_3 10x1 column vector
%  
%  %step2 : calculating delta in the output node
%  y_bin = ybin(t,:)';   %y_bin is 10 x 1 column vector
%  delta_3 = a_3 - y_bin;   %delta_3 is 10 x 1 column vector
%  %implementation using for loop :
%  %for k=1:num_labels
%  %  delta_3(k)=a_3(k)-ybin(t,k);   %delta3 = 10x1 column vector
%  %endfor
%  
%  %step3 : calculating delta in the hidden layer(s), recall that no delta for input layer
%  delta_2=Theta2'*(delta_3.*sigmoidGradient(z_2)); 
%  delta_2=delta_2(2:end);
%  %delta_2 = 25x1 vector (perhitungan dimensi tertulis di buku tulis)
%  
%  %step4 : obtaining the accumulation of gradient
%  Theta1_grad = Theta1_grad + (delta_2*(a_1'));   %Theta1_grad is 25 x 401 matrix
%  Theta2_grad = Theta2_grad + (delta_3*(a_2'));   %Theta1_grad is 10 x 26 matrix
%end
%
%%step4 : obtaining unregularized gradient by dividing gradient accumulation by m
%Theta1_grad = Theta1_grad/m;
%Theta2_grad = Theta2_grad/m;








% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
