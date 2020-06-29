function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for i=1:size(X,1)
    idx(i)=1;
    %min_distance = sum(((X(i,:)-centroids(1,:))).^2);
    min_distance = (X(i,:)-centroids(1,:))*((X(i,:)-centroids(1,:))');
    % ^ another way to compute distance
    
    for k=2:K
        %distance = sum(((X(i,:)-centroids(k,:))).^2);
        distance = (X(i,:)-centroids(k,:))*((X(i,:)-centroids(k,:))');
        % ^ another way to compute distance
        if distance < min_distance
            idx(i)=k;
            min_distance = distance;
        end
    end
end


%another implementation (not yet checked)
%for i=1:size(X,1)
%    distance= zeros(K, 1);
%    for k=1:K
%        distance(k) = sum(((X(i,:)-centroids(k,:))).^2);
%    end
%    [M,idx(i)]=min(distance);
%end


% =============================================================

end

