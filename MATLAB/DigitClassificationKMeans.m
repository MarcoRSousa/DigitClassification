%%%K-Means Method - US Postal

%%The following requires a supervised Method of K-means

%We read in our data
train = readmatrix('zip.train', 'FileType', "text");
test = readmatrix('zip.test', 'FileType', "text");

%Cutting out the last column and first digit
A = train(:, 2:257);

%Creating test without the first digit
B = test(:, 2:257);

%Performing kmeans with k=10
rand('seed', 1)
[guess,centroid] = kmeans(A, 10);

%We unfortunately have to supervise which letter corresponds to which
%centroid
% 1-6, 2-8, 3-0,4-2,5-9,6-1,7-6, 8-7,9-0,10-3

guessRecognition = [6 8 0 2 9 1 6 7 0 3].';

centroid = cat(2, guessRecognition, centroid);

centroidError = zeros(1,10);
%Create an array where the predicted values of the test are stored.
N = size(B,1);
predictedValues = zeros(N,1);


%For each test value, perform a check.
%Determines least distance 2norm from each centroid
%Then picks centroid with least distance
for n = 1:N
    
    %taking distance from each centroid for a digit n
    for k = 1:10
        centroidError(k) = norm(  B(n,:) - centroid(k, 2:257));
    end
    
    
    %Takes lowest error in distance from centroid
    [minError,minIndex] = min(centroidError);
    
    % Determines value for that index
    predictedValues(n) = centroid(minIndex, 1);
    
    
end

%Actual digit values
actualValues = test(:,1);

%A confusion matrix compares the values of the actual versus the predicted
C = confusionmat(actualValues,predictedValues)

%Creating a Confusion Matrix Chart to display as a figure
D = confusionchart(actualValues,predictedValues);
D.RowSummary = 'row-normalized';
D.Title = 'Classification of Digits Using K-Means'

% The accuracy can be represented by the trace/total
accuracy = trace(C) / sum(sum(C))



