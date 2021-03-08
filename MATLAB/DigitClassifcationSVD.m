clear

%%%SVD Method
%The following is an SVD method of classifying digits.
%The greatest Accuracy I was able to do with this method was ~94.5%
%Many random subsets of the data had similar accuracy (implied stable)

%Furthermore, this method uses a svd with k singular values
%Thus, the real-time calculation is quite small, compared to other 
%Classifaction methods like Tangent distance (at the cost of some accuracy)

%%Bringing in Data

%We read in our data
train = readmatrix('zip.train', 'FileType', "text");
test = readmatrix('zip.test', 'FileType', "text");

%Cutting out the last column; rename for convinience
A = train(:, 1:257);

%%Training

%Choosing Asub allows us to choose a training set.
% I chose to take 4014 random digits from the set

%This increases accuracy sometimes because some digits are written poorly
%And by chance those digits are isolated out
k = randperm(7291);
Asub = A(k(1:4014),:);

%Take the transpose so the digits align the columns, rather than rows
A = Asub.';
%A = A.'; %If we choose the whole data set

%Cut the first column that identifies the values for classification and
%transpose similarly
B = test(:, 2:257);
B = B.';

%Boolean comprehension to isolate/partition each digit into its own matrix for SVD
%If many  digits, the following can be GENERALIZED with a cell array.

%A0,A1,A2...A9 represent the submatrices for respective 0...9 digit.
index = A(1, :) == 0;
A0 = A(2:257, index);
index = A(1, :) == 1;
A1 = A(2:257, index);
index = A(1, :) == 2;
A2 = A(2:257, index);
index = A(1, :) == 3;
A3 = A(2:257, index);
index = A(1, :) == 4;
A4 = A(2:257, index);
index = A(1, :) == 5;
A5 = A(2:257, index);
index = A(1, :) == 6;
A6 = A(2:257, index);
index = A(1, :) == 7;
A7 = A(2:257, index);
index = A(1, :) == 8;
A8 = A(2:257, index);
index = A(1, :) == 9;
A9 = A(2:257, index);


%We want a thin decomposition of u1...uk with k singular values.
k = 10;
[U0, S0, V0] = svds(A0,k);
[U1, S1, V1] = svds(A1,k);
[U2, S2, V2] = svds(A2,k);
[U3, S3, V3] = svds(A3,k);
[U4, S4, V4] = svds(A4,k);
[U5, S5, V5] = svds(A5,k);
[U6, S6, V6] = svds(A6,k);
[U7, S7, V7] = svds(A7,k);
[U8, S8, V8] = svds(A8,k);
[U9, S9, V9] = svds(A9,k);


%%Classification

%Set an array that the residuals will be stored into
residualError = zeros(1,10);
%Create an array where the predicted values of the test are stored.
N = size(B,2);
predictedValues = zeros(1,N);

for n = 1:N
    
    %Cuts out a column to test against for cols n to N 
    singleTest = B(:,n);
    
    %Stores into residual array. Can be generalized with cell array
    residualError(1) = norm(singleTest - U0*U0.' * singleTest);
    residualError(2) = norm(singleTest - U1*U1.' * singleTest);
    residualError(3) = norm(singleTest - U2*U2.' * singleTest);
    residualError(4) = norm(singleTest - U3*U3.' * singleTest);
    residualError(5) = norm(singleTest - U4*U4.' * singleTest);
    residualError(6) = norm(singleTest - U5*U5.' * singleTest);
    residualError(7) = norm(singleTest - U6*U6.' * singleTest);
    residualError(8) = norm(singleTest - U7*U7.' * singleTest);
    residualError(9) = norm(singleTest - U8*U8.' * singleTest);
    residualError(10) = norm(singleTest - U9*U9.' * singleTest);
    
    %Finds minimum residual and index in which it occurs
    [minResidual,minIndex] = min(residualError);
    
    %The index is 1 greater than the predicted value (index 1 maps to 0)
    %Stores in predictions array.
    predictedValue = minIndex - 1;
    predictedValues(n) = predictedValue;
    
end

%Extracting out the actual test data
actualValues = test(:,1).';

%A confusion matrix compares the values of the actual versus the predicted
C = confusionmat(actualValues,predictedValues)

%Creating a Confusion Matrix Chart to display as a figure
D = confusionchart(actualValues,predictedValues);
D.RowSummary = 'row-normalized';
D.Title = 'Classification of Digits Using SVD'

% The accuracy can be represented by the trace of the confusion/total
accuracy = trace(C) / sum(sum(C))
