clear 

%may want to consider leaving with negatives....hmmm
%% Loading Data
% Reading in Data 
train = readmatrix('zip.train', 'FileType', "text");
test = readmatrix('zip.test', 'FileType', "text");

%Removing first column; transpose (digits now go down a column)
A = train(:, 2:257);
A = A.';

%Remove first column; transpose (digits now go down a column)
B = test(:, 2:257);
B = B.';

%% Pre-Processing
%Each digit must be processed.
trainRange = size(A,2);
%Wanted to process B but it lowers accuracy SOMEHOW!?!?
testRange = size(B,2);

% Pre-processing Training(A)
for trainNum = 1:trainRange
    
    %Extracting the column digit at training number index; formating in R2
    e = A(:,trainNum);
    e = reshape(e,16,16);
    e = rot90(e,1);
    e = flipdim(e,1);
    
    %converting to a better grayscale type
    e = mat2gray(e,[-1 1]);
    %blurring the image using gaussian filter
    e = imgaussfilt(e,0.9);
    
    
    
    %Reshape back from matrix to vector form
    eVec = reshape(e,256,1);
    
    %Replace the smoothed vector values inplace
    A(:,trainNum) = eVec;
    
    disp("Training Digit Processed:" + trainNum)
    
end

%% Tangent Distance Prep

%This is how far into A, our training set will train
trainRange = 5000;
%This is how many we will test
testRange = 2000;

%Empty arrays. Used to store tangent matrices and calculate distances
tangentVectors = cell(1,trainRange);
tangentDistances = zeros(1,trainRange);

% Empty array to store the classified values
testedValues = zeros(1,testRange);
%% Training Digits

%Creating tangent vectors for training
for trainNum = 1:trainRange
    
    %Extracting the column digit at training number index; formating in R2
    e = A(:,trainNum);
    e = reshape(e,16,16);
    %Do not do this; redoing it once reshape back to R256 is an error:
    %e = rot90(e,1);
    %e = flipdim(e,1);
    
    %calculating a tangent matrix consisting of tangent vectors
    Te = createTangentMatrix(e);
    M{trainNum} = Te;
    
    disp("Trained:" + trainNum)
    
end

%% Classification

%For every testing digit, classify based on min distance...

for testNum = 1:testRange
    
    %Current tested digit
    pVec = B(:,testNum);
    p = reshape(pVec,16,16);
    p = rot90(p,1);
    p = flipdim(p,1);
    
    %converting to a better grayscale type
    p = mat2gray(p,[-1 1]);
    %blurring the imagel using gaussian filter
    p = imgaussfilt(p,0.9);
    
    %constructing tangent matrix for this test digit
    Tp = createTangentMatrix(p);
    
    %I reshape it back so that it retains the greyscale and blur
    p = reshape(p,256,1);
        
    %Calculating distance from p to every e in train
    for testingNum = 1:trainRange
        %extracting e and its tangent matrix
        e = A(:,testingNum);
        Te = M{testingNum};
        
        %storing the distances for a particular p versus every e
        tangentDistances(testingNum) = distance(p,Tp,e,Te);
    
    end
    
    [minDistance,minIndex] = min(tangentDistances);
    testValue = train(minIndex,1);
    testedValues(testNum) = testValue;
    disp("Tested:"+testNum)
    
end

%Here we compute our confusion matrix, since we computed our values
actualValues = test(1:testRange,1).';
C = confusionmat(actualValues,testedValues);
D = confusionchart(actualValues,testedValues);
D.RowSummary = 'row-normalized';
D.Title = 'Classification of Digits Using Tangent Distance'
accuracy = trace(C) / sum(sum(C))

%% Helper Methods

%Calculates the y derivative of an image by finite difference (center)
function Py = yDeriv(p)
    
    [m,n] = size(p);
    
    p1 = [p,zeros(m,1)];  % horizontal concatenation
    p1 = [zeros(m,1),p1];
    
    p1 = [p1;zeros(1,n+2)]; % vertical concatenation
    p = [zeros(1,n+2);p1];
    
    m = m+2;
    n = n+2;
    
    Py = zeros(m-2,n-2);
    for i=2:(m-1)
        for j=2:(n-1)
            Py(i-1,j-1) = (p(i+1,j) - p(i-1,j))/2;
        end
    end
    
end

%Calculates the x derivative of an image by finite difference (center)
function Px = xDeriv(p)
    
    [m,n] = size(p);
        
    p1 = [p,zeros(m,1)];  % horizontal concatenation
    p1 = [zeros(m,1),p1];
    
    p1 = [p1;zeros(1,n+2)]; % vertical concatenation
    p = [zeros(1,n+2);p1];
    
    m = m+2;
    n = n+2;
    
    Px = zeros(m-2,n-2);
    for i=2:(m-1)
        for j=2:(n-1)
            Px(i-1,j-1) = (p(i,j+1) - p(i,j-1))/2;
        end
    end
    
end

%The following function was borrowed to determine 97% 'energy' for the SVD
%See Jose Israel Pacheco (May 2011) for more information
function n = cumEnergy(D)
    totalE = sum(D.^2);
    n = 1;

    partialE = 0;
    while true
        partialE = partialE + D(n)^2;
    if partialE/totalE > .97
        break
    else
        n = n + 1;
    end
    end
end


%Forms tangent matrix
function tangentMatrix = createTangentMatrix(p)
    
   %p is in form of 16 by 16 at input
   
   %tangent derivatives using finite differences
   py = yDeriv(p);
   px = xDeriv(p);

   %ones matrix for below
   vals = ones(16,16);
   
   %Calculation of y for invariance transformations
   format3 = 1:-(1/16):(1/16);
   y = vals.*format3';

   %Calculation of x for invariance transformations
   x = vals.*[1:-(1/16):(1/16)];
   
   %format chooses our alphas in p +T_p*a
   left = (-1):(0.2):-0.2;
   right = (0.2):(0.2):1;
   format = [left right];
   
   %The following are the invariance transformations
   %p2 is an instance of a transformation of p under a given transform t
   
   %can improve with exact size and input, rather than appending
   basis = [];
   
   %x translation
   tangent = px;
   for a = format
       %calculate
       p2 = p + tangent*a;
       %reshape into R256 vector
       p2 = reshape(p2,256,1);
       %append
       basis = [basis p2(:)];
   end
   
   %y translation
   tangent = py;
   for a = format
       p2 = p + tangent*a;
       p2 = reshape(p2,256,1);
       basis = [basis p2(:)];
   end
   
   %rotation
   tangent = y.*px - x.*py;
   for a = format
       p2 = p + tangent*a;
       p2 = reshape(p2,256,1);
       basis = [basis p2(:)];
   end
   
   %thickening
   tangent = px.^2 + py.^2;
   for a = format
       p2 = p + tangent*a;
       p2 = reshape(p2,256,1);
       basis = [basis p2(:)];
   end
   
   %thickening
   tangent = px.^2 + py.^2;
   for a = format
       p2 = p + tangent*a;
       p2 = reshape(p2,256,1);
       basis = [basis p2(:)];
   end
   
   tangentMatrix = basis;
    
   %2.subtract p from every column of basis
   l = size(basis,2);
    
   %copies = repmat(p2,1,l);
   subtractedCols = basis - repmat(p2,1,l);
    
   %3.Take the SVD of the Matrix
   [UP,SP,~] = svd(subtractedCols);

   %4. Determine singular values to caputure x % of energy
   P_energy = cumEnergy(diag(SP));
    
   %5. Form the tangent basis
   tangentMatrix = UP(:,1:P_energy);
    
end


%The following function determines tangent distance (acquired from Jose Israel Pacheco (May 2011))
function tangentDistance = distance(p,tp,e,te)
    
    LEE = te'*te;
    LEP = te'*tp;
    LPE = tp'*te;
    LPP = tp'*tp;
    
    Bp = (LPE*inv(LEE)*te.' - tp.')*(e-p);
    Ap = LPE*inv(LEE)*LEP - LPP;
    ap = Ap\Bp;
    
    Be = (LEP*inv(LPP)*tp' - te')*(e-p);
    Ae = LEE - LEP*inv(LPP)*LPE;
    ae = Ae\Be;
    
    %Computation of the taylor approximation
    pCurve = p + tp*ap;
    eCurve = e + te*ae;
    
    %tangentDistance is the difference of the taylor approximations
    tangentDistance = norm(pCurve - eCurve)^2;
   
end