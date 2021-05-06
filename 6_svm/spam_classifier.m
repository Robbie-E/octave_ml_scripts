% Classify a given email, x, is spam (y=1) or non-spam (y=0).
% Use the body of the email (excluding the email headers)

% Extract Features
file_contents = readFile('emailSample1.txt');
% x(j) = index of jth word in email according to a dict
word_indices  = processEmail(file_contents);

% Print Stats
disp(word_indices)

% Extract Features, map to binary 
% x(j) = 1 if jth word in dict is in the email
features = emailFeatures(word_indices);

% Print Stats
% Expect feature vector had length 1899 and 45 non-zero entries
fprintf('Length of feature vector: %d\n', length(features));
fprintf('Number of non-zero entries: %d\n', sum(features > 0));

% spamTrain.mat contains 4000 training examples of spam and non-spam email
% spamTest.mat contains 1000 test examples.
% training accuracy ~99.8% and a test accuracy ~98.5%

% Load the Spam Email dataset
% You will have X, y (training set) in your environment
load('spamTrain.mat');
C = 0.1;
model = svmTrain(X, y, C, @linearKernel);
p = svmPredict(model, X);
fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);

% Load the test dataset
% You will have Xtest, ytest in your environment
load('spamTest.mat');

p = svmPredict(model, Xtest);
fprintf('Test Accuracy: %f\n', mean(double(p == ytest)) * 100);

% Sort the weights and obtain the vocabulary list
% Determine words that are top predictors of spam
[weight, idx] = sort(model.w, 'descend');
vocabList = getVocabList();
for i = 1:15
    if i == 1
        fprintf('Top predictors of spam: \n');
    end
    fprintf('%-15s (%f) \n', vocabList{idx(i)}, weight(i));
end