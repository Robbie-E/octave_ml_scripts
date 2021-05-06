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

% Try other examples provided
examples = cellstr(['emailSample1.txt'; 'emailSample2.txt'; 'spamSample1.txt'; 'spamSample2.txt']);
for email_file = 1:length(examples)
    file_contents = readFile(examples{email_file});
    word_indices  = processEmail(file_contents);
    features = emailFeatures(word_indices);
    svmPredict(model, features)
end