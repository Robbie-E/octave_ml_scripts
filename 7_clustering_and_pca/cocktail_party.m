% Cocktail party algorithm for 2 sources and 2 speakers
% Load two audio files with mixed sound signals
[x1, fs1] = audioread('mix1.wav');
[x2, fs2] = audioread('mix2.wav');

% X are the mixed microphone measurements (one per column)
X = [x1, x2]';

% Normalize the data
Y = sqrtm(inv(cov(X')))*(X-repmat(mean(X,2),1,size(X,2)));

% Independent component analysis using SVD
[W,s,v] = svd((repmat(sum(Y.*Y,1),size(Y,1),1).*Y)*Y');

% Recover the unmixed source signals, W is unmixing matrix
S = W*X;

% Plot the signals 
subplot(2,2,1); plot(x1); title('mixed audio - mic 1');
subplot(2,2,2); plot(x2); title('mixed audio - mic 2');
subplot(2,2,3); plot(S(1,:), 'g'); title('unmixed wave 1');
subplot(2,2,4); plot(S(2,:),'r'); title('unmixed wave 2');

audiowrite('unmixed1.wav', S(1,:), fs1);
audiowrite('unmixed2.wav', S(2,:), fs1);