% Trains an ELM model on the provided dataset.
%
%      [inW bias outW] = mexElmTrain( X, Y [, nhn, C ] );
%
% INPUT :
%      X       - samples matrix (samples in columns)
%      Y       - labels vector
%      nhn     - number of hidden neurons (default: dims / 2)
%      C       - regularization parameter (default: 1)
%
% OUTPUT :
%      inW     - output weights matrix
%      bias    - bias vector
%      outW    - output weights matrix
%