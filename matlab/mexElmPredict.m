% Predicts class scores using a trained ELM model.
%
%      scores = mexElmPredict( inW, bias, outW, X );
%
% INPUT :
%      inW     - input weights matrix (trained model)
%      bias    - bias vector (trained model)
%      outW    - output weights matrix (trained model)
%      X       - samples matrix (vectors in columns)
%
% OUTPUT :
%      scores  - prediction scores for samples in matrix X
%