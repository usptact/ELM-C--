function N = elmScale( A )
% FUNCTION scales data to -1;+1 range.
%

% get column max and min
maxA = max( A, [], 2 );
minA = min( A, [], 2 );

maxA = repmat( maxA, 1, size(A,2) );
minA = repmat( minA, 1, size(A,2) );

% normalize to -1...1
N = ( (A-minA)./(maxA-minA) - 0.5 ) *2;
