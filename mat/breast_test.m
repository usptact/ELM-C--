load breast.mat

%% cross-validation

if 0
	
	nhn_list = 1 : 10;
	C = 0.001;
	
	[trn vld] = cross_valid_ids( size(X,2), 10, [0.9 0.1 0] );
	
	perf_valid = zeros( length(nhn_list), length(nhn_list) );
	
	for hid = 1 : length(nhn_list)
		
		nhn = nhn_list(hid);
		
		for fid = 1 : 10
			
			i = trn( :, fid );
			j = vld( :, fid );
			
			p = zeros(10,1);
			for rid = 1 : 10
				[inW bias outW] = mexElmTrain( X(:,i), Y(i), nhn, C );
				scores = mexElmPredict( inW, bias, outW, X(:,j) );
				[~,Yhat] = max( scores, [], 1 );
				p(rid) = sum( Yhat(:) == Y(j) ) / length(j);
			end
			
			perf_valid(fid,hid) = mean(p);
			
		end
		
	end
	
	surf( perf_valid );
	
end

%% performance prediction

nhn = 5;
C = 1;

[trn tst] = cross_valid_ids( size(X,2), 10, [0.9 0.1 0] );

perf_test = zeros( 1, 10 );

for fid = 1 : 10
	
	i = trn( :, fid );
	j = tst( :, fid );
	
	p = zeros(10,1);
	for rid = 1 : 10
		[inW bias outW] = mexElmTrain( X(:,i), Y(i), nhn, C );
		scores = mexElmPredict( inW, bias, outW, X(:,j) );
		[~,Yhat] = max( scores, [], 1 );
		p(rid) = sum( Yhat(:) == Y(j) ) / length(j);
	end
	
	perf_test(fid) = mean(p);
	
end
