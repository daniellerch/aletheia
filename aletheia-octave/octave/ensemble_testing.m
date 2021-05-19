function results = ensemble_testing(X,trained_ensemble)
% -------------------------------------------------------------------------
% Ensemble Classification | June 2013 | version 2.0 | TESTING ROUTINE
% -------------------------------------------------------------------------
% INPUT:
%  - X - testing features (in a row-by-row manner)
%  - trained_ensemble - trained ensemble - cell array of individual base
%              learners (output of the 'ensemble_training' routine)
% OUTPUT:
%  - results.predictions - individual cover (-1) and stego (+1) predictions
%              based on the majority voting scheme
%  - results.votes - sum of all votes (gives some information about
%              prediction confidence)
% -------------------------------------------------------------------------
% Please see the main routine 'ensemble_training' for more information.
% -------------------------------------------------------------------------
% Copyright (c) 2013 DDE Lab, Binghamton University, NY.
% All Rights Reserved.
% -------------------------------------------------------------------------
% Permission to use, copy, modify, and distribute this software for
% educational, research and non-profit purposes, without fee, and without a
% written agreement is hereby granted, provided that this copyright notice
% appears in all copies. The program is supplied "as is," without any
% accompanying services from DDE Lab. DDE Lab does not warrant the
% operation of the program will be uninterrupted or error-free. The
% end-user understands that the program was developed for research purposes
% and is advised not to rely exclusively on the program for any reason. In
% no event shall Binghamton University or DDE Lab be liable to any party
% for direct, indirect, special, incidental, or consequential damages,
% including lost profits, arising out of the use of this software. DDE Lab
% disclaims any warranties, and has no obligations to provide maintenance,
% support, updates, enhancements or modifications.
% -------------------------------------------------------------------------
% Contact: jan@kodovsky.com | fridrich@binghamton.edu | June 2013
%          http://dde.binghamton.edu/download/ensemble
% -------------------------------------------------------------------------
% References:
% [1] - J. Kodovsky, J. Fridrich, and V. Holub. Ensemble classifiers for
% steganalysis of digital media. IEEE Transactions on Information Forensics
% and Security. Currently under review.
% -------------------------------------------------------------------------

% simple majority voting scheme
votes = zeros(size(X,1),1);
for i = 1:length(trained_ensemble)
    proj = X(:,trained_ensemble{i}.subspace)*trained_ensemble{i}.w-trained_ensemble{i}.b;
    votes = votes+sign(proj);
end

results.votes_ = votes;

% resolve ties randomly
votes(votes==0) = rand(sum(votes==0),1)-0.5;
% form final predictions
results.predictions = sign(votes);
% output also the sum of the individual votes (~confidence info)
results.votes = votes;
