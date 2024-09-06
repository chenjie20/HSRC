function [Z, W] = sp_representation(X, lambda, sparse_mode)

n = size(X, 2);
W = zeros(n, n);
affine = false;
if sparse_mode == 0
    W = admmLasso_mat_func(X, affine, lambda);
else
    for idx = 1 : n
        % 1. Initaliz data samples.
        y = X(:,idx);
        if idx == 1
            Y = X(:,idx + 1 : end);
        elseif ((idx > 1) && (idx < n))
            Y = [X(:,1 : idx - 1) X(:, idx + 1 : n)];        
        else
            Y = X(:,1: n - 1);
        end

        %2. Sparse representation
         cvx_begin quiet;
            cvx_precision high
            variable c(n-1,1);
            minimize( norm(c,1) );
            subject to
            norm(Y * c  - y) <= lambda;
        cvx_end;         
        if idx == 1   
            W(1, 1) = 0;
            W(2 : n, 1) = c(1 : n - 1);       
        elseif ( (idx > 1) && (idx < n) )
            W(1 : idx - 1, idx) = c(1 : idx - 1);
            W(idx, idx) = 0;
            W(idx + 1 : n, idx) = c(idx : n - 1);
        else
            W(1 : n - 1, n) = c(1 : n - 1);
            W(n, n) = 0;
        end    
    end
end
    
% sign_nan = find(any(isnan(W), 1) > 1e-6);
% if ~isempty(sign_nan)
%     W(sign_nan,:) = [];
%     W(:, sign_nan) = [];
%     ground_lables(:, sign_nan) = [];
% end
Z = (abs(W) + abs(W')) / 2;

