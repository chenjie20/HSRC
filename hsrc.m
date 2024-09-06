function [purtiy, fmeasure, W] = hsrc(X, ground_lables, lambda, K, n_times, sparse_mode)

num_clusters_merged = 0;

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
    
sign_nan = find(any(isnan(W), 1) > 1e-6);
if ~isempty(sign_nan)
    W(sign_nan,:) = [];
    W(:, sign_nan) = [];
    ground_lables(:, sign_nan) = [];
end
Z = (abs(W) + abs(W')) / 2;
    

k = n_times * K;
    

            
%     for cluster_idx = 1 : 4
%         switch cluster_idx
%             case 1
%                 actual_ids = sparse_spectral_clustering(L, D, k);  % Max + Sparse             
%             case 2
%                 actual_ids = spectral_clustering(L, k); % Max
%             case 3
%                 actual_ids = sparse_spectral_clustering_a(L, D, k); % Min + Sparse
%             case 4
%                 actual_ids = spectral_clustering_a(L, k); % Min  
%         end
        
actual_ids = spectral_clustering(Z, k); % Max
mirco_clusters = cell(k, 1);
    
%4. Merge mirco-clusters

% 4.1 Initalize
for idx = 1 : k
    mirco_clusters(idx, 1) = { find(actual_ids == idx)};         
end
mirco_clusters_merged = mirco_clusters;

 % 4.2 merge micro-clusters
 current_num_clusters_merged = k;     
 merged_continue = true;
 while merged_continue % check the stop condition         
     % 4.2.1 find out micro-clusters merged
     num_clusters_merged = 0;
     sign_merged = zeros(current_num_clusters_merged, current_num_clusters_merged);
     for idx = 1 : current_num_clusters_merged
         micro_clusters_idx = find_micro_clusters_merged(idx, mirco_clusters_merged, X, X, W, Z);
         if micro_clusters_idx > 0
            sign_merged(idx, micro_clusters_idx) = 1;
         end             
     end  
%          sign_merged = (sign_merged + sign_merged') / 2;
%          disp(sign_merged);
    % 4.2.2 merge mirco-clusters into a relatively large cluster
    merged_pos = zeros(current_num_clusters_merged, 1);
    current_mirco_clusters_merged = cell(current_num_clusters_merged , 1);
    pos = 1;
    for idx = 1 : current_num_clusters_merged
        if isempty(find(merged_pos == idx, 1))
            num_clusters_merged = num_clusters_merged + 1;
            row_sign =  find(sign_merged(idx, :) > 1e-6);
            col_sign = find(sign_merged(:, idx) > 1e-6);
             intersection_pos = intersect(row_sign, col_sign);
%                 if idx == 1
%                     intersection_pos = [2];
%                 end
             if ~isempty(intersection_pos)                     
                 % merge
                 major_cluster_tmp = mirco_clusters_merged(idx);
                 major_cluster = major_cluster_tmp{1};
                 for inter_idx = 1 : length(intersection_pos)
                     another_cluster_tmp = mirco_clusters_merged(intersection_pos(inter_idx));
                     another_cluster = another_cluster_tmp{1};
                     major_cluster = union(major_cluster, another_cluster);
                 end
                 current_mirco_clusters_merged(num_clusters_merged) = { major_cluster };
                 merged_pos(pos) = idx;
                 pos = pos + 1;
                 for pos_idx = 1 : length(intersection_pos)
                     merged_pos(pos) = intersection_pos(pos_idx);
                     pos = pos + 1;
                 end
%                      intersection_pos = [];
             else
                   current_mirco_clusters_merged(num_clusters_merged) = mirco_clusters_merged(idx);    
             end
         end
     end
     
     % 4.2.3 set the stop condition
     if(current_num_clusters_merged == num_clusters_merged) 
         merged_continue = false;
     else
         if num_clusters_merged == 2
             merged_continue = false;
         end
         current_num_clusters_merged =  num_clusters_merged;
         mirco_clusters_merged = cell(num_clusters_merged, 1);
         for idx = 1 : num_clusters_merged 
            mirco_clusters_merged(idx) = current_mirco_clusters_merged(idx);
         end
     end     
 end
     
     % 5 fine tune
%      new_actual_ids =  zeros(num, 1);
%      for idx = 1 : num_clusters_merged
%         current_mirco_cluster_tmp = mirco_clusters_merged(idx);
%         current_mirco_cluster = current_mirco_cluster_tmp{1};
%         if ~isempty(current_mirco_cluster)
%             for c_idx = 1 : length(current_mirco_cluster)
%                 current_cluster_idx = idx;
%                 sample_idx = current_mirco_cluster(c_idx);
%                 sample = X(:, sample_idx);
%                 %final_error = 0;               
%                 for fine_idx = 1 : num_clusters_merged 
%                     if c_idx ~= fine_idx
%                         fine_mirco_cluster_tmp = mirco_clusters_merged(fine_idx);
%                         fine_mirco_cluster = fine_mirco_cluster_tmp{1};
%                         w = W(:, sample_idx);
%                         other_lables = setdiff(ground_lables, fine_mirco_cluster);
%                         final_error = norm(sample - X(:, other_lables) * w(other_lables, 1));
%                         %                     current_error = length(find(abs(w(fine_mirco_cluster, 1)) > 1e-6));                      
%                         current_error = norm(sample - X(:, fine_mirco_cluster) * w(fine_mirco_cluster, 1));
% %                         if final_error < 1e-6
% %                             final_error = current_error;
% %                         elseif current_error > final_error
% %                             final_error = current_error;
% %                         current_cluster_idx = fine_idx;
% %                         end
%                         if current_error < final_error
%                             current_cluster_idx = fine_idx;
%                         end
%                     end
%                 end
%                 new_actual_ids(sample_idx) = current_cluster_idx;
%             end
%         end
%      end
%     final_mirco_clusters = cell(k, 1);
%     for idx = 1 : k
%         final_mirco_clusters(idx, 1) = { find(new_actual_ids == idx)};
%     end
     
     % 6 get the clustering accuracy
    total_cluster_num = 0;
    score_results = zeros(num_clusters_merged, 1);
    clustering_results = zeros(num_clusters_merged, 1);
    for idx = 1 : num_clusters_merged
        middle_mirco_cluster_tmp = mirco_clusters_merged(idx);
        middle_mirco_cluster = middle_mirco_cluster_tmp{1};
        if ~isempty(middle_mirco_cluster)
            table = tabulate(ground_lables(middle_mirco_cluster));            
            [clustering_results(idx), row_idx] = max(table(:,3));              
            total_cluster_num = total_cluster_num + table(row_idx, 2);
            
            current_cluster_num = table(row_idx, 2);
            wnd_num = length(find(ground_lables == table(row_idx, 1)));
            recall = current_cluster_num / wnd_num;
            precision = clustering_results(idx) / 100;
            score_results(idx) = 2 * (precision * recall) / (precision + recall);        
            
        end
    end
    
   fmeasure = mean(score_results(score_results > 1e-6));     
   purtiy = mean(clustering_results(clustering_results > 1e-6));

end
