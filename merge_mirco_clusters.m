function [mirco_clusters_merged, num_clusters_merged] = merge_mirco_clusters(X, W, Z, K, m, enable_finetuning)


    k = K * m;
    actual_ids = spectral_clustering(Z, k);
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
    % enable_finetuing: 0 disable, 1 enable
    if enable_finetuning == 1
        n = size(W, 1);
        new_actual_ids =  zeros(n, 1);
        mirco_clusters = cell(1, num_clusters_merged);
        for idx = 1 : num_clusters_merged
            mirco_clusters(1, idx) = mirco_clusters_merged(idx);
        end
        for idx = 1 : num_clusters_merged
            current_mirco_cluster_tmp = mirco_clusters_merged(idx);
            current_mirco_cluster = current_mirco_cluster_tmp{1};
            if ~isempty(current_mirco_cluster)                
                for c_idx = 1 : length(current_mirco_cluster)
                    current_cluster_idx = idx;
                    sample_idx = current_mirco_cluster(c_idx);
                    sample = X(:, sample_idx);
                    w = W(:, sample_idx);
                    final_error = norm(sample - X(:, current_mirco_cluster) * w(current_mirco_cluster, 1));               
                    for fine_idx = 1 : num_clusters_merged 
                        if idx ~= fine_idx
                            fine_mirco_cluster_tmp = mirco_clusters_merged(fine_idx);
                            fine_mirco_cluster = fine_mirco_cluster_tmp{1};                                               
                            current_error = norm(sample - X(:, fine_mirco_cluster) * w(fine_mirco_cluster, 1));
                            if current_error < final_error
                                current_cluster_idx = fine_idx;
                            end
                        end
                    end
                    new_actual_ids(sample_idx) = current_cluster_idx;                    
                end
            end
        end
        final_mirco_clusters = cell(k, 1);
        for idx = 1 : k
            final_mirco_clusters(idx, 1) = { find(new_actual_ids == idx)};
        end
        mirco_clusters_merged = final_mirco_clusters;
    end
    
end

