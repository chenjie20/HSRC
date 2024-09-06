function [ micro_clusters_merged_idx ] = find_micro_clusters_merged(micro_cluster_idx, mirco_clusters, data, Dp, W, L)
        
    num_micro_clusters = numel(mirco_clusters);
%     similarity_between_clusters =  zeros(num_micro_clusters, num_micro_clusters);
%     for row_idx = 1 : num_micro_clusters
%         for col_idx = 1 : num_micro_clusters
%             if row_idx ~= col_idx
%                 row = mirco_clusters(col_idx);
%                 rows = row{1, 1};
%                 col = mirco_clusters(row_idx);
%                 cols = col{1, 1};
%                 similarity_between_clusters(row_idx, col_idx) =  sum(sum(L(rows, cols)));
%             end
%         end
%     end
    
    micro_clusters_merged_idx = 0;  
    cluster_tmp = mirco_clusters(micro_cluster_idx);
    cluster = cluster_tmp{1, 1};
    
    total_error = 0;
    for idx = 1 : num_micro_clusters
        if idx ~= micro_cluster_idx
            current_total_error = 0;
            micro_cluster_tmp = mirco_clusters(idx);
            micro_cluster = micro_cluster_tmp{1, 1};
             for point_idx = 1 : length(cluster)                 
                 w = W(:, cluster(point_idx));
                 current_total_error = current_total_error + norm(data(:, cluster(point_idx)) - Dp(:, micro_cluster) * w(micro_cluster, 1));  
             end
             if total_error < 1e-6
                total_error = current_total_error;
                micro_clusters_merged_idx = idx;  
             elseif current_total_error < total_error
                 total_error = current_total_error;
                 micro_clusters_merged_idx = idx;                     
             end
        end
    end
     
    if micro_clusters_merged_idx > 0
         major_mirco_cluster_tmp = mirco_clusters(micro_cluster_idx);
        major_mirco_cluster = major_mirco_cluster_tmp{1, 1};
         mirco_cluster_merged_tmp = mirco_clusters(micro_clusters_merged_idx);
        mirco_cluster_merged = mirco_cluster_merged_tmp{1, 1};
%         original_similarity = similarity_between_clusters(micro_cluster_idx, micro_clusters_merged_idx);
        original_similarity = sum(sum(L(major_mirco_cluster, mirco_cluster_merged)));
%         new_clusetrs_merged_idx = union(mirco_clusters(micro_cluster_idx), mirco_clusters(micro_clusters_merged_idx));      
       
        new_clusetrs_merged_idx = union(major_mirco_cluster, mirco_cluster_merged);
        new_max_similarity = 0;
        for current_idx = 1 : num_micro_clusters
            if current_idx ~= micro_cluster_idx && current_idx ~= micro_clusters_merged_idx
                current_mirco_cluster_tmp = mirco_clusters(current_idx);
                current_mirco_cluster = current_mirco_cluster_tmp{1, 1};
                current_similarity = sum(sum(L(current_mirco_cluster, new_clusetrs_merged_idx)));
                if current_similarity > new_max_similarity
                    new_max_similarity = current_similarity;
                end
            end
        end
        % check the merged condition
        if new_max_similarity > original_similarity
            micro_clusters_merged_idx = 0; % doesn't statisfiy the merged condition
        end
    end
end
