close all;
clear;
clc;

addpath('data/');
addpath('utility/');
addpath('ssc_files/');


%betas = [1, 2];
%finetuning_params = [0, 1]; %enable_finetuing: 0 disable, 1 enable
betas = [1];
finetuning_params = [0];
dim = 0;

% 1 Network Intrusion; 2 Keystroke; 3 Forest Cover; 4 COIL-100; 5 USPS
for data_index = 2 : 2
switch data_index
    case 1
%         lambdas = [0.01, 0.05, 0.1, 0.5, 1, 5];
        lambdas = [0.1];
        load('network_data.mat');
        filename = "network";
        hsrc_data = network_data;
        hsrc_labels = network_labels;
        total_num = size(hsrc_data, 2);
        n = 1000;
        % num_windows = floor(total_num / n);
        num_windows = 50;

    case 2
        %lambdas =[10, 20, 50, 100, 200, 500];
        lambdas =[100];
        load('key_data.mat');
        filename = "key";
        hsrc_data = key_data;
        hsrc_labels = key_labels;
        total_num = size(hsrc_data, 2);
        n = 200;
        num_windows = floor(total_num / n);   

    case 3
%         lambdas = [10, 50, 100, 150, 200, 300, 500, 1000];
        lambdas = [50];
        load('forest_cover_data.mat');
        filename = "forest_cover";
        hsrc_data = forest_cover_data;
        hsrc_labels = forest_cover_labels;
        total_num = size(hsrc_data, 2);
        n = 1000;
        % num_windows = floor(total_num / n);
        num_windows = 50; 

     case 4
        lambdas = [10, 20, 50, 100, 200, 500, 1000];
        dim = 50;
        load('COIL100.mat');
        filename = "coil100";  
        hsrc_data = im2double(fea');
        hsrc_labels = gnd';
        total_num = size(hsrc_data, 2);
        rand('state', 100);
        y = randperm(total_num);
        hsrc_data = hsrc_data(:, y);
        hsrc_labels = hsrc_labels(y);        
        n = 1000;
        num_windows = floor(total_num / n);

    case 5   
        lambdas = [10, 20, 50, 100, 200, 500];
        dim = 50;
        load('usps.mat');
        filename = "usps";        
        hsrc_data = mat2gray(data(:, 2 : end))';   
        total_num = size(hsrc_data, 2);
        hsrc_labels = data(1 : total_num, 1)';
        n = 1000;
        num_windows = floor(total_num / n);

end
K = max(hsrc_labels);

sparse_mode = 0;
lmd_num = length(lambdas);
beta_num = length(betas);
ft_num = length(finetuning_params);

final_clustering_purs = zeros(lmd_num, beta_num, ft_num, num_windows);
final_clustering_fms =  zeros(lmd_num, beta_num, ft_num, num_windows);
final_clustering_nmis =  zeros(lmd_num, beta_num, ft_num, num_windows);
final_clustering_costs = zeros(lmd_num, beta_num, ft_num, num_windows);

final_result = strcat('results/', filename, '_final_result.txt');
final_average_result = strcat('results/', filename, '_final_average_result.txt');
final_result_mat = strcat(filename, '_final_result.mat');
final_average_result_mat = strcat(filename, '_final_average_result.mat');

for lmd_idx = 1 : lmd_num
    lambda = lambdas(lmd_idx);
    for beta_idx = 1 : beta_num
        n_times = betas(beta_idx);    
        for ft_idx = 1 : ft_num
            enable_ft = finetuning_params(ft_idx);             
            for wnd_idx = 1 : num_windows                 
                start_idx = (wnd_idx - 1) * n + 1;
                X = hsrc_data(:, start_idx : start_idx + n - 1);
%                 for i = 1 : n
%                     X(:, i) = X(:, i) ./ max(1e-12,norm(X(:, i)));
%                 end
                ground_lables = hsrc_labels(start_idx : start_idx + n - 1);                    
%                 if wnd_idx == 1
%                     X_full = X;                        
%                 else
%                     X_full = [X, Xs];                        
%                 end
                X_full = X;   
                if dim > 1e-6
                    [eigen_vector, ~, ~] = f_pca(X_full, dim);
                    XX = normc(eigen_vector' * X_full);
                else
                    XX = normc(X_full);
                end
                tic;
                [Z, W] = sp_representation(XX, lambda, sparse_mode);
                [mirco_clusters_merged, num_clusters_merged] = merge_mirco_clusters(XX, W, Z, K, n_times, enable_ft);

                total_cluster_num = 0;
                final_clustering_results = zeros(num_clusters_merged, 1);
                final_score_results = zeros(num_clusters_merged, 1);
                for idx = 1 : num_clusters_merged
                    final_mirco_cluster_tmp = mirco_clusters_merged(idx);
                    final_mirco_cluster = final_mirco_cluster_tmp{1};
                    if ~isempty(final_mirco_cluster)
                        table = tabulate(ground_lables(final_mirco_cluster));
                        [final_clustering_results(idx), row_idx] = max(table(:,3));
                        total_cluster_num = total_cluster_num + table(row_idx, 2);
        
                        current_cluster_num =  table(row_idx, 2);                
                        orginal_wnd_num = length(find(ground_lables == table(row_idx, 1)));
                        recall = current_cluster_num / orginal_wnd_num;
                        precision = final_clustering_results(idx) / 100;
                        final_score_results(idx) = 2 * (precision * recall) / (precision + recall); 
                        
                    end
                end
                purtiy = mean(final_clustering_results(final_clustering_results > 1e-6));
                fmeasure = mean(final_score_results(final_score_results > 1e-6));

                time_cost1 = toc;
%                 if wnd_idx == 1
%                     % the representative objects used in the next window
%                     Xs = X;
%                 else
%                     % the representative objects used in the next window
%                     Xs = construct_representative_objects(X_full, X_full, Z, n, actual_ids, k);                                             
%                 end
%                             
%                 [current_ground_lables, num_current_clusters] = refresh_labels(ground_lables, K);
%                 [actual_ids, ~] = refresh_labels(actual_ids(1 : n), K);

                time_cost = toc;

                final_clustering_purs(lmd_idx, beta_idx, ft_idx, wnd_idx) = purtiy;
                final_clustering_fms(lmd_idx, beta_idx, ft_idx, wnd_idx) = fmeasure;
                final_clustering_costs(lmd_idx, beta_idx, ft_idx, wnd_idx) = time_cost;
                disp([wnd_idx, purtiy, fmeasure, time_cost]);
                writematrix([wnd_idx, roundn(purtiy, -4), roundn(fmeasure, -4), roundn(time_cost, -2)], final_result, "Delimiter", 'tab', 'WriteMode', 'append'); 
            end
            average_purtiy =  mean(final_clustering_purs(lmd_idx, beta_idx, ft_idx, :));
            average_fm =  mean(final_clustering_fms(lmd_idx, beta_idx, ft_idx, :));                    
            average_cost = mean(final_clustering_costs(lmd_idx, beta_idx, ft_idx, :));
            disp([lambda, n_times, average_purtiy, average_fm, average_cost]);
            writematrix([lambda, n_times, enable_ft, roundn(average_purtiy, -4), roundn(average_fm, -4), roundn(average_cost, -2)], final_average_result, "Delimiter", 'tab', 'WriteMode', 'append'); 
        end
    end
end
save(final_average_result_mat, 'final_clustering_purs', 'final_clustering_fms', 'final_clustering_costs');

end
