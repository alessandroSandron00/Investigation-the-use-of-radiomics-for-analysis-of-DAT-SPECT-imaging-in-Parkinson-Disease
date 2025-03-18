clearvars; close all; clc;

rng(123)
base_path='.';

%% Set up the Import Options and import the data 
opts = spreadsheetImportOptions("NumVariables", 178);
% Specify sheet and range
opts.Sheet = "Radiomics";
opts.DataRange = "A2:FV54";
% Specify column names and types
opts.VariableNames = ["id_subject", "loc_peak_loc", "loc_peak_glob", "stat_mean", "stat_var", "stat_skew", "stat_kurt", "stat_median", "stat_min", "stat_p10", "stat_p90", "stat_max", "stat_iqr", "stat_range", "stat_mad", "stat_rmad", "stat_medad", "stat_cov", "stat_qcod", "stat_energy", "stat_rms", "ivh_v10", "ivh_v25", "ivh_v50", "ivh_v75", "ivh_v90", "ivh_i10", "ivh_i25", "ivh_i50", "ivh_i75", "ivh_i90", "ivh_diff_v10_v90", "ivh_diff_v25_v75", "ivh_diff_i10_i90", "ivh_diff_i25_i75", "morph_volume", "morph_vol_approx", "morph_area_mesh", "morph_av", "morph_comp_1", "morph_comp_2", "morph_sph_dispr", "morph_sphericity", "morph_asphericity", "morph_com", "morph_diam", "morph_pca_maj_axis", "morph_pca_min_axis", "morph_pca_least_axis", "morph_pca_elongation", "morph_pca_flatness", "morph_vol_dens_aabb", "morph_area_dens_aabb", "morph_vol_dens_aee", "morph_area_dens_aee", "morph_vol_dens_conv_hull", "morph_area_dens_conv_hull", "morph_integ_int", "morph_moran_i", "morph_geary_c", "ih_mean_fbs_w00125", "ih_var_fbs_w00125", "ih_skew_fbs_w00125", "ih_kurt_fbs_w00125", "ih_median_fbs_w00125", "ih_min_fbs_w00125", "ih_p10_fbs_w00125", "ih_p90_fbs_w00125", "ih_max_fbs_w00125", "ih_mode_fbs_w00125", "ih_iqr_fbs_w00125", "ih_range_fbs_w00125", "ih_mad_fbs_w00125", "ih_rmad_fbs_w00125", "ih_medad_fbs_w00125", "ih_cov_fbs_w00125", "ih_qcod_fbs_w00125", "ih_entropy_fbs_w00125", "ih_uniformity_fbs_w00125", "ih_max_grad_fbs_w00125", "ih_max_grad_g_fbs_w00125", "ih_min_grad_fbs_w00125", "ih_min_grad_g_fbs_w00125", "cm_joint_max_d1_3d_avg_fbs_w00125", "cm_joint_avg_d1_3d_avg_fbs_w00125", "cm_joint_var_d1_3d_avg_fbs_w00125", "cm_joint_entr_d1_3d_avg_fbs_w00125", "cm_diff_avg_d1_3d_avg_fbs_w00125", "cm_diff_var_d1_3d_avg_fbs_w00125", "cm_diff_entr_d1_3d_avg_fbs_w00125", "cm_sum_avg_d1_3d_avg_fbs_w00125", "cm_sum_var_d1_3d_avg_fbs_w00125", "cm_sum_entr_d1_3d_avg_fbs_w00125", "cm_energy_d1_3d_avg_fbs_w00125", "cm_contrast_d1_3d_avg_fbs_w00125", "cm_dissimilarity_d1_3d_avg_fbs_w00125", "cm_inv_diff_d1_3d_avg_fbs_w00125", "cm_inv_diff_norm_d1_3d_avg_fbs_w00125", "cm_inv_diff_mom_d1_3d_avg_fbs_w00125", "cm_inv_diff_mom_norm_d1_3d_avg_fbs_w00125", "cm_inv_var_d1_3d_avg_fbs_w00125", "cm_corr_d1_3d_avg_fbs_w00125", "cm_auto_corr_d1_3d_avg_fbs_w00125", "cm_clust_tend_d1_3d_avg_fbs_w00125", "cm_clust_shade_d1_3d_avg_fbs_w00125", "cm_clust_prom_d1_3d_avg_fbs_w00125", "cm_info_corr1_d1_3d_avg_fbs_w00125", "cm_info_corr2_d1_3d_avg_fbs_w00125", "rlm_sre_3d_avg_fbs_w00125", "rlm_lre_3d_avg_fbs_w00125", "rlm_lgre_3d_avg_fbs_w00125", "rlm_hgre_3d_avg_fbs_w00125", "rlm_srlge_3d_avg_fbs_w00125", "rlm_srhge_3d_avg_fbs_w00125", "rlm_lrlge_3d_avg_fbs_w00125", "rlm_lrhge_3d_avg_fbs_w00125", "rlm_glnu_3d_avg_fbs_w00125", "rlm_glnu_norm_3d_avg_fbs_w00125", "rlm_rlnu_3d_avg_fbs_w00125", "rlm_rlnu_norm_3d_avg_fbs_w00125", "rlm_r_perc_3d_avg_fbs_w00125", "rlm_gl_var_3d_avg_fbs_w00125", "rlm_rl_var_3d_avg_fbs_w00125", "rlm_rl_entr_3d_avg_fbs_w00125", "szm_sze_3d_fbs_w00125", "szm_lze_3d_fbs_w00125", "szm_lgze_3d_fbs_w00125", "szm_hgze_3d_fbs_w00125", "szm_szlge_3d_fbs_w00125", "szm_szhge_3d_fbs_w00125", "szm_lzlge_3d_fbs_w00125", "szm_lzhge_3d_fbs_w00125", "szm_glnu_3d_fbs_w00125", "szm_glnu_norm_3d_fbs_w00125", "szm_zsnu_3d_fbs_w00125", "szm_zsnu_norm_3d_fbs_w00125", "szm_z_perc_3d_fbs_w00125", "szm_gl_var_3d_fbs_w00125", "szm_zs_var_3d_fbs_w00125", "szm_zs_entr_3d_fbs_w00125", "dzm_sde_3d_fbs_w00125", "dzm_lde_3d_fbs_w00125", "dzm_lgze_3d_fbs_w00125", "dzm_hgze_3d_fbs_w00125", "dzm_sdlge_3d_fbs_w00125", "dzm_sdhge_3d_fbs_w00125", "dzm_ldlge_3d_fbs_w00125", "dzm_ldhge_3d_fbs_w00125", "dzm_glnu_3d_fbs_w00125", "dzm_glnu_norm_3d_fbs_w00125", "dzm_zdnu_3d_fbs_w00125", "dzm_zdnu_norm_3d_fbs_w00125", "dzm_z_perc_3d_fbs_w00125", "dzm_gl_var_3d_fbs_w00125", "dzm_zd_var_3d_fbs_w00125", "dzm_zd_entr_3d_fbs_w00125", "ngt_coarseness_3d_fbs_w00125", "ngt_contrast_3d_fbs_w00125", "ngt_busyness_3d_fbs_w00125", "ngt_complexity_3d_fbs_w00125", "ngt_strength_3d_fbs_w00125", "ngl_lde_d1_a00_3d_fbs_w00125", "ngl_hde_d1_a00_3d_fbs_w00125", "ngl_lgce_d1_a00_3d_fbs_w00125", "ngl_hgce_d1_a00_3d_fbs_w00125", "ngl_ldlge_d1_a00_3d_fbs_w00125", "ngl_ldhge_d1_a00_3d_fbs_w00125", "ngl_hdlge_d1_a00_3d_fbs_w00125", "ngl_hdhge_d1_a00_3d_fbs_w00125", "ngl_glnu_d1_a00_3d_fbs_w00125", "ngl_glnu_norm_d1_a00_3d_fbs_w00125", "ngl_dcnu_d1_a00_3d_fbs_w00125", "ngl_dcnu_norm_d1_a00_3d_fbs_w00125", "ngl_dc_perc_d1_a00_3d_fbs_w00125", "ngl_gl_var_d1_a00_3d_fbs_w00125", "ngl_dc_var_d1_a00_3d_fbs_w00125", "ngl_dc_entr_d1_a00_3d_fbs_w00125", "ngl_dc_energy_d1_a00_3d_fbs_w00125"];
opts.VariableTypes = ["string", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];
% Specify variable properties
opts = setvaropts(opts, "id_subject", "WhitespaceRule", "preserve");
opts = setvaropts(opts, "id_subject", "EmptyFieldRule", "auto");
% Import the data
Radiomics = readtable("Homework_Dataset.xlsx", opts, "UseExcel", false);
clear opts;

opts = spreadsheetImportOptions("NumVariables", 8);
% Specify sheet and range
opts.Sheet = "Demographics Clinical";
opts.DataRange = "A2:H54";
% Specify column names and types
opts.VariableNames = ["id_subject", "Group1PD0Controls", "Age", "GenderM0F1", "EducationYears", "Height", "Weight", "BMI"];
opts.VariableTypes = ["string", "double", "double", "double", "double", "double", "double", "double"];
% Specify variable properties
opts = setvaropts(opts, "id_subject", "WhitespaceRule", "preserve");
opts = setvaropts(opts, "id_subject", "EmptyFieldRule", "auto");
% Import the data
DemographicsClinical = readtable("Homework_Dataset.xlsx", opts, "UseExcel", false);
clear opts;

opts = spreadsheetImportOptions("NumVariables", 10);
% Specify sheet and range
opts.Sheet = "CLINICAL";
opts.DataRange = "A2:J34";
% Specify column names and types
opts.VariableNames = ["id_subject", "LEDDTotal", "UPDRSI", "UPDRSII", "UPDRSIII", "UPDRSIV", "UPDRSTotal", "NMSQ", "MMSE", "MoCA"];
opts.VariableTypes = ["string", "double", "double", "double", "double", "double", "double", "double", "double", "double"];
% Specify variable properties
opts = setvaropts(opts, "id_subject", "WhitespaceRule", "preserve");
opts = setvaropts(opts, "id_subject", "EmptyFieldRule", "auto");
% Import the data
CLINICAL = readtable("Homework_Dataset.xlsx", opts, "UseExcel", false);
clear opts;

% Variable number 174: 
% check in Radiomics: it is a variable with all "1" values
% Dependence count percentage =
% Fraction of the number of realised neighbourhoods and
% the maximum number of potential neighbourhoods. This
% value is usually equal to 1 (documentation)
% is makes sense to remove it form the radiomics features, since it does
% not add any kind of informations to the analysis
Radiomics = removevars(Radiomics, 'ngl_dc_perc_d1_a00_3d_fbs_w00125');

% remove also the column "id_subject", since it is not a relevant variable
Radiomics = removevars(Radiomics, 'id_subject');
nfeatures = size(Radiomics,2);
CLINICAL  = removevars(CLINICAL, 'id_subject');
nclinical = size(CLINICAL,2);

patients_index = DemographicsClinical.Group1PD0Controls==1;
controls_index = DemographicsClinical.Group1PD0Controls==0;
npatients = sum(patients_index);
ncontrols = sum(controls_index);
nsubjects = npatients+ncontrols;

RadiomicsPatients = Radiomics(patients_index,:);
RadiomicsControls = Radiomics(controls_index,:);
DemographicsPatients = DemographicsClinical(patients_index,:);
DemographicsControls = DemographicsClinical(controls_index,:);

%% PART 1A: Is there any difference between controls and PD in radiomics features (or combined score)?

%% Check on the distributions

% Check on the distribution of Demographics 
for ii = 3:size(DemographicsClinical,2)  
    % display(['Demographics #',num2str(ii)])
    % figure; qqplot(table2array(DemographicsClinical(:,ii)))
    % k = kurtosis(table2array(DemographicsClinical(:,ii))); 
    % s = skewness(table2array(DemographicsClinical(:,ii))); 
    % display(['kurtosis=',num2str(k)])
    % display(['skewness=',num2str(s)])  
    hD(ii-2) = lillietest(table2array(DemographicsClinical(:,ii)));
    % pause
end

% Check on the distribution of Radiomics 
for ii = 1:nfeatures 
    % display(['Feature #',num2str(ii)])
    % figure; qqplot(table2array(Radiomics(:,ii)))
    % k = kurtosis(table2array(Radiomics(:,ii)));
    % s = skewness(table2array(Radiomics(:,ii))); 
    % display(['kurtosis=',num2str(k)])
    % display(['skewness=',num2str(s)]) 
    hR(ii) = lillietest(table2array(Radiomics(:,ii)));
    % pause
end

%% cleaning
clear ii  k  s  hD; close all;

%% Statistical analysis to check if groups are matched (looking at demographic features)
% labels for boxplots
g1 = repmat({'Patients'},npatients,1);
g2 = repmat({'Controls'},ncontrols,1);

% Age
figure;
boxplot([DemographicsPatients.Age(:); DemographicsControls.Age(:)], [g1;g2])
ylabel('Age');title('Distribution of age in Patients and in Controls')

[h,p] = ttest2(DemographicsPatients.Age(:), DemographicsControls.Age(:));
if h==0
    disp(['Null hypothesis accepted: groups are matched by Age (p-value=',num2str(p),')']);
end

% Gender
figure;
histogram(categorical(DemographicsPatients.GenderM0F1(:),[0 1],{'Male','Female'}))
ylabel('#subj'); 
hold on
histogram(categorical(DemographicsControls.GenderM0F1(:),[0 1],{'Male','Female'}))
legend('Patients','Controls')
x = [sum(DemographicsPatients.GenderM0F1(:)), length(DemographicsPatients.GenderM0F1(:))-sum(DemographicsPatients.GenderM0F1(:));...
    sum(DemographicsControls.GenderM0F1(:)), length(DemographicsControls.GenderM0F1(:))-sum(DemographicsControls.GenderM0F1(:))];
[h,p,~] = fishertest(x,'Tail','Right');
if h==0
    disp(['Null hypothesis accepted: groups are matched by Gender (p-value=',num2str(p),')']);
end

% BMI
figure;
boxplot([DemographicsPatients.BMI(:); DemographicsControls.BMI(:)], [g1;g2])
ylabel('BMI');title('Distribution of BMI in Patients and in Controls')

[p,h] = ranksum(DemographicsPatients.BMI(:), DemographicsControls.BMI(:));
if h==0
    disp(['Null hypothesis accepted: groups are matched by BMI (p-value=',num2str(p),')']);
end

% Weight
figure;
boxplot([DemographicsPatients.Weight(:); DemographicsControls.Weight(:)], [g1;g2])
ylabel('Weight');title('Distribution of Weight in Patients and in Controls')

[h,p] = ttest2(DemographicsPatients.Weight(:), DemographicsControls.Weight(:));
if h==0
    disp(['Null hypothesis accepted: groups are matched by Weight (p-value=',num2str(p),')']);
end

% Height
figure;
boxplot([DemographicsPatients.Height(:); DemographicsControls.Height(:)], [g1;g2])
ylabel('Height');title('Distribution of Height in Patients and in Controls')

[h,p] = ttest2(DemographicsPatients.Height(:), DemographicsControls.Height(:));
if h==0
    disp(['Null hypothesis accepted: groups are matched by Height (p-value=',num2str(p),')']);
end

% EducationYears
figure;
boxplot([DemographicsPatients.EducationYears(:); DemographicsControls.EducationYears(:)], [g1;g2])
ylabel('EducationYears');title('Distribution of EducationYears in Patients and in Controls')

[p,h] = ranksum(DemographicsPatients.EducationYears(:), DemographicsControls.EducationYears(:));
if h==0
    disp(['Null hypothesis accepted: groups are matched by EducationYears (p-value=',num2str(p),')']);
end

% Patients and Controls are matched in all their demographic features

%% cleaning
clear h  p  g1  g2  x; close all;

%% Cross Sectional Analysis (CSA) without covariates 
CSA_wo_covariates = zeros(nfeatures,2);
for ii = 1:nfeatures
    if hR(ii)==1
        [p,h] = ranksum(table2array(RadiomicsControls(:,ii)),table2array(RadiomicsPatients(:,ii)));
        CSA_wo_covariates(ii,:) = [h p];
    else
        [h,p] = ttest2(table2array(RadiomicsControls(:,ii)),table2array(RadiomicsPatients(:,ii)));
        CSA_wo_covariates(ii,:) = [h p];
    end
end
nfeat_CSA_wo_covariates = sum(length(find(CSA_wo_covariates(:,1)==1))); % 158 Radiomics features
CSA_wo_covariates = array2table(CSA_wo_covariates,'RowNames',RadiomicsControls.Properties.VariableNames,'VariableNames',{'H0','P_value'});
% the majority of radiomics features are significantly different in the
% groups: this means that it is a good choice to try to calssify patients
% vs healthy people using radiomics

clear ii  h  p  hR; 

%% False Discovery Rate: Benjamini-Yekutieli method
disp('-------------------------------------------------------------------------------------')
% Calculation of the significance threshold
alpha = 0.05;
[h, ~, ~, ~] = fdr_bh(table2array(CSA_wo_covariates(:,2)), alpha, 'dep', 'yes');
disp('--------------------------------------------------------------------------------------')

% updated data 
woCovData = Radiomics(:,h==1); % 147 Radiomics features

clear h;

%% Cross Sectional Analysis (CSA) with covariates
% for saying that radiomics features are significantly different between
% patients and controls we must verify that the previous result does not
% depend on covariates

for jj = 1:size(woCovData,2)
    [tbl_all_var(jj).Pval] = anovan(woCovData.(jj), ...
    {DemographicsClinical.GenderM0F1, ...
    DemographicsClinical.Group1PD0Controls, ...
    DemographicsClinical.Age, DemographicsClinical.BMI, ...
    DemographicsClinical.Weight, DemographicsClinical.Height,...
    DemographicsClinical.EducationYears}, 'varnames', ...
    {'G' 'T' 'A' 'BMI' 'W' 'H' 'EDY'}, ...
    'continuous', [3,4,5,6,7], 'model',[1 0 0 0 0 0 0; 0 1 0 0 0 0 0;...
    0 0 1 0 0 0 0; 0 0 0 1 0 0 0; 0 0 0 0 1 0 0; 0 0 0 0 0 1 0; ...
    0 0 0 0 0 0 1; 1 0 1 0 0 0 0; 1 0 0 1 0 0 0; 1 0 0 0 1 0 0; ...
    1 0 0 0 0 1 0; 1 0 0 0 0 0 1; 0 0 1 1 0 0 0; 0 0 1 0 1 0 0; ...
    0 0 1 0 0 1 0; 0 0 1 0 0 0 1; 0 0 0 1 1 0 0; 0 0 0 1 0 1 0; ...
    0 0 0 1 0 0 1; 0 0 0 0 1 1 0; 0 0 0 0 1 0 1; 0 0 0 0 0 1 1],...
    'alpha',0.05,'display','off');
end

CSA_covariates = zeros(size(woCovData,2),2);
for ii = 2:size(woCovData,2)
    p = tbl_all_var(ii).Pval(2); % pvalue for T
    if p>0.05 % significance level
        h=0;
    else 
        h=1;
    end
    CSA_covariates(ii,:) = [h p];
end
nfeat_CSA_covariate = sum(CSA_covariates(:,1)); % 142 Radiomics features
CSA_covariates = array2table(CSA_covariates,'RowNames',woCovData.Properties.VariableNames,'VariableNames',{'H0','P_value'});

clear jj  ii  p  h;

%% False Discovery Rate: Benjamini-Yekutieli method
p = [];
for ii = 1:size(woCovData,2)
    p(ii)=tbl_all_var(ii).Pval(2); % pvalue for T
end

disp('-------------------------------------------------------------------------------------')
% Calculation of the significance threshold
alpha = 0.05;
[h, ~, ~, ~] = fdr_bh(p, alpha, 'dep', 'yes');
disp('--------------------------------------------------------------------------------------')

% new updated data
CSAData = woCovData(:,h==1);

clear ii  p  tbl_all_var;

%% Features that are significantly different between groups (HC vs PD)
features = table;
for ii = 1:size(woCovData,2) 
    if h(ii)==1 
        features = [features; table(ii)];
    end
end
features.Properties.VariableNames = {'index'};
features.Properties.RowNames      = CSAData.Properties.VariableNames;

% vector: 1 corresponds to patients, 0 corresponds to controls
Y = DemographicsClinical.Group1PD0Controls;

clear h; 

%% Test on multicollinearity of Radiomics using VIF
% division of Radiomics based on the class
% if VIF>10, there is collinearity

% vector with the initial position of each class
% the last index (123) is added to make the function work always
positions = [1,3,18,31,38,55,73,86,99,108,110,123]; 

% CLASS: local intensity
length_class = positions(1:2);
VIF_loc      = VIF_selection(CSAData,features,length_class);
% CLASS: stat
length_class = positions(2:3);
VIF_stat     = VIF_selection(CSAData,features,length_class);
% CLASS: ivh
length_class = positions(3:4);
VIF_ivh      = VIF_selection(CSAData,features,length_class);
% CLASS: morphology
length_class = positions(4:5);
VIF_morph    = VIF_selection(CSAData,features,length_class);
% CLASS: intensity histogram
length_class = positions(5:6);
VIF_ih       = VIF_selection(CSAData,features,length_class);
% CLASS: gray level co-occurence matrix
length_class = positions(6:7);
VIF_cm       = VIF_selection(CSAData,features,length_class);
% CLASS: gray level run length matrix
length_class = positions(7:8);
VIF_rlm      = VIF_selection(CSAData,features,length_class);
% CLASS: gray level size zone matrix
length_class = positions(8:9);
VIF_szm      = VIF_selection(CSAData,features,length_class);
% CLASS: gray level distant zone matrix
length_class = positions(9:10);
VIF_dzm      = VIF_selection(CSAData,features,length_class);
% CLASS: Neighbourhood gray tone different matrix
length_class = positions(10:11);
VIF_ngt      = VIF_selection(CSAData,features,length_class);
% CLASS: Neighbouring gray level different matrix 
length_class = positions(11:12);
VIF_ngl      = VIF_selection(CSAData,features,length_class);

% final indexes vector
VIF_index = [VIF_loc;VIF_stat;VIF_ivh;VIF_morph;VIF_ih;VIF_cm;VIF_rlm;...
             VIF_szm;VIF_dzm;VIF_ngt;VIF_ngl];
% selection of features
VIFData_tab       = woCovData(:,VIF_index.index);
VIFData           = zscore(table2array(VIFData_tab)); % double and zscore
VIFfeatures_names = VIFData_tab.Properties.VariableNames;

% on all classes
length_class = [1, size(VIFData,2)+1];
VIF_index    = VIF_selection(VIFData_tab,VIF_index,length_class);
% selection of features (refreshed)
VIFData_tab       = woCovData(:,VIF_index.index);
VIFData           = zscore(table2array(VIFData_tab)); % double and zscore
VIFfeatures_names = VIFData_tab.Properties.VariableNames;

clear VIF_loc  VIF_stat  VIF_ivh  VIF_morph  VIF_ih  VIF_cm  VIF_rlm;
clear VIF_szm  VIF_dzm   VIF_ngt  VIF_ngl  positions;

%% PART 1B: Are the radiomics features capable to distinguish patients from controls?

%% Association Rdata1 and Y (vector with patients and HC)

% logistic with zscored Rdata1 and then correlation Y-Y_hat
r_corr      = zeros(size(VIFData_tab,2),1);
r_corr_pval = zeros(size(VIFData_tab,2),1);
for ii = 1:size(VIFData_tab,2)
    beta  = glmfit(VIFData(:,ii),Y,'binomial','constant','off','link','logit');
    Y_hat = glmval(beta,VIFData(:,ii),'logit','constant','off');
    [r_corr(ii),r_corr_pval(ii)] = corr(Y,Y_hat);
end
r_corr(r_corr_pval>0.05) = 0;
r_corr_tab = array2table(r_corr,'RowNames',VIFData_tab.Properties.VariableNames,'VariableNames',{'r'});
r_corr_tab = [r_corr_tab, array2table(r_corr_pval,'RowNames',VIFData_tab.Properties.VariableNames,'VariableNames',{'pval'})];

clear ii  r_corr  beta;

%% False Discovery Rate: Benjamini-Yekutieli method
disp('-------------------------------------------------------------------------------------')
% Calculation of the significance threshold
alpha = 0.05;
[h, ~, ~, ~] = fdr_bh(r_corr_pval, alpha, 'dep', 'yes');
disp('-------------------------------------------------------------------------------------')

clear h  alpha;

%% LASSO - different seeds
rng('default')
seeds=randi(1000,[1 10]); % 10 different seeds

lasso_results = struct;  % struct of results initialization

% seed 1
disp(' '); disp('operating with seed 1');
[LASSO_features,B,FitInfo,radiomics_score] = lasso_selection(seeds(1),VIFData,Y,VIFfeatures_names);
lasso_results.seed1.features       = LASSO_features;
lasso_results.seed1.B              = B;
lasso_results.seed1.FitInfo        = FitInfo;
lasso_results.seed1.RadiomicsScore = radiomics_score;
% seed 2
disp(' '); disp('operating with seed 2');
[LASSO_features,B,FitInfo,radiomics_score] = lasso_selection(seeds(2),VIFData,Y,VIFfeatures_names);
lasso_results.seed2.features       = LASSO_features;
lasso_results.seed2.B              = B;
lasso_results.seed2.FitInfo        = FitInfo;
lasso_results.seed2.RadiomicsScore = radiomics_score;
% seed 3
disp(' '); disp('operating with seed 3');
[LASSO_features,B,FitInfo,radiomics_score] = lasso_selection(seeds(3),VIFData,Y,VIFfeatures_names);
lasso_results.seed3.features       = LASSO_features;
lasso_results.seed3.B              = B;
lasso_results.seed3.FitInfo        = FitInfo;
lasso_results.seed3.RadiomicsScore = radiomics_score;
% seed 4
disp(' '); disp('operating with seed 4');
[LASSO_features,B,FitInfo,radiomics_score] = lasso_selection(seeds(4),VIFData,Y,VIFfeatures_names);
lasso_results.seed4.features       = LASSO_features;
lasso_results.seed4.B              = B;
lasso_results.seed4.FitInfo        = FitInfo;
lasso_results.seed4.RadiomicsScore = radiomics_score;
% seed 5
disp(' '); disp('operating with seed 5');
[LASSO_features,B,FitInfo,radiomics_score] = lasso_selection(seeds(5),VIFData,Y,VIFfeatures_names);
lasso_results.seed5.features       = LASSO_features;
lasso_results.seed5.B              = B;
lasso_results.seed5.FitInfo        = FitInfo;
lasso_results.seed5.RadiomicsScore = radiomics_score;
% seed 6
disp(' '); disp('operating with seed 6');
[LASSO_features,B,FitInfo,radiomics_score] = lasso_selection(seeds(6),VIFData,Y,VIFfeatures_names);
lasso_results.seed6.features       = LASSO_features;
lasso_results.seed6.B              = B;
lasso_results.seed6.FitInfo        = FitInfo;
lasso_results.seed6.RadiomicsScore = radiomics_score;
% seed 7
disp(' '); disp('operating with seed 7');
[LASSO_features,B,FitInfo,radiomics_score] = lasso_selection(seeds(7),VIFData,Y,VIFfeatures_names);
lasso_results.seed7.features       = LASSO_features;
lasso_results.seed7.B              = B;
lasso_results.seed7.FitInfo        = FitInfo;
lasso_results.seed7.RadiomicsScore = radiomics_score;
% seed 8
disp(' '); disp('operating with seed 8');
[LASSO_features,B,FitInfo,radiomics_score] = lasso_selection(seeds(8),VIFData,Y,VIFfeatures_names);
lasso_results.seed8.features       = LASSO_features;
lasso_results.seed8.B              = B;
lasso_results.seed8.FitInfo        = FitInfo;
lasso_results.seed8.RadiomicsScore = radiomics_score;
% seed 9
disp(' '); disp('operating with seed 9');
[LASSO_features,B,FitInfo,radiomics_score] = lasso_selection(seeds(9),VIFData,Y,VIFfeatures_names);
lasso_results.seed9.features       = LASSO_features;
lasso_results.seed9.B              = B;
lasso_results.seed9.FitInfo        = FitInfo;
lasso_results.seed9.RadiomicsScore = radiomics_score;
% seed 10
disp(' '); disp('operating with seed 10');
[LASSO_features,B,FitInfo,radiomics_score] = lasso_selection(seeds(10),VIFData,Y,VIFfeatures_names);
lasso_results.seed10.features       = LASSO_features;
lasso_results.seed10.B              = B;
lasso_results.seed10.FitInfo        = FitInfo;
lasso_results.seed10.RadiomicsScore = radiomics_score;

clear LASSO_features  B  FitInfo  radiomics_score; 

%% cleaning
close all;

%% FINAL DATASET to use for the classification

% chosen LASSO result: seed 1 (by visual inspection of the most common 
% features selected by Lasso)
lasso_results = lasso_results.seed1;

% non zero coefficients
ii = lasso_results.B(:,lasso_results.FitInfo.IndexMinDeviance)~=0;

% final dataset
dataset = VIFData(:,ii); % zscored data
% table to be imported in R
Rdataset = VIFData_tab(:,ii);
writetable(Rdataset,'Rdataset1.csv');

clear ii; 

%% Spearman correlation between Radiomics patients Score and CLINICALS
patients_score = lasso_results.RadiomicsScore(1:npatients);

clinical_score_corr        = zeros(nclinical,1);
clinical_score_corr_pvalue = zeros(nclinical,1);
for jj = 1:nclinical
    [tmp,pval] = corr([patients_score,table2array(CLINICAL(:,jj))],'Type','spearman');
    clinical_score_corr(jj)        = tmp(1,2);
    clinical_score_corr_pvalue(jj) = pval(1,2);
end
% keeping only the significant values, the others set to zero
clinical_score_corr(clinical_score_corr_pvalue>0.05) = 0;
[r,c] = find(clinical_score_corr~=0);
clinical_score_corr_tab = array2table(clinical_score_corr,'RowNames',CLINICAL.Properties.VariableNames,'VariableNames',{'Radiomic Score'});

clear clinical_score_corr  clinical_score_corr_pvalue;

%% PART 2:  Are the radiomic features or any derived scores associated to PD clinical symptoms severity?

%% ANOVAN analysis

% ANOVA with all covariates
for jj = 1:nfeatures
    for ii = 1:nclinical
    [tbl_all_var(jj,ii).Pval] = anovan(RadiomicsPatients.(jj), ...
    {DemographicsPatients.GenderM0F1, ...
    table2array(CLINICAL(:,ii)), ...
    DemographicsPatients.Age, DemographicsPatients.BMI, ...
    DemographicsPatients.Weight, DemographicsPatients.Height,...
    DemographicsPatients.EducationYears}, 'varnames', ...
    {'G' 'symptoms severity' 'A' 'BMI' 'W' 'H' 'EDY'}, ...
    'continuous', [2,3,4,5,6,7], 'model',[1 0 0 0 0 0 0; 0 1 0 0 0 0 0;...
    0 0 1 0 0 0 0; 0 0 0 1 0 0 0; 0 0 0 0 1 0 0; 0 0 0 0 0 1 0; ...
    0 0 0 0 0 0 1; 1 0 1 0 0 0 0; 1 0 0 1 0 0 0; 1 0 0 0 1 0 0; ...
    1 0 0 0 0 1 0; 1 0 0 0 0 0 1; 0 0 1 1 0 0 0; 0 0 1 0 1 0 0; ...
    0 0 1 0 0 1 0; 0 0 1 0 0 0 1; 0 0 0 1 1 0 0; 0 0 0 1 0 1 0; ...
    0 0 0 1 0 0 1; 0 0 0 0 1 1 0; 0 0 0 0 1 0 1; 0 0 0 0 0 1 1],...
    'alpha',0.05,'display','off');
    end
end
clinical_covariates = significance_selection(nfeatures,nclinical,tbl_all_var);
clear jj  ii  tbl_all_var

% ANOVA without covariates 
for jj = 1:nfeatures
    for ii = 1:nclinical
    [tbl_all_var(jj,ii).Pval] = anovan(RadiomicsPatients.(jj), ...
    {table2array(CLINICAL(:,ii))}, 'varnames', ...
    {'symptoms severity'}, ...
    'continuous', 1,'alpha',0.05,'display','off');
    end
end
clinical_wo_covariates = significance_selection(nfeatures,nclinical,tbl_all_var);
clear jj  ii  tbl_all_var

% ANOVA with just gender
for jj = 1:nfeatures
    for ii = 1:nclinical
    [tbl_all_var(jj,ii).Pval] = anovan(RadiomicsPatients.(jj), ...
    {DemographicsPatients.GenderM0F1,table2array(CLINICAL(:,ii))}, ...
    'varnames', {'gender','symptoms severity'}, ...
    'continuous', 2,'alpha',0.05,'display','off');
    end
end
clinical_gender = significance_selection(nfeatures,nclinical,tbl_all_var);
clear jj  ii  tbl_all_var

% ANOVA with just age
for jj = 1:nfeatures
    for ii = 1:nclinical
    [tbl_all_var(jj,ii).Pval] = anovan(RadiomicsPatients.(jj), ...
    {DemographicsPatients.Age,table2array(CLINICAL(:,ii))}, 'varnames', ...
    {'age','symptoms severity'}, ...
    'continuous', [1,2],'alpha',0.05,'display','off');
    end
end
clinical_age = significance_selection(nfeatures,nclinical,tbl_all_var);
clear jj  ii  tbl_all_var

% ANOVA with just education years
for jj = 1:nfeatures
    for ii = 1:nclinical
    [tbl_all_var(jj,ii).Pval] = anovan(RadiomicsPatients.(jj), ...
    {DemographicsPatients.EducationYears,table2array(CLINICAL(:,ii))}, 'varnames', ...
    {'education years','symptoms severity'}, ...
    'continuous', [1,2],'alpha',0.05,'display','off');
    end
end
clinical_edy = significance_selection(nfeatures,nclinical,tbl_all_var);
clear jj  ii  tbl_all_var

% ANOVA with just height
for jj = 1:nfeatures
    for ii = 1:nclinical
    [tbl_all_var(jj,ii).Pval] = anovan(RadiomicsPatients.(jj), ...
    {DemographicsPatients.Height,table2array(CLINICAL(:,ii))}, 'varnames', ...
    {'height','symptoms severity'}, ...
    'continuous', [1,2],'alpha',0.05,'display','off');
    end
end
clinical_height = significance_selection(nfeatures,nclinical,tbl_all_var);
clear jj  ii  tbl_all_var

% ANOVA with just weight
for jj = 1:nfeatures
    for ii = 1:nclinical
    [tbl_all_var(jj,ii).Pval] = anovan(RadiomicsPatients.(jj), ...
    {DemographicsPatients.Weight,table2array(CLINICAL(:,ii))}, 'varnames', ...
    {'weight','symptoms severity'}, ...
    'continuous', [1,2],'alpha',0.05,'display','off');
    end
end
clinical_weight = significance_selection(nfeatures,nclinical,tbl_all_var);
clear jj  ii  tbl_all_var

% ANOVA with just BMI
for jj = 1:nfeatures
    for ii = 1:nclinical
    [tbl_all_var(jj,ii).Pval] = anovan(RadiomicsPatients.(jj), ...
    {DemographicsPatients.BMI,table2array(CLINICAL(:,ii))}, 'varnames', ...
    {'BMI','symptoms severity'}, ...
    'continuous', [1,2],'alpha',0.05,'display','off');
    end
end
clinical_BMI = significance_selection(nfeatures,nclinical,tbl_all_var);
clear jj  ii  tbl_all_var

%% 
covariates_names = {'w/o covariates','Age','Gender','EducationYears','Height','Weight','BMI'};
feature_names    = Radiomics.Properties.VariableNames;
ClinicalResults  = struct;

%% LEDDTotal
LEDDTotal = [clinical_wo_covariates(:,1), clinical_age(:,1), ...
            clinical_gender(:,1),clinical_edy(:,1),clinical_height(:,1), ...
            clinical_weight(:,1),clinical_BMI(:,1)];
LEDDTotal = array2table(LEDDTotal,"VariableNames",covariates_names,"RowNames",feature_names);

idx = significance_indexes(LEDDTotal, nfeatures);

regressors = zscore([DemographicsPatients.Age, DemographicsPatients.GenderM0F1, ...
             DemographicsPatients.Height, DemographicsPatients.Weight, ...
             DemographicsPatients.BMI, table2array(RadiomicsPatients(:,idx))]);
regressors = [ones(33,1), regressors]; % adding the intercept

[ClinicalResults(1).regressionCoef, ClinicalResults(1).prediction, ...
ClinicalResults(1).rho, ClinicalResults(1).pvalue] = LinearRegressionModel(regressors,CLINICAL(:,1));

%% UPDRSI
UPDRSI = [clinical_wo_covariates(:,2),clinical_age(:,2), ...
         clinical_gender(:,2),clinical_edy(:,2),clinical_height(:,2), ...
         clinical_weight(:,2),clinical_BMI(:,2)];
UPDRSI = array2table(UPDRSI,"VariableNames",covariates_names,"RowNames",feature_names);

idx = significance_indexes(UPDRSI, nfeatures);

regressors = zscore([DemographicsPatients.Age, DemographicsPatients.EducationYears, ...
             DemographicsPatients.Height, table2array(RadiomicsPatients(:,idx))]);
regressors = [ones(33,1), regressors];

[ClinicalResults(2).regressionCoef, ClinicalResults(2).prediction, ...
ClinicalResults(2).rho, ClinicalResults(2).pvalue] = LinearRegressionModel(regressors,CLINICAL(:,2));

%% UPDRSII
UPDRSII = [clinical_wo_covariates(:,3),clinical_age(:,3), ...
          clinical_gender(:,3),clinical_edy(:,3),clinical_height(:,3), ...
          clinical_weight(:,3),clinical_BMI(:,3)];
UPDRSII = array2table(UPDRSII,"VariableNames",covariates_names,"RowNames",feature_names);
 
idx = significance_indexes(UPDRSII, nfeatures);

regressors = zscore([DemographicsPatients.Age, DemographicsPatients.Height, ...
             table2array(RadiomicsPatients(:,idx))]);
regressors = [ones(33,1), regressors];

[ClinicalResults(3).regressionCoef, ClinicalResults(3).prediction, ...
ClinicalResults(3).rho, ClinicalResults(3).pvalue] = LinearRegressionModel(regressors,CLINICAL(:,3));

%% UPDRSIII
UPDRSIII = [clinical_wo_covariates(:,4),clinical_age(:,4), ...
           clinical_gender(:,4),clinical_edy(:,4),clinical_height(:,4), ...
           clinical_weight(:,4),clinical_BMI(:,4)];
UPDRSIII = array2table(UPDRSIII,"VariableNames",covariates_names,"RowNames",feature_names);

idx = significance_indexes(UPDRSIII, nfeatures);

regressors = zscore([DemographicsPatients.Age, DemographicsPatients.GenderM0F1, ...
             DemographicsPatients.EducationYears, ...
             DemographicsPatients.Height, DemographicsPatients.Weight, ...
             DemographicsPatients.BMI,  table2array(RadiomicsPatients(:,idx))]);
regressors = [ones(33,1), regressors];

[regressionCoef4, prediction4, rho4, pvalue4] = LinearRegressionModel(regressors,CLINICAL(:,4));

% VIF selection
positions  = [8,11,18,19,22,25,26]; 
feats      = array2table([1:length(regressors)]');
regressors = array2table(regressors);
% CLASS: stat
length_class = positions(1:2);
VIF_stat     = VIF_selection(regressors,feats,length_class);
% CLASS: morph
length_class = positions(2:3);
VIF_morph    = VIF_selection(regressors,feats,length_class);
% CLASS: ih
length_class = positions(3:4);
VIF_ih       = VIF_selection(regressors,feats,length_class);
% CLASS: cm
length_class = positions(4:5);
VIF_cm       = VIF_selection(regressors,feats,length_class);
% CLASS: szm
length_class = positions(5:6);
VIF_szm      = VIF_selection(regressors,feats,length_class);
% CLASS: dzm
length_class = positions(6:7);
VIF_dzm      = VIF_selection(regressors,feats,length_class);

% final indexes vector
VIF_index = [VIF_stat;VIF_morph;VIF_ih;VIF_cm;VIF_szm;VIF_dzm];

% selection of features
newRegressors = regressors(:,[1:7,table2array(VIF_index)']);

[ClinicalResults(4).regressionCoef, ClinicalResults(4).prediction, ...
ClinicalResults(4).rho, ClinicalResults(4).pvalue] = LinearRegressionModel(table2array(newRegressors),CLINICAL(:,4));

%% UPDRSIV
UPDRSIV = [clinical_wo_covariates(:,5),clinical_age(:,5), ...
          clinical_gender(:,5),clinical_edy(:,5),clinical_height(:,5), ...
          clinical_weight(:,5),clinical_BMI(:,5)];
UPDRSIV = array2table(UPDRSIV,"VariableNames",covariates_names,"RowNames",feature_names);

idx = significance_indexes(UPDRSIV, nfeatures);

regressors = zscore([DemographicsPatients.Age, DemographicsPatients.GenderM0F1, ...
             DemographicsPatients.EducationYears, ...
             DemographicsPatients.Height, DemographicsPatients.Weight, ...
             DemographicsPatients.BMI,  table2array(RadiomicsPatients(:,idx))]);
regressors = [ones(33,1), regressors];

[regressionCoef5, prediction5, rho5, pvalue5] = LinearRegressionModel(regressors,CLINICAL(:,5));

% VIF selection
positions  = [8,14,21,29,32,35,38,41,44]; 
feats      = array2table([1:length(regressors)]');
regressors = array2table(regressors);
% CLASS: stat
length_class = positions(1:2);
VIF_stat     = VIF_selection(regressors,feats,length_class);
% CLASS: ivh
length_class = positions(2:3);
VIF_ivh      = VIF_selection(regressors,feats,length_class);
% CLASS: ih
length_class = positions(3:4);
VIF_ih       = VIF_selection(regressors,feats,length_class);
% CLASS: cm
length_class = positions(4:5);
VIF_cm       = VIF_selection(regressors,feats,length_class);
% CLASS: rlm
length_class = positions(5:6);
VIF_rlm      = VIF_selection(regressors,feats,length_class);
% CLASS: szm
length_class = positions(6:7);
VIF_szm      = VIF_selection(regressors,feats,length_class);
% CLASS: dzm
length_class = positions(7:8);
VIF_dzm      = VIF_selection(regressors,feats,length_class);
% CLASS: ngl
length_class = positions(8:9);
VIF_ngl      = VIF_selection(regressors,feats,length_class);

% final indexes vector
VIF_index = [VIF_stat;VIF_ivh;VIF_ih;VIF_cm;VIF_rlm;...
             VIF_szm;VIF_dzm;VIF_ngl];

% selection of features
newRegressors = regressors(:,[1:7,table2array(VIF_index)']);

[ClinicalResults(5).regressionCoef, ClinicalResults(5).prediction, ...
ClinicalResults(5).rho, ClinicalResults(5).pvalue] = LinearRegressionModel(table2array(newRegressors),CLINICAL(:,5));

%% UPDRSTotal
UPDRSTotal = [clinical_wo_covariates(:,6),clinical_age(:,6), ...
             clinical_gender(:,6),clinical_edy(:,6),clinical_height(:,6), ...
             clinical_weight(:,6),clinical_BMI(:,6)];
UPDRSTotal = array2table(UPDRSTotal,"VariableNames",covariates_names,"RowNames",feature_names);

idx = significance_indexes(UPDRSTotal, nfeatures);

regressors = zscore([DemographicsPatients.GenderM0F1, ...
             DemographicsPatients.EducationYears, ...
             DemographicsPatients.Height, DemographicsPatients.Weight, ...
             DemographicsPatients.BMI,  table2array(RadiomicsPatients(:,idx))]);
regressors = [ones(33,1), regressors];

[ClinicalResults(6).regressionCoef, ClinicalResults(6).prediction, ...
ClinicalResults(6).rho, ClinicalResults(6).pvalue] = LinearRegressionModel(regressors,CLINICAL(:,6));

%% NMSQ
NMSQ = [clinical_wo_covariates(:,7),clinical_age(:,7),clinical_gender(:,7),...
       clinical_edy(:,7),clinical_height(:,7),clinical_weight(:,7),clinical_BMI(:,7)];
NMSQ = array2table(NMSQ,"VariableNames",covariates_names,"RowNames",feature_names);

idx = significance_indexes(NMSQ, nfeatures);

regressors = zscore([DemographicsPatients.Age, DemographicsPatients.EducationYears, ...
             DemographicsPatients.Height, table2array(RadiomicsPatients(:,idx))]);
regressors = [ones(33,1), regressors];

[ClinicalResults(7).regressionCoef, ClinicalResults(7).prediction, ...
ClinicalResults(7).rho, ClinicalResults(7).pvalue] = LinearRegressionModel(regressors,CLINICAL(:,7));

%% MMSE
MMSE = [clinical_wo_covariates(:,8),clinical_age(:,8),clinical_gender(:,8), ...
       clinical_edy(:,8),clinical_height(:,8),clinical_weight(:,8),clinical_BMI(:,8)];
MMSE = array2table(MMSE,"VariableNames",covariates_names,"RowNames",feature_names);

idx = significance_indexes(MMSE, nfeatures);

regressors = zscore([DemographicsPatients.Age, DemographicsPatients.GenderM0F1, ...
             DemographicsPatients.Weight, DemographicsPatients.BMI, ...
             table2array(RadiomicsPatients(:,idx))]);
regressors = [ones(33,1), regressors];

[ClinicalResults(8).regressionCoef, ClinicalResults(8).prediction, ...
ClinicalResults(8).rho, ClinicalResults(8).pvalue] = LinearRegressionModel(regressors,CLINICAL(:,8));

%% MoCA
MoCA = [clinical_wo_covariates(:,9),clinical_age(:,9),clinical_gender(:,9), ...
       clinical_edy(:,9),clinical_height(:,9),clinical_weight(:,9),clinical_BMI(:,9)];
MoCA = array2table(MoCA,"VariableNames",covariates_names,"RowNames",feature_names);

idx = significance_indexes(MoCA, nfeatures);

regressors = zscore([DemographicsPatients.EducationYears, ...
             DemographicsPatients.Height, DemographicsPatients.BMI, ...
             table2array(RadiomicsPatients(:,idx))]);
regressors = [ones(33,1), regressors];

[ClinicalResults(9).regressionCoef, ClinicalResults(9).prediction, ...
ClinicalResults(9).rho, ClinicalResults(9).pvalue] = LinearRegressionModel(regressors,CLINICAL(:,9));

%% cleaning
close all;
