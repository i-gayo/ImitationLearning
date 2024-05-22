%% Code for visualising data 


%% Reading action files
%ppo_basic_penalty
% Reading output actions file 
clear all

% TODO : Change PATHS to where your cSV files are saved!
MODEL_NAME = 'GS_1_1'
OUTPUT_ACTION_PATH = append('/Users/ianijirahmae/Documents/IPCAI/RESULTS/FINAL/', MODEL_NAME, '/_output_actions_test_1.csv')
PATIENT_NAMES_PATH = append('/Users/ianijirahmae/Documents/IPCAI/RESULTS/FINAL/', MODEL_NAME, '/_patient_namestest_1.csv')
ALL_INFO_PATH = append('/Users/ianijirahmae/Documents/IPCAI/RESULTS/FINAL/', MODEL_NAME, '/IPCAI_GS_1_1_NEW.csv')

opts = detectImportOptions(OUTPUT_ACTION_PATH);
opts.EmptyLineRule = 'read';
output_actions = readtable(OUTPUT_ACTION_PATH, opts);

% Reading patietn_names file 
patient_names = readtable(PATIENT_NAMES_PATH, 'NumHeaderLines', 1, 'Delimiter', ',') ; %, 'Delimiter', 'space');
all_info = readtable(ALL_INFO_PATH, 'NumHeaderLines', 1, 'Delimiter', ',')
%opts = detectImportOptions('/Users/ianijirahmae/Documents/PhD_project/Experiments/Multiple_lesions/MULTIPLE_hyperparameter_1/_output_actionstrain_0.csv')


%%% TODO: DEAL WITH ERROR: 
% lesion_mask is 5d int64, instead of 200x200x96 so change this when svaing
% new volume!
%% PLOTTING PROSTATE AND LESION MASKS in for loop nsearse

% optimise policy, rather than predictions (we'r enot correctly predicting
% -> we want to plan and improve things!!!

% TODO: Change folders path to where your lesion and prostate masks are
% saved! 
LESION_FOLDERS = append('/Users/ianijirahmae/Documents/IPCAI/RESULTS/FINAL/', MODEL_NAME, '/lesion_masks')
PROSTATE_FOLDERS = append('/Users/ianijirahmae/Documents/IPCAI/RESULTS/FINAL/', MODEL_NAME, '/prostate_masks')
PROSTATE_COLOUR = [1 0.4 0.5]
LESION_COLOUR = [0.4 1 0.8];


needle_colours = [0.05, 0, 0; 
    0.1, 0, 0.0;
    0.15, 0, 0;
    0.2, 0, 0.0;
    0.25, 0, 0.0;
     0.3 ,0, 0.00;
     0.35, 0, 0; 
     0.4, 0, 0.00;
     0.45, 0., 0;
     0.5 ,0. ,0.00;
     0.55 ,0., 0.00;
     0.60 ,0. ,0.00;
     0.65, 0., 0.00;
     0.70 ,0., 0.00;
     0.75, 0., 0.0;
     0.80 ,0. ,0.00;
     0.85, 0, 0;
     0.90 ,0., 0.00;
     0.95, 0., 0; 
     1.0 ,0, 0.0 ];

needle_colours = needle_colours(:, [3,2,1]);


x_vals = 1:1:200;
y_vals = 1:1:200;
z_vals = 1:1:96;
[X,Y,Z] = meshgrid(x_vals,y_vals,z_vals); 


%camlight;
%lighting gouraud;
i
close all

action_idx = 1; 
patient_idx = 1;

IMG_FOLDER = append('/Users/ianijirahmae/Documents/PhD_Project/Experiments/IPCAI/Figures/',MODEL_NAME, '/')
SUCCESS_FOLD = append(IMG_FOLDER, '/SUCCESS')
FAIL_FOLD = append(IMG_FOLDER, '/FAIL')
mkdir(IMG_FOLDER)
mkdir(SUCCESS_FOLD)
mkdir(FAIL_FOLD)


for cases = 1:270
    
    episode_reward = 0

    figure()
    axis off 
    % Load masks in 
    % +2 for rl, +1 for il
    patient_name = string(patient_names{cases+1, :})
    
    patient = string(all_info{cases, 2})
    lesion = string(double(all_info{cases, 3})+1) % MIGHT NEED TO REMOVE THIS LATER
    lesion_name = append(lesion, '_', patient)
    lesion_mask = squeeze(niftiread(fullfile(LESION_FOLDERS,  lesion_name)));
    prostate_mask = niftiread(fullfile(PROSTATE_FOLDERS, patient_name));
    
    %transpose masks
    lesion_mask = permute(lesion_mask, [2,1,3]);
    prostate_mask = permute(prostate_mask, [2,1,3]);

    % Plot prostate and lesion masks 
    output_prostate = display_object(X,Y,Z, prostate_mask, PROSTATE_COLOUR)
    alpha(0.2)
    hold on; 
    output_lesion = display_object(X,Y,Z, lesion_mask, LESION_COLOUR)

    %material shiny
    view(89.7, 90)
    camlight
    lighting gouraud
    %axis equal off

    %view(89.7, 120)
    
    % Compute COM of prostate for needles 
    com_prostate = compute_com(prostate_mask); 
    com_prostate(3) = 0;
    
    %%

    % Plot needles 
    while  ~isnan(output_actions.x_grid(action_idx))

        action = output_actions{action_idx, ["y_grid", "x_grid", "depth"]} .* [2, 2, 1]; % ensures dimensions match the prostate masks
        action_plot = action + com_prostate; 

        % Only plot if needle was fired, else leave blank  id
        if action_plot(3) ~= 0
            %correct VISUALISATION : plot(1,2,3) NOT plot(2,1,3)
            plot3([action_plot(1), action_plot(1)], [action_plot(2), action_plot(2)] ,[0, action_plot(3)], 'k*-', 'Linewidth' , 5)
        else
            plot3([action_plot(1), action_plot(1)], [action_plot(2), action_plot(2)] ,[0, action_plot(3)], 'k*-', 'Linewidth' , 5)
        end

        episode_reward = episode_reward + output_actions{action_idx, "reward"}; 
        
        % Increase action index
        action_idx = action_idx + 1;         
        pause(0.05)
        colororder(needle_colours);
    end

    % check episode reward 
    if episode_reward < 0
       img_title = append(IMG_FOLDER, 'FAIL/', num2str(cases), '.png')
       saveas(gcf, img_title)
       disp(['FAIL is: [' num2str(cases) ']']) ;
    else
        img_title = append(IMG_FOLDER, 'SUCCESS/', num2str(cases), '.png')
        saveas(gcf, img_title)
        disp(['SUCCESS is: [' num2str(cases) ']']) ;
    end 

    %Name
    %view(-89.667222871761396,-90)
    fig_title = append(IMG_FOLDER, num2str(cases), '.png')
    saveas(gcf, fig_title)
    
    %Skip nan line to move onto next case
    action_idx = action_idx + 1;
    
    close
    
    pause(0.1)

    
end 

%% EXTRA CODE

% INPUT PATIENT NAMES 
PATIENT_NAME = 'Patient001061633_study_0.nii.gz';

LESION_FOLDERS = '/Users/ianijirahmae/Documents/DATASETS/Data_by_modality/lesion'
PROSTATE_FOLDERS = '/Users/ianijirahmae/Documents/DATASETS/Data_by_modality/prostate_mask'

PROSTATE_COLOUR = [1 0.4 0.5];
LESION_COLOUR = [0.4 1 0.8];

lesion_mask = niftiread(LESION_PATH);
prostate_mask = niftiread(PROSTATE_PATH);

com_prostate = compute_com(prostate_mask); 
com_prostate(3) = 0;
action = output_actions{2, ["y_grid", "x_grid", "depth"]} .* [-2, 2, 1]; % ensures dimensions match the prostate masks
%action = [0, 30, 50] .* [-2,2,1]; %action has to be in order of y_grid, x_grid, z_depth
action_plot = action + com_prostate; 
plot3([action_plot(1), action_plot(1)], [action_plot(2), action_plot(2)] ,[0, action_plot(3)], '*-', 'Linewidth' , 8) 

action = output_actions{1, ["y_grid", "x_grid", "depth"]} .* [-2, 2, 1]; % ensures dimensions match the prostate masks
%action = [0, 30, 50] .* [-2,2,1]; %action has to be in order of y_grid, x_grid, z_depth
action_plot = action + com_prostate; 
plot3([action_plot(1), action_plot(1)], [action_plot(2), action_plot(2)] ,[0, action_plot(3)], '*-', 'Linewidth' , 8) 

action = output_actions{3, ["y_grid", "x_grid", "depth"]} .* [-2, 2, 1]; % ensures dimensions match the prostate masks
%action = [0, 30, 50] .* [-2,2,1]; %action has to be in order of y_grid, x_grid, z_depth
action_plot = action + com_prostate; 
plot3([action_plot(1), action_plot(1)], [action_plot(2), action_plot(2)] ,[0, action_plot(3)], '*-', 'Linewidth' , 8) 



view(89.7, 90) % for 2d flat view 
view(89.7, 70) % for 3d view 



%% FUNCTIONS TO USE FOR PLOTS 

% s = isosurface(X,Y,Z,patient_prostate);
% p = patch(s);
% isonormals(X,Y,Z,patient_prostate,p)
% view(3);
% set(p,'FaceColor',[0.5 1 0.7]);  
% set(p,'EdgeColor','none');
% alpha(.5)
% 
% hold on 
% 
% s = isosurface(X,Y,Z,patient_vol);
% p = patch(s);
% isonormals(X,Y,Z,patient_vol,p)
% view(3);
% set(p,'FaceColor',[1 0.4 0.5]);  
% set(p,'EdgeColor','none');
% %alpha(1)
% 
% 
% camlight;
% lighting gouraud;
%%

%output_prostate = display_object(X,Y,Z, prostate_mask, [0.5 1 0.7])

%% FUNCTIONS FOR USE 
function output_patch = display_object(X,Y,Z, volume, face_colour)
%% A function which displays the object given 
% Input params : X,Y,Z meshgrid, mask_volume and which object type it is

    s = isosurface(X,Y,Z,volume);
    output_patch = patch(s);
    isonormals(X,Y,Z,volume,output_patch)
    view(3);
    set(output_patch,'FaceColor',face_colour);  
    set(output_patch,'EdgeColor','none');
    
    %alpha(alpha_val)

end 

function com_coords = compute_com(prostate_mask)
%% A function that computes the centre of mass of the prostate volume 

    idx_nonzero = find(prostate_mask);
    [x_coords, y_coords, z_coords] = ind2sub(size(prostate_mask), idx_nonzero);

    com_x = round(mean(x_coords));
    com_y = round(mean(y_coords));
    com_z = round(mean(z_coords));

    com_coords = [com_x, com_y, com_z];

end 
