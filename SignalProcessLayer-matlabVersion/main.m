close all;
clear all;
a=6;

%% Set the method of loading CIRMatrix
rootdir='C:\Users\gewan\Desktop\RF-Carer\RF-Carer-Dataset\Raw\1-Angle&5-User&12-Condition\';
volun='V1-M';
agl='30';
% states='static';
% states='moving';
states='dynamic';


fps=40; %Sampling rate
fps_ecg=125;
fps_bre=50;
str_second=8;
en_second=1;
str=str_second*fps+1;
en=4800-fps*en_second;
t=37; %second
count=floor(floor((en-str+1)/fps)/t);
cut_sample=t*fps;
drop_t=5;
cut_drop=drop_t*fps;
sample_uwb=fps*count*t;
sample_ecg=fps_ecg*count*t;
sample_bre=fps_bre*count*t;

resolution=fps/cut_sample;
rangeK=15;


for doc=1:5

    FilePath=[rootdir,'\',volun,'\',agl,'\',states,'\'];
    
    filepath=dir(fullfile(FilePath,'*.mat'));
    fileNames={filepath.name};
    file=fileNames{doc};
    FileName=[FilePath,file];
    savedir=['F:\MyResearch\UWBH\UWB Network\Respiration_NetWork\FinalDataset\train\'];
    
    
    [Breath_sig_final]=generate_gt_B(rootdir,volun,agl,states,doc);
    
    CIRMatrix = load(FileName).CIRMatrix;
    
    % figure,
    % mesh(abs(CIRMatrix));
    % colormap();
    % view(0,90);

end
    
if length(Breath_sig_final)>=sample_uwb
    
    [breath_gt,record_I1,record_Q1,record_P1,record_I2,record_Q2,record_P2]=preProcessVital(str,en,cut_sample,CIRMatrix,Breath_sig_final);
   
end

