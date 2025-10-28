function [Breath_sig_final]=generate_gt_B(rootdir,volun,agl,states,doc)
fps=40;
str_second=8;
en_second=-1;
str=str_second*fps+1;
en=4800+fps*en_second;
t=37;
count=floor(floor((en-str+1)/fps)/t);

seconds=120;
fps_uwb=40;
fps_ecg=125;
fps_bre=50;
sample=fps_uwb*seconds;
sample_bre=fps_bre*seconds;

%% Read files

    FilePath=[rootdir,'\',volun,'\',agl,'\',states,'\'];
    filepath=dir(fullfile(FilePath,'*.mat'));
    fileNames={filepath.name}';
    file=fileNames{doc};
    FileName=[FilePath,file];

    CIRMatrix = load(FileName).CIRMatrix;
    BreathData = load(FileName).BreathData;
    TimeStamp = load(FileName).TimeStamp;
    
    % % % % % % % % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

    Breath_sig=[];

    for i=1:sample
        % ECG_sig=[ECG_sig,ECGData{i}];
        Breath_sig=[Breath_sig,BreathData{i}];
    end
    % ECG_sig(ECG_sig==-1000)=[];
    % ECG_sig_pro = eliminateError(ECG_sig,10);
    
    Breath_sig(Breath_sig==-1000)=[];
    Breath_sig_pro = removeError(Breath_sig);
 
    flag=0;
    
    if length(Breath_sig_pro)==sample_bre
        flag=flag+1;
    elseif length(Breath_sig_pro)>sample_bre
        Breath_sig_pro(sample_bre+1:end)=[];
        flag=flag+1;
    elseif (length(Breath_sig_pro)>sample_bre-30) && (length(Breath_sig_pro)<sample_bre)
        tmp2=zeros(1, sample_bre-length(Breath_sig_pro));
        Breath_sig_pro=[Breath_sig_pro,tmp2];
        flag=flag+1;
    end
    
    if flag==1

        Breath_sig_pro=resample(Breath_sig_pro,sample,sample_bre);
        Breath_sig_final=Breath_sig_pro(str_second*fps_uwb+1:fps_uwb*(seconds+en_second));
    else
        Breath_sig_final=0;

    end

   end

