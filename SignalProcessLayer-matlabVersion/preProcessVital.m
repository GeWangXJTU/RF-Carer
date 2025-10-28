function [breath_gt,record_I1,record_Q1,record_P1,record_I2,record_Q2,record_P2]=preProcessVital(str,en,cut_sample,CIRMatrix,Breath_sig_final)

    [len,wid]=size(CIRMatrix);

    windows=3;
    Fs=40;
    heartRate=[1,2];
    respRate=[0.2,0.4];
    sample=len;
    max_resp=zeros(1,wid-windows);
    max_heart=zeros(1,wid-windows);
    max_peo=ones(1,wid-windows);
    
    f=Fs*(0:(sample/2))/sample;
    heart_a=min(find(f>=heartRate(1)));
    heart_b=max(find(f<=heartRate(2)));
    resp_a=min(find(f>=respRate(1)));
    resp_b=max(find(f<=respRate(2)));   
    
    gap=1480;
    f_gap=Fs*(0:(gap/2))/gap;
    gap_a=min(find(f_gap>=heartRate(1)));
    gap_b=max(find(f_gap<=heartRate(2)));
    
%     figure,
%     mesh(abs(CIRMatrix));

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  ground truth
    breath_gt=processGT(Breath_sig_final,heart_a,heart_b,gap,gap_a,gap_b); 

    for i=1:wid-windows
        realPlus = ones(sample,1);
        imagPlus = ones(sample,1);
        fft_peo = zeros(sample,1);
        fft_mul = zeros(sample,1);
        
        temp = ones(sample,1);
        
        for j=1:windows
            rawSig=CIRMatrix(:,i+j);
            
            realSig=real(rawSig);
            imagSig=imag(rawSig);
            realFFT=fft(realSig);
            imagFFT=fft(imagSig);
            realPlus=realPlus.*abs(realFFT/sample);
            imagPlus=imagPlus.*abs(imagFFT/sample); 
            
            fft_peo=fft_peo+realPlus*2+imagPlus*2;
            fft_mul=fft_mul+realPlus*2.*imagPlus*2;
            
            temp=temp.*fft(rawSig);

        end                 
        max_resp(1,i)=max(fft_peo(resp_a:resp_b));
        max_heart(1,i)=max(fft_peo(heart_a:heart_b));
        max_peo(1,i)=max(fft_mul(heart_a:heart_b));
    end

    [val_rp,pos_rp] = sort(max_resp,'descend');

    Rpidx=pos_rp;
    Rpidx(Rpidx<9)=[];
    Rpidx(Rpidx>25)=[];
    

    Rp_idx = Rpidx(1);


    if max_resp(Rp_idx+1)>max_resp(Rp_idx-1)
        Rp_idx_2=Rp_idx+1;
    else
        Rp_idx_2=Rp_idx-1;
    end
    
    RpIdxCIR_1= Rp_idx+ceil(windows/2);
    RpIdxCIR_2= Rp_idx_2+ceil(windows/2);
    
    
    raw_rp_sig_1 = CIRMatrix(:,RpIdxCIR_1);
    raw_rp_sig_2 = CIRMatrix(:,RpIdxCIR_2);
   
    raw_rp_sig_2 = CIRMatrix(:,RpIdxCIR_2);
    
    
    %bandpass filter:respiration
    wp=[0.1/(sample/2),(heartRate(1)*100)/(sample/2)];
    N=128;
    b=fir1(N,wp,blackman(N+1));
    sig_rp_tmp1=filtfilt(b,1,raw_rp_sig_1);
    sig_rp_tmp1=sig_rp_tmp1';
    seg_rp_1=sig_rp_tmp1(str:en);
    seg_raw_rp_1=raw_rp_sig_1(str:en);
    
    sig_rp_tmp2=filtfilt(b,1,raw_rp_sig_2);
    sig_rp_tmp2=sig_rp_tmp2';
    seg_rp_2=sig_rp_tmp2(str:en);  
    
    [trans_rp_sig1,trans_rp_sig2,raw_sig,raw_sig2]=transformVectorNew(cut_sample,seg_rp_1,seg_rp_2);
%      
    [record_I1,record_Q1,record_P1,record_I2,record_Q2,record_P2]=DualRotate(trans_rp_sig1,trans_rp_sig2,raw_rp_sig_1(str:en),raw_rp_sig_2(str:en),cut_sample,breath_gt);

    lineWid=3;
    test_len=600;
    for idx=1:3

        sig_gt=mapminmax(breath_gt(idx,1:test_len),0,1);
        sig_raw=mapminmax(real(seg_raw_rp_1((idx-1)*1480+1:(idx-1)*1480+test_len))',0,1);
        sig_our=mapminmax(real(trans_rp_sig1(idx,1:test_len)),0,1);
        figure,
        subplot(2,1,1),plot(sig_gt,'-.','Linewidth',1.5);
        hold on,plot(sig_raw,'k','Linewidth',lineWid);
        legend('GT','Raw data');
        ylabel('Amplitude');
        set(gca, 'FontSize', 30, 'Linewidth', 1);
        box on;

        subplot(2,1,2),plot(sig_gt,'-.','Linewidth',1.5);
        hold on,plot(sig_our,'Linewidth',lineWid);
        legend('GT','RF-Carer*');
        xlabel('Sample');
        ylabel('Amplitude');
        set(gcf, 'Position', [200, 200, 900, 700]);
        set(gca, 'FontSize', 30, 'Linewidth', 1);
        box on;
        
        sig_raw=mapminmax(imag(seg_raw_rp_1((idx-1)*1480+1:(idx-1)*1480+test_len))',0,1);
        sig_our=mapminmax(imag(trans_rp_sig1(idx,1:test_len)),0,1);

        figure,
        subplot(2,1,1),plot(sig_gt,'-.','Linewidth',1.5);
        hold on,plot(sig_raw,'k','Linewidth',lineWid);
        legend('GT','Raw data');
        ylabel('Amplitude');
        set(gca, 'FontSize', 30, 'Linewidth', 1);
        box on;

        subplot(2,1,2),plot(sig_gt,'-.','Linewidth',1.5);
        hold on,plot(sig_our,'Linewidth',lineWid);
        legend('GT','RF-Carer*');
        xlabel('Sample');
        ylabel('Amplitude');
        set(gcf, 'Position', [200, 200, 900, 700]);
        set(gca, 'FontSize', 30, 'Linewidth', 1);
        box on;
    end


end