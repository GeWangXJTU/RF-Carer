function [record_I1,record_Q1,record_P1,record_I2,record_Q2,record_P2]=DualRotate(sig1,sig2,raw_sig1,raw_sig2,sample,breath_gt)
 [wid,len]=size(sig1);
 record_I1=zeros(2*wid,len);
 record_Q1=zeros(2*wid,len);
 record_P1=zeros(2*wid,len);
 record_I2=zeros(2*wid,len);
 record_Q2=zeros(2*wid,len);
 record_P2=zeros(2*wid,len);
 str=-0.5*pi;
 
 %mean filter
 window=30;
 len_sig=length(raw_sig1);
 agl_sig1=angle(raw_sig1);
 agl_sig2=angle(raw_sig2);
 agl_sig1=unwrap(agl_sig1);
 agl_sig2=unwrap(agl_sig2);
 pro_sig1(1:floor(window/2))=agl_sig1(1:floor(window/2));
 pro_sig1(len_sig-floor(window/2)+1:len_sig)=agl_sig1(len_sig-floor(window/2)+1:len_sig);
 pro_sig2(1:floor(window/2))=agl_sig2(1:floor(window/2));
 pro_sig2(len_sig-floor(window/2)+1:len_sig)=agl_sig2(len_sig-floor(window/2)+1:len_sig);
 for x=floor(window/2)+1:len_sig-floor(window/2)
     pro_sig1(x)=mean(agl_sig1(x-floor(window/2):x+floor(window/2)));
     pro_sig2(x)=mean(agl_sig2(x-floor(window/2):x+floor(window/2)));
 end

 
for u=1:wid
    cut1=sig1(u,:);
    cut2=sig2(u,:);
    ref1=pro_sig1(1+(u-1)*sample:u*sample);
    ref2=pro_sig2(1+(u-1)*sample:u*sample);
    
    ref=breath_gt(u,:);
%     ref = raw_sig(u,:);

    I_sig1=real(cut1);
    Q_sig1=imag(cut1);
    I_sig2=real(cut2);
    Q_sig2=imag(cut2);    
    count=1;
    
    for i=str:0.01:0.5*pi
        tmp_I1=I_sig1*cos(i)+Q_sig1*sin(i);
        tmp_Q1=I_sig1*sin(i)*-1+Q_sig1*cos(i);
        tmp_I2=I_sig2*cos(i)+Q_sig2*sin(i);
        tmp_Q2=I_sig2*sin(i)*-1+Q_sig2*cos(i);

        range_I=(max(tmp_I1)-min(tmp_I1))+(max(tmp_I2)-min(tmp_I2));
        range_Q=(max(tmp_Q1)-min(tmp_Q1))+(max(tmp_Q2)-min(tmp_Q2));

        rotate(count)=range_I*range_Q*(exp(-1*abs(i)));
        count=count+1;
    end
    rotate(end)=[];
    [val,idx]=max(rotate);
    
    temp_idx=str+0.01*(idx-1);
    I=I_sig1*cos(temp_idx)+Q_sig1*sin(temp_idx);
    Q=I_sig1*sin(temp_idx)*-1+Q_sig1*cos(temp_idx);
    
    flag_I=1;
    flag_Q=1;
    flat_P=1;
    [min_val,min_idx]=min(ref(1:200));
    [max_val,max_idx]=max(ref(1:200));
    corcof_I=I(max_idx)-I(min_idx);
    corcof_Q=Q(max_idx)-Q(min_idx);
    corcof_P=ref1(max_idx)-ref1(min_idx);
    if corcof_I<0
        flag_I=-1;
    end
    if corcof_Q<0
        flag_Q=-1;
    end
    if corcof_P<0
        flat_P=-1;
    end
    
    tmp_I=mapminmax(flag_I*I,0,1);
    tmp_Q=mapminmax(flag_Q*Q,0,1);
    tmp_P=mapminmax(flat_P*ref1,0,1);
    record_I1(u,:)=tmp_I;
    record_Q1(u,:)=tmp_Q;
    record_P1(u,:)=tmp_P;
    
    I=I_sig2*cos(temp_idx)+Q_sig2*sin(temp_idx);
    Q=I_sig2*sin(temp_idx)*-1+Q_sig2*cos(temp_idx);
    
    tmp_I=mapminmax(flag_I*I,0,1);
    tmp_Q=mapminmax(flag_Q*Q,0,1);
    tmp_P=mapminmax(flat_P*ref2,0,1);
    record_I2(u,:)=tmp_I;
    record_Q2(u,:)=tmp_Q;
    record_P2(u,:)=tmp_P;
    

%     figure,plot(real(trans_1));
%     hold on,plot(real(trans_2));
%     figure,plot(imag(trans_1));
%     hold on,plot(imag(trans_2));
 end
 

end