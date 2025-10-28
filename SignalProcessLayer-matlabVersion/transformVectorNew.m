function [final_sig1,final_sig2,raw_sig,raw_sig2]=transformVectorNew(cut_sample,sig,sig2)
    win=4;
    step=1;

    s0=[];
    x_b1=zeros(1,floor((length(sig)-win)/step));
    y_b1=zeros(1,floor((length(sig)-win)/step)); 
    x_b2=zeros(1,floor((length(sig)-win)/step)); 
    y_b2=zeros(1,floor((length(sig)-win)/step)); 
    theta=zeros(1,floor((length(sig)-win)/step)); 
    alpha_x=zeros(1,floor((length(sig)-win)/step)); 
    alpha_y=zeros(1,floor((length(sig)-win)/step));
    count=1;
    % options = optimoptions('fsolve','Display','off');
    % options = optimoptions('fmincon', 'Algorithm', 'sqp'); % 使用序列二次规划算法
    
    for i=1:step:length(sig)-win
       p(1,1:win)=real(sig(i:i+win-1));%x_1
       p(2,1:win)=imag(sig(i:i+win-1));%y_1
       p(3,1:win)=real(sig2(i:i+win-1));%x_2
       p(4,1:win)=imag(sig2(i:i+win-1));%y_2
       
       fun=@(s)paramfun(s,p);
       
       if isempty(s0)
           sx_b1 = min(real(sig));
           sy_b1 = min(imag(sig));
           sx_b2 = min(real(sig2));
           sy_b2 = min(imag(sig2));
                                    
           if  (imag(sig2(i))<0) 
               stheta=angle(sig(i))-angle(sig2(i));
           else
               stheta=angle(sig2(i))-angle(sig(i));
           end

%            stheta=pi/4;
           salpha_x=(max(real(sig))-min(real(sig)))/(max(real(sig2))-min(real(sig2)));
           salpha_y=(max(imag(sig))-min(imag(sig)))/(max(imag(sig2))-min(imag(sig2)));

           s0=[sx_b1,sy_b1,sx_b2,sy_b2,stheta,salpha_x,salpha_y];
       else
           s0=s;
       end

       % s=fsolve(fun,s0,options);
       s=fsolve(fun,s0);
       
       if ~isempty(s)
           % s=[x_b1 y_b1 x_b2 y_b2 theta alpha_x alpha_y]
            x_b1(1,count)=s(1);
            y_b1(1,count)=s(2);
            x_b2(1,count)=s(3);
            y_b2(1,count)=s(4); 
            theta(1,count)=s(5);
            alpha_x(1,count)=s(6);
            alpha_y(1,count)=s(7);
            count=count+1;
       else
           continue;
       end
    end

    % Part signals: algorithm 0
    count=1;    
    
    seg_sig=sig;
    seg_sig2=sig2;
    
%     seg_phase=raw_sig;
%     seg_phase2=raw_sig2;
    seg_phase=sig;
    seg_phase2=sig2;
    
    for v=1:cut_sample:length(seg_sig)-cut_sample+1
        compare_sig=seg_sig(v:v+cut_sample-1);
        compare_sig2=seg_sig2(v:v+cut_sample-1);
        
        % raw_phase(count,1:cut_sample)=angle(seg_phase(v:v+cut_sample-1));
        % raw_phase2(count,1:cut_sample)=angle(seg_phase2(v:v+cut_sample-1));

        raw_sig(count,1:cut_sample)=sig(v:v+cut_sample-1);
        raw_sig2(count,1:cut_sample)=sig2(v:v+cut_sample-1);
    
        residual=zeros(1,length(x_b1));
        trans_sig=zeros(1,length(cut_sample));
        trans_sig2=zeros(1,length(cut_sample));
        for j=1:length(x_b1)
            try_x_b1=x_b1(j);
            try_y_b1=y_b1(j);
            try_x_b2=x_b2(j);
            try_y_b2=y_b2(j); 
            try_t=theta(j);
            try_ax=alpha_x(j);
            try_ay=alpha_y(j);
            tmp_sig=compare_sig-complex(try_x_b1,try_y_b1);
            tmp_sig2_x=try_ax*((real(compare_sig2)-try_x_b2)*cos(try_t)+(imag(compare_sig2)-try_y_b2)*sin(try_t));
            tmp_sig2_y=try_ay*((real(compare_sig2)-try_x_b2)*sin(try_t)*-1+(imag(compare_sig2)-try_y_b2)*cos(try_t));
            tmp_sig2=complex(tmp_sig2_x,tmp_sig2_y);
            residual(j)=sum(abs(tmp_sig-tmp_sig2));
        end
        
        [val,idx]=min(residual);      
        min_x_b1=x_b1(idx);
        min_y_b1=y_b1(idx);
        min_x_b2=x_b2(idx);
        min_y_b2=y_b2(idx);
        min_t=theta(idx);
        min_ax=alpha_x(idx);
        min_ay=alpha_y(idx);

        trans_sig=compare_sig-complex(min_x_b1,min_y_b1);
        trans_sig2_x=min_ax*((real(compare_sig2)-min_x_b2)*cos(min_t)+(imag(compare_sig2)-min_y_b2)*sin(min_t));
        trans_sig2_y=min_ay*((real(compare_sig2)-min_x_b2)*sin(min_t)*-1+(imag(compare_sig2)-min_y_b2)*cos(min_t));
        trans_sig2=complex(trans_sig2_x,trans_sig2_y);

%         trans_sig=compare_sig;
%         trans_sig2_x=min_ax*((real(compare_sig2)-min_x_b2)*cos(min_t)+(imag(compare_sig2)-min_y_b2)*sin(min_t));
%         trans_sig2_y=min_ay*((real(compare_sig2)-min_x_b2)*sin(min_t)*-1+(imag(compare_sig2)-min_y_b2)*cos(min_t));
%         trans_sig2=complex(trans_sig2_x,trans_sig2_y)+complex(min_x_b1,min_y_b1);
        
        temp_center=complex(median(real(trans_sig)),median(imag(trans_sig)));
        trans_sig=trans_sig-temp_center;
        trans_sig2=trans_sig2-temp_center;
        
        final_sig1(count,1:cut_sample)=trans_sig;
        final_sig2(count,1:cut_sample)=trans_sig2;


% % figure,
%         plot(trans_sig,'.');
%         hold on,plot(trans_sig2,'ro');


%     figure,plot(trans_sig,'b.');
%     hold on,plot(trans_sig2,'ro');

        count=count+1;
    end
    

    
% %% Part signals: algorithm 1
% %Calculate initial parameters
% gap=10;
% seg_sig_0=sig(1:gap);
% seg_sig2_0=sig2(1:gap);
% 
% compare_sig=sig(1:1000);
% compare_sig2=sig(1:1000);
% residual=zeros(1,length(x_b1));
% for j=1:length(x_b1)
%     try_x_b1=x_b1(j);
%     try_y_b1=y_b1(j);
%     try_x_b2=x_b2(j);
%     try_y_b2=y_b2(j);
%     try_t=theta(j);
%     try_ax=alpha_x(j);
%     try_ay=alpha_y(j);
% 
%     tmp_sig=compare_sig-complex(try_x_b1,try_y_b1);
%     tmp_sig2_x=try_ax*((real(compare_sig2)-try_x_b2)*cos(try_t)+(imag(compare_sig2)-try_y_b2)*sin(try_t));
%     tmp_sig2_y=try_ay*((real(compare_sig2)-try_x_b2)*sin(try_t)*-1+(imag(compare_sig2)-try_y_b2)*cos(try_t));
%     tmp_sig2=complex(tmp_sig2_x,tmp_sig2_y);
%     residual(j)=sum(abs(tmp_sig-tmp_sig2));
% end
% 
% [val,idx]=min(residual);
% min_x_b1=x_b1(idx);
% min_y_b1=y_b1(idx);
% min_x_b2=x_b2(idx);
% min_y_b2=y_b2(idx);
% min_t=theta(idx);
% min_ax=alpha_x(idx);
% min_ay=alpha_y(idx);
% 
% trans_sig=seg_sig_0-complex(min_x_b1,min_y_b1);
% trans_sig2_x=min_ax*((real(seg_sig2_0)-min_x_b2)*cos(min_t)+(imag(seg_sig2_0)-min_y_b2)*sin(min_t));
% trans_sig2_y=min_ay*((real(seg_sig2_0)-min_x_b2)*sin(min_t)*-1+(imag(seg_sig2_0)-min_y_b2)*cos(min_t));
% trans_sig2=complex(trans_sig2_x,trans_sig2_y);
% 
% tmp_sig=sig-complex(min_x_b1,min_y_b1);
% mean_I=mean(real(tmp_sig));
% mean_Q=mean(imag(tmp_sig));
% std_I=std(real(tmp_sig));
% std_Q=std(imag(tmp_sig));
% 
% k=1;
% dif_I_min=mean_I-k*std_I;
% dif_I_max=mean_I+k*std_I;
% dif_Q_min=mean_Q-k*std_Q;
% dif_Q_max=mean_Q+k*std_Q;
% 
% for v=gap+1:length(x_b1)
%     point=sig(v)-complex(min_x_b1,min_y_b1);
%     dif_I=real(point)-real(trans_sig(end));
%     dif_Q=imag(point)-imag(trans_sig(end));
%     if (dif_I>dif_I_max | dif_I<dif_I_min | dif_Q>dif_Q_max | dif_Q<dif_Q_min)
%         %recalculate the background
% %         seg_sig_0=sig(v:v+gap);
% %         seg_sig2_0=sig2(v:v+gap);
%         residual=ones(1,length(x_b1));
%         residual=residual*10e6;
%         for j=1:length(x_b1)
%             try_x_b1=x_b1(j);
%             try_y_b1=y_b1(j);
%             try_x_b2=x_b2(j);
%             try_y_b2=y_b2(j);
%             try_t=theta(j);
%             try_ax=alpha_x(j);
%             try_ay=alpha_y(j);
% 
%             tmp_point=sig(v)-complex(try_x_b1,try_y_b1);
%             dif_I_p=real(tmp_point)-real(trans_sig(end));
%             dif_Q_p=imag(tmp_point)-imag(trans_sig(end));
%             if (dif_I_p<dif_I_max && dif_I_p>dif_I_min && dif_Q_p<dif_Q_max && dif_Q_p>dif_Q_min)
%                 tmp_sig_p=sig(v)-complex(try_x_b1,try_y_b1);
%                 tmp_sig2_x_p=try_ax*((real(sig2(v))-try_x_b2)*cos(try_t)+(imag(sig2(v))-try_y_b2)*sin(try_t));
%                 tmp_sig2_y_p=try_ay*((real(sig2(v))-try_x_b2)*sin(try_t)*-1+(imag(sig2(v))-try_y_b2)*cos(try_t));
%                 tmp_sig2_p=complex(tmp_sig2_x_p,tmp_sig2_y_p);
%                 residual(j)=sum(abs(tmp_sig_p-tmp_sig2_p));                
%             end
%         end
%         
%         [val_p,idx_p]=min(residual);
%         if val_p<10e6
%             min_x_b1=x_b1(idx_p);
%             min_y_b1=y_b1(idx_p);
%             min_x_b2=x_b2(idx_p);
%             min_y_b2=y_b2(idx_p);
%             min_t=theta(idx_p);
%             min_ax=alpha_x(idx_p);
%             min_ay=alpha_y(idx_p);
% 
%             trans_sig=[trans_sig,sig(v)-complex(min_x_b1,min_y_b1)];
%             point_sig2_x=min_ax*((real(sig2(v))-min_x_b2)*cos(min_t)+(imag(sig2(v))-min_y_b2)*sin(min_t));
%             point_sig2_y=min_ay*((real(sig2(v))-min_x_b2)*sin(min_t)*-1+(imag(sig2(v))-min_y_b2)*cos(min_t));
%             trans_sig2=[trans_sig2,complex(point_sig2_x,point_sig2_y)];
%             disp(val_p);
%         else
%             trans_sig=[trans_sig,point];
%             point_sig2_x=min_ax*((real(sig2(v))-min_x_b2)*cos(min_t)+(imag(sig2(v))-min_y_b2)*sin(min_t));
%             point_sig2_y=min_ay*((real(sig2(v))-min_x_b2)*sin(min_t)*-1+(imag(sig2(v))-min_y_b2)*cos(min_t));
%             trans_sig2=[trans_sig2,complex(point_sig2_x,point_sig2_y)];
%         end   
%         
%     else
%         trans_sig=[trans_sig,point];
%         point_sig2_x=min_ax*((real(sig2(v))-min_x_b2)*cos(min_t)+(imag(sig2(v))-min_y_b2)*sin(min_t));
%         point_sig2_y=min_ay*((real(sig2(v))-min_x_b2)*sin(min_t)*-1+(imag(sig2(v))-min_y_b2)*cos(min_t));
%         trans_sig2=[trans_sig2,complex(point_sig2_x,point_sig2_y)];
%     end
% end
% 

% % Full signals: Algotirhm 2
%     for j=1:length(x_b1)
%         try_x_b1=x_b1(j);
%         try_y_b1=y_b1(j);
%         try_x_b2=x_b2(j);
%         try_y_b2=y_b2(j);
%         try_t=theta(j);
%         try_ax=alpha_x(j);
%         try_ay=alpha_y(j);
%         tmp_sig=sig-complex(try_x_b1,try_y_b1);
%         tmp_sig2_x=try_ax*((real(sig2)-try_x_b2)*cos(try_t)+(imag(sig2)-try_y_b2)*sin(try_t));
%         tmp_sig2_y=try_ay*((real(sig2)-try_x_b2)*sin(try_t)*-1+(imag(sig2)-try_y_b2)*cos(try_t));
%         tmp_sig2=complex(tmp_sig2_x,tmp_sig2_y);
%         residual(j)=sum(abs(tmp_sig-tmp_sig2));
%     end
%     
%     [val,idx]=min(residual);
%     min_x_b1=x_b1(idx);
%     min_y_b1=y_b1(idx);
%     min_x_b2=x_b2(idx);
%     min_y_b2=y_b2(idx);
%     min_t=theta(idx);
%     min_ax=alpha_x(idx);
%     min_ay=alpha_y(idx);
%     
%     trans_sig=sig-complex(min_x_b1,min_y_b1);
%     trans_sig2_x=min_ax*((real(sig2)-min_x_b2)*cos(min_t)+(imag(sig2)-min_y_b2)*sin(min_t));
%     trans_sig2_y=min_ay*((real(sig2)-min_x_b2)*sin(min_t)*-1+(imag(sig2)-min_y_b2)*cos(min_t));
%     trans_sig2=complex(trans_sig2_x,trans_sig2_y);

end

function F=paramfun(s,p)
       x_b1=s(1);
       y_b1=s(2);
       x_b2=s(3);
       y_b2=s(4); 
       theta=s(5);
       alpha_x=s(6);
       alpha_y=s(7);
       F=[alpha_x*((p(3,1)-x_b2)*cos(theta)+(p(4,1)-y_b2)*sin(theta))+x_b1-p(1,1) 
           alpha_x*((p(3,2)-x_b2)*cos(theta)+(p(4,2)-y_b2)*sin(theta))+x_b1-p(1,2) 
           alpha_x*((p(3,3)-x_b2)*cos(theta)+(p(4,3)-y_b2)*sin(theta))+x_b1-p(1,3) 
           alpha_x*((p(3,4)-x_b2)*cos(theta)+(p(4,4)-y_b2)*sin(theta))+x_b1-p(1,4) 
           alpha_y*((p(3,1)-x_b2)*sin(theta)*-1+(p(4,1)-y_b2)*cos(theta))+y_b1-p(2,1) 
           alpha_y*((p(3,2)-x_b2)*sin(theta)*-1+(p(4,2)-y_b2)*cos(theta))+y_b1-p(2,2) 
           alpha_y*((p(3,3)-x_b2)*sin(theta)*-1+(p(4,3)-y_b2)*cos(theta))+y_b1-p(2,3)];
%            alpha_y*((p(3,4)-x_b2)*sin(theta)*-1+(p(4,4)-y_b2)*cos(theta))+y_b1-p(2,4)];
end