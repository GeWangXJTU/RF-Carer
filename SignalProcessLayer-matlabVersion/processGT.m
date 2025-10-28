function [breath_gt]=processGT(Breath,heart_a,heart_b,gap,gap_a,gap_b)
    
    tmp = abs(fft(Breath(1:gap)));
    len=length(Breath);
    len=len/3;
    breath_gt=zeros(3,len);
    for i=1:3
        tmp=Breath(1+(i-1)*len:i*len);
        tmp=mapminmax(tmp,0,1);
        
        breath_gt(i,:)=tmp;
    end
    
    
end