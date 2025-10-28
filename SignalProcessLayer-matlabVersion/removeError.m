function sig=removeError(sig)

    pre=sig(1);
    nxt=sig(3);
    thre=256*0.8;
    for i=1:length(sig)
        sig_high=floor(sig(i)/256);
        sig_low=mod(sig(i),256); 
        if(sig_low==170)
            if abs(pre-sig(i))>thre
                pre_high=floor(pre/256);
                pre_low=mod(pre,256);
                tmp=pre_high*256+sig_high;
                if abs(tmp-pre)<thre
                    sig(i)=tmp;
                else
                    if tmp>pre
                        sig(i)=tmp-256;
                    else
                        sig(i)=tmp+256;
                    end
                end
                
                
            elseif abs(nxt-sig(i))>thre
            end
        end
    end
end