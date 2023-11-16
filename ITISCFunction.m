function history= ITISCFunction(filename)
load(filename,'initCenters','data','T1','T2','outputFile')
x0=initCenters;
history.x = [];
history.fval = [];
disp('Init center is');
disp(x0) 

options = optimoptions(@fminunc,'OutputFcn',@outfun,'Display','iter','Algorithm','quasi-newton','HessianApproximation','bfgs');
[xsol,fval,exitflag,output] = fminunc(@objfun,x0,options) 
 function stop = outfun(x,optimValues,state)
     stop = false;
     switch state
         case 'iter'
           history.fval = [history.fval; optimValues.fval];
           history.x = cat(3,history.x, x); 
         otherwise
     end
 end
 
 function f = objfun(x)
          f =T2.*log(sum(sum((pdist2(data, x).^2).^(-1/T1),2).^(-T1/T2)));
 end

historyCenters = permute(history.x,[3 1 2]);
save(outputFile,"historyCenters","exitflag") 
end
