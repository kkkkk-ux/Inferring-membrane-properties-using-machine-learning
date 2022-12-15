
clear;

saves = dir('..\fitting_results\*save*');

%start from index 1
for k=1:length(saves)
    file_name=saves(k).name;
    filename=['..\fitting_results\',file_name,'\profile_gather.csv'];
    experiment=readmatrix(filename);
   
    
    %Delete previously runed files

    if exist(['..\fitting_results\',file_name,'\profile_BVP.csv'])==2
        recycle('on') %If 'off' is set, it will not enter the recycle bin and will be deleted directly.
        delete(['..\fitting_results\',file_name,'\profile_BVP.csv'])
        delete(['..\fitting_results\',file_name,'\profile_Force.csv'])
    end


    %start from index 0
    for i=0:length(experiment)/6-1
        
        % read parameters from ML's outputs
        p = experiment(6*i+1,1)*1e-5;
        sigma = experiment(6*i+2,1)*1e-4;
        Rb = experiment(6*i+3,1);
        z0 = experiment(6*i+4,1)
        

        yeq = @(u,y,para) shape2(u,y,para,sigma,p,0,30,0.001,2);
        ybc = @(ya,yb,para) twobc(ya,yb,para,sigma,p,0,30,0.001,2);
        yinit = @(u) guess(u,Rb,0.001);
        solinit = bvpinit(linspace(0,1,50),yinit,[0.00,Rb]);
        opts = bvpset('RelTol',1e-5,'AbsTol',1e-10,'NMax',5e5 ...
            ...
            );
        sol = bvp5c(yeq,ybc,solinit,opts);
   
        for z = linspace(z0/4,z0,4)
            for r = linspace(30,Rb,4)
                z_r_and_i=[z,r,i]
                para = sol.parameters;
                yeq = @(u,y,para) shape2(u,y,para,sigma,p,z,r,0.001,2);
                ybc = @(ya,yb,para) twobc(ya,yb,para,sigma,p,z,r,0.001,2);
                sol = bvp5c(yeq,ybc,sol,opts);
            end
        end

        u=linspace(0,1,51);
        r=spline(sol.x,sol.y(4,:),u);
        z=spline(sol.x,sol.y(5,:),u);
        
        % write the FD method results and force
        writematrix(r,['..\fitting_results\',file_name,'\profile_BVP.csv'],'WriteMode','append')
        writematrix(z,['..\fitting_results\',file_name,'\profile_BVP.csv'],'WriteMode','append')
        writematrix(sol.parameters(1),['..\fitting_results\',file_name,'\profile_Force.csv'],'WriteMode','append')
        
    end
    
end






