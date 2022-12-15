function dy = shape2(u,y,para,sigma,p,z0,rb,r0,model)
f = para(1);
dy = zeros(6,1);    %差在这
psi = y(1);
dpsi = y(2);
h = y(3);
r = y(4);
%z = y(5);
alpha = y(6);  %差在这
kappa = 1;
if model == 1
    dy(1) = dpsi;
    dy(2) = -f*cos(psi)*h^2/r/kappa ...
        + sigma*h^2*sin(psi)/kappa ...
        + cos(psi)*h^2*sin(psi)/r^2 ...
        + h^2*sin(psi)*alpha/r/kappa ...
        - cos(psi)*h*dpsi/r ...
        + p/2*cos(psi)*r*h^2/kappa;
    dy(3) = 0;
    dy(4) = h*cos(psi);
    dy(5) = -h*sin(psi);
    dy(6) = sigma*(1-cos(psi))*h-kappa/2*h*sin(psi)^2/r^2+kappa/2*dpsi^2/h+p*r*sin(psi)*h;
else
    dy(1) = dpsi;
    dy(2) = -f*cos(psi)*h^2/r/kappa ...
        + cos(psi)*h^2*sin(psi)/r^2 ...
        + h^2*sin(psi)*alpha/r/kappa ...
        - cos(psi)*h*dpsi/r ...
        + p/2*cos(psi)*r*h^2/kappa;
    dy(3) = 0;
    dy(4) = h*cos(psi);
    dy(5) = -h*sin(psi);
    dy(6) = sigma*h-kappa/2*h*sin(psi)^2/r^2+kappa/2*dpsi^2/h + p*h*r*sin(psi);  %差在这
end