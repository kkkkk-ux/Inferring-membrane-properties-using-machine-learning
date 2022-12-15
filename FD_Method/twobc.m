function res = twobc(ya,yb,para,sigma,p,z0,rb,r0,model)
  psi0 = 0.000;
  h0 = para(2);
  kappa = 1;
  if model == 1
      res = [ ya(1)-psi0;... % psi(0) = 0;
          ya(3)-h0;...   % h(0) = rb;
          ya(4)-r0;...   % r(0) = r0;
          ya(5)-z0;...   % z(0) = z0;
          ya(6)+kappa/2*r0*ya(2)^2/h0^2;...
          yb(1);... % psi(1) = 0;
          yb(4)-rb;...   % r(1) = rb;
          yb(5)-0];      % z(1) = 0;
  else
      res = [ ya(1)-psi0;... % psi(0) = 0;
          ya(3)-h0;...   % h(0) = rb;
          ya(4)-r0;...   % r(0) = r0;
          ya(5)-z0;...   % z(0) = z0;
          ya(6)+kappa/2*r0*ya(2)^2/h0^2-sigma*r0;...  %差在这
          yb(1);... % psi(1) = 0;
          yb(4)-rb;...   % r(1) = rb;
          yb(5)-0];      % z(1) = 0;
  end