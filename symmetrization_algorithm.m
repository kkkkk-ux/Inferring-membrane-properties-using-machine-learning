
filename = './Experimental_data_index.csv';
T = readtable(filename);



for id=T.ID'
    id
    idx = find(T.ID==id);
    
    filename = ['./Experimental_data/Profile',num2str(id),'.txt'];
    XY = load(filename);
    x = XY(:,1);
    y = XY(:,2);
    [~,idmax] = max(y);
    x1 = x(idmax:-1:1)-x(idmax);
    y1 = y(idmax:-1:1);
    s1 = cumsum([0;sqrt(diff(x1).^2+diff(y1).^2)]);
    u1 = s1/s1(end);
    [u1,idx,~] = uniquetol(u1,1e-4);
    x1 = x1(idx);
    y1 = y1(idx);

    
    x2 = x(idmax:end)-x(idmax);
    y2 = y(idmax:end);
    s2 = cumsum([0;sqrt(diff(x2).^2+diff(y2).^2)]);
    u2 = s2/s2(end);
    [u2,idx,~] = uniquetol(u2,1e-4);
    x2 = x2(idx);
    y2 = y2(idx);

    u = linspace(0,1,51);
    yleft = interp1(u1,y1,u);
    yright = interp1(u2,y2,u);
    y = .5*(yleft+yright);
    xleft = -interp1(u1,x1,u);
    xright = interp1(u2,x2,u);
    x = .5*(xleft+xright);

%     plot(x,y,'linewidth',1,'color',c_blue);
%     hold on;
%     plot(-x,y,'linewidth',1,'color',c_blue);
%     plot(x1,y1,'color',c_orange);
%     plot(x2,y2,'color',c_orange);

    writematrix(reshape(XY(:,1),1,[]),['original_shapes.csv'],'WriteMode','append')
    writematrix(reshape(XY(:,2),1,[]),['original_shapes.csv'],'WriteMode','append')

    writematrix(x,['symmetried_shapes.csv'],'WriteMode','append')
    writematrix(y,['symmetried_shapes.csv'],'WriteMode','append')
end

