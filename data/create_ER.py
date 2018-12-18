import networkx as nx
import matplotlib.pyplot as plt
WS = nx.random_graphs.watts_strogatz_graph(50,4,0.3)  #生成包含20个节点、每个节点4个近邻、随机化重连概率为0.3的小世界网络

fout = open('WS_50_4_0.3.txt','w')
for i in WS.edges():
    print(str(i[0])+' '+str(i[1]),file=fout)
fout.close()

pos = nx.circular_layout(WS)          #定义一个布局，此处采用了circular布局方式
nx.draw(WS,pos,with_labels=False,node_size = 30)  #绘制图形
plt.show()
