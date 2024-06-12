from prettytable import PrettyTable
import pandas as pd
import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from math import ceil, sqrt

class Twomode_Net(object):
    """
    Twomode_Net 类用于将连接关系转化为邻接矩阵的无向图，提供无向图特征展示功能
    这个类可以显示图的基本信息，如节点数、边数、网络密度和集聚系数。

    使用方法：
        实例化，传递参数边关系表格：a = Twomode_Net('边列表.xlsx')
        获取邻接矩阵：a.matrixdf
        展示无向图特征：a.show_data()

    属性:
        __filename（不可访问）: 存储读取数据的文件名。
        __filedf（不可访问）: DataFrame格式存储Excel文件中的原始数据。
        matrixdf（可访问）: 存储生成的邻接矩阵，以DataFrame格式表示。
    """

    def __init__(self, filename):
        """初始化类，读取Excel文件，生成邻接矩阵。"""
        self.__filename = filename
        self.__filedf = self.__read_excel()
        self.matrixdf = self.__matrix()

    def __read_excel(self):
        """读取Excel文件，返回一个DataFrame。"""
        return pd.read_excel(self.__filename)

    def __matrix(self):
        """根据__filedf生成邻接矩阵。"""
        nodes = np.unique(self.__filedf.iloc[:, [0, 1]])
        n_alt = len(nodes)
        adj_matrix = pd.DataFrame(np.zeros((n_alt, n_alt), dtype=int), index=nodes, columns=nodes)
        for _, row in self.__filedf.iterrows():
            adj_matrix.at[row.iloc[0], row.iloc[1]] = 1
            adj_matrix.at[row.iloc[1], row.iloc[0]] = 1
        return adj_matrix

    def show_data(self):
        """创建并使用NetworkX库从邻接矩阵生成图，计算并显示图的基本统计信息。"""
        self.__G = nx.from_pandas_adjacency(self.matrixdf)

        table = PrettyTable()

        table.field_names = ["指标", "值"]
        num_nodes = self.__G.number_of_nodes()

        table.add_row(["网络节点数", num_nodes])
        num_edges = self.__G.number_of_edges()

        table.add_row(["网络边数", num_edges])
        density = nx.density(self.__G)

        table.add_row(["网络密度", f"{density:.4f}"])
        clustering_coefficient = nx.average_clustering(self.__G)

        table.add_row(["集聚系数", f"{clustering_coefficient:.4f}"])
        print(table)

class Con_Component(object):
    """
    Con_component 类用于从给定的邻接矩阵中提取连通分量，分析图的结构，保存连通分量的图形和数据，
    以及生成和显示连通分量的统计信息。

    使用方法：
        实例化，传递无向图df：b = Con_component(a.matrixdf)
        获取连通分量df列表：b.component_list
        将连通分量图片保存到文件夹：b.save_picfolder()
        将连通分量excel保存到文件夹：b.save_filefolder()
        展示连通分量信息：b.show_data()
        绘制最终拼接图：b.creat_finalpic()

    属性:
        __matrixdf（不可访问）: 存储传入的邻接矩阵。
        component_list（可访问）: 存储从主邻接矩阵中提取的所有连通分量的邻接矩阵。
    """

    def __init__(self, matrixdf):
        """初始化时保存邻接矩阵，并获取连通分量"""
        self.__matrixdf = matrixdf
        self.component_list = []
        self.__get_component()

    @classmethod
    def __dfs(cls, matrix, visited, i, component):
        """使用深度优先搜索（DFS）找到所有连通分量"""
        visited[i] = True
        component.append(i)
        for j in range(len(matrix.columns)):
            if matrix.iloc[i, j] == 1 and not visited[j]:
                cls.__dfs(matrix, visited, j, component)

    def __get_component(self):
        """遍历所有节点，应用DFS，收集并保存每个连通分量的邻接矩阵"""
        visited = [False] * len(self.__matrixdf.columns)
        components = []
        for i in range(len(self.__matrixdf.columns)):
            if not visited[i]:
                component = []
                self.__dfs(self.__matrixdf, visited, i, component)
                components.append(component)
        for idx, component in enumerate(components):
            component_matrix = self.__matrixdf.iloc[component, component]
            self.component_list.append(component_matrix)

    def save_filefolder(self, folder_path='连通分量_excel'):
        """将每个连通分量的邻接矩阵保存为Excel文件"""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for idx, component_df in enumerate(self.component_list):
            file_path = os.path.join(folder_path, f"component_{idx + 1}.xlsx")
            component_df.to_excel(file_path, index=True)

    def __create_graph(self, confusion_matrix):
        """从邻接矩阵创建图"""
        G = nx.Graph()
        G.add_nodes_from(confusion_matrix.index)
        for i in confusion_matrix.index:
            for j in confusion_matrix.index:
                if i != j and confusion_matrix.loc[i, j] != 0:
                    G.add_edge(i, j)
        return G

    def __plot_graph(self, G, fig_size, save_path):
        """# 绘制并保存图像"""
        plt.figure(figsize=fig_size)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, node_size=400, node_color="black", with_labels=False)
        plt.savefig(save_path)
        plt.close()

    def save_picfolder(self, folder_path='连通分量_pic'):
        """将每个连通分量的图形保存为PNG文件"""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for idx, component_df in enumerate(self.component_list):
            file_path = os.path.join(folder_path, f"component_{idx + 1}.png")
            G = self.__create_graph(component_df)
            fig_size = (12, 12)
            self.__plot_graph(G, fig_size, file_path)

    def __extr_unistructures(self):
        """使用图哈希识别独特的图结构"""
        unique_graphs = {}
        for component_df in self.component_list:
            G = self.__create_graph(component_df)
            graph_hash = nx.weisfeiler_lehman_graph_hash(G)
            if graph_hash not in unique_graphs:
                unique_graphs[graph_hash] = G
        return unique_graphs

    def show_data(self):
        """显示连通分量的总数和各个大小的独特结构数量及连通分量个数"""
        component_counts = len(self.component_list)
        print(f"总连通分量个数: {component_counts}")

        size_distribution = defaultdict(int)
        component_distribution = defaultdict(int)
        unique_graphs = self.__extr_unistructures()

        for graph in unique_graphs.values():
            size = len(graph.nodes())
            size_distribution[size] += 1

        for component_df in self.component_list:
            size = len(component_df)
            component_distribution[size] += 1

        size_table = PrettyTable(["节点数", "独特结构个数", "连通分量个数"])
        for size in sorted(size_distribution):
            num_structures = size_distribution[size]
            num_components = component_distribution[size]
            size_table.add_row([size, num_structures, num_components])
        print(size_table)

    def creat_finalpic(self, save_path='连通分量图.png'):
        """按节点数排序并创建大图，绘制每个图，并保存"""
        unique_graphs = self.__extr_unistructures()

        sorted_graphs = sorted(unique_graphs.values(), key=lambda g: len(g.nodes()))

        num_graphs = len(sorted_graphs)
        num_cols = num_rows = ceil(sqrt(num_graphs))

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
        axes = axes.flatten()

        for idx, graph in enumerate(sorted_graphs):
            pos = nx.spring_layout(graph)
            nx.draw(graph, pos, ax=axes[idx], node_size=400, node_color="black", with_labels=False)

        for idx in range(num_graphs, num_cols * num_rows):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

if __name__ == '__main__':
    # a = Twomode_Net('边列表.xlsx')
    # # print(a.matrixdf)
    # a.show_data()
    # # a.matrixdf.to_excel('邻接矩阵.xlsx')
    #
    # b = Con_component(a.matrixdf)
    # # print(b.component_list)
    # b.save_picfolder()
    # b.show_data()
    # b.save_filefolder()
    # b.creat_finalpic()

    help(Twomode_Net)
    help(Con_Component)




