import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import silhouette_score


class ClusterAnalysis2D():
    def __init__(self, target=None, hue=None):
        if target is not None:
            self.target = target
        if hue is not None:
            self.hue = hue

    def fit_transform_kmeans(self, train_data, n_clusters, random_state=42):
        estimator = KMeans(n_clusters=n_clusters,
                           init='k-means++',
                           random_state=42)
        
        estimator.fit(train_data)       
        
        self.n_clusters = n_clusters
        self.train_data = train_data
        self.estimator = estimator

        predictions = pd.DataFrame(estimator.predict(train_data), index=train_data.index, columns=['Predictions'])
        self.predictions = predictions['Predictions']
        self.train_data_predictions = pd.concat([train_data, predictions], axis=1)

    def get_homogeneity(self):
        predictions = self.predictions
        target = self.target
        if target is False:
            return None
        else:
            homogeneity = homogeneity_score(labels_true=target, labels_pred=predictions)
        return homogeneity

    def get_completeness(self):
        predictions = self.predictions
        target = self.target
        if target is False:
            return None
        else:
            completeness = completeness_score(labels_true=target, labels_pred=predictions)
        return completeness

    def get_v_measure(self):
        predictions = self.predictions
        target = self.target
        if target is False:
            return None
        else:
            v_measure = v_measure_score(labels_true=target, labels_pred=predictions)
        return v_measure

    def get_silhouette_score(self):
        train_data = self.train_data
        estimator = self.estimator
        if self.n_clusters >=2:
            silhouette = silhouette_score(train_data, estimator.labels_)
        else:
            silhouette = None
        return silhouette

    def get_wccs(self):
        estimator = self.estimator
        wcss = estimator.inertia_
        return wcss

    def optimize_n_clusters(self, train_data, metric='silhouette', max_n_clusters=None):
        if max_n_clusters == None:
            max_n_clusters = train_data.shape[0] // 2
        n_clusters_arr = range(1, max_n_clusters+1)

        metric_values = []
        for n_clusters in n_clusters_arr:
            self.fit_transform_kmeans(train_data=train_data, n_clusters=n_clusters)
            if metric == 'silhouette':
                value = self.get_silhouette_score()
            elif metric == 'v_measure':
                value = self.get_v_measure()
            metric_values.append(value)
        
        opt_n_clusters = n_clusters_arr[np.argmax(metric_values)]
        return opt_n_clusters

    def get_clustering_visualization(self, figsize=(11, 7.5), markersize=14, palette='Dark2', path_to_save=None, dpi=300):
        # preparing data
        vis_data = self.train_data_predictions
        x_col_name = vis_data.columns[0]
        y_col_name = vis_data.columns[1]
        predictions_col = vis_data.columns[2]

        hue_col = self.hue
        opt_n_clusters = self.n_clusters

        x_cluster_centers = self.estimator.cluster_centers_[:, 0]
        y_cluster_centers = self.estimator.cluster_centers_[:, 1]  

        # matplotlib & seaborn code
        sns.set_theme()

        fig = plt.figure(figsize=figsize, dpi=600, linewidth=1.0)

        ax = fig.gca()
        ax.grid(True)
        ax.tick_params(direction='in')

        # scatter plot of data
        scatter = sns.scatterplot(data=vis_data,
                                  x=x_col_name,
                                  y=y_col_name,
                                  hue=hue_col,
                                  palette=palette,
                                  size=hue_col,
                                  s=markersize,
                                  # ax=ax
                                 )

        # visualizing lines between point in cluster centroid
        for clust_number in range(opt_n_clusters):
            temp_df = vis_data.loc[vis_data[predictions_col] == clust_number]
            temp_x = temp_df.iloc[:, 0].to_numpy()
            temp_y = temp_df.iloc[:, 1].to_numpy()

            x_clust_center = x_cluster_centers[clust_number]
            y_clust_center = y_cluster_centers[clust_number]
            
            x_point_cluster = [[x, x_clust_center] for x in temp_x]
            y_point_cluster = [[y, y_clust_center] for y in temp_y]

            ax.scatter(x=x_clust_center,
                       y=y_clust_center,
                       marker='D',
                       s=markersize // 4,
                       color='black')
            
            for x_x_cl, y_y_cl in zip(x_point_cluster, y_point_cluster):
                ax.plot(x_x_cl,
                        y_y_cl,
                        '--',
                        linewidth=1.5,
                        color='black')
        
        plt.tight_layout()

        if path_to_save is not None:
            plt.savefig(path_to_save, dpi=dpi)

        plt.show()


if __name__ == '__main__':
	pass
