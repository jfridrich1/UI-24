import tkinter as tk
import numpy as np
import random
import time

# User chooses clustering method
choice = int(input("Vyberte metodu zhlukovania (1/2/3)\n1: k-means-centroid\n2: k-means-medoid\n3: divizne zhlukovanie: "))

# Tkinter initialization
root = tk.Tk()
max_x_can, max_y_can=600, 600
canvas=tk.Canvas(root, width=max_x_can, height=max_y_can)
canvas.pack()
#canvas.create_rectangle(50, 50, 550, 550)

# Coordinates range for point generation
MAX_X, MAX_Y=5000, 5000
MIN_X, MIN_Y=-5000, -5000

# Clustering parameters
num_points=20000
num_clus=20
radius=1
scaling_down=20
distances=np.zeros((num_points+20, num_clus))

# Arrays for points and centroids
array_points=[]
array_worst_dist=[]
centroids_id=[]
medoids_id=[]

# Initialize 20 unique points with random coordinates
def init_20():
    while len(array_points) < 20:
        new_point = (random.randint(MIN_X, MAX_X), random.randint(MIN_Y, MAX_Y))
        if new_point not in array_points:
            array_points.append(new_point)

# Generate more points around initial points with random offset
def generate_more(count):
    for i in range(count):
        point=random.choice(array_points)
        while True:
            new_x = point[0] + random.randint(-100, 100)
            new_y = point[1] + random.randint(-100, 100)
            if MIN_X<=new_x<=MAX_X and MIN_Y<=new_y<= MAX_Y:
                array_points.append((new_x, new_y))
                break

# k-means++ initialization
def kmeans_plus_plus_init(arr, k, choice):
    if choice==1:
        #print("jednotka")
        first_center=(random.uniform(MIN_X, MAX_X), random.uniform(MIN_Y, MAX_Y))
        centers=[first_center]
    elif choice==2:
        #print("dvojka")
        centers = [random.choice(arr)]

    for _ in range(1, k):
        distances = np.array(
            [min([np.linalg.norm(np.array(point) - np.array(center)) ** 2 for center in centers]) 
                for point in arr]
            )
        
        probs=distances/distances.sum()
        cumprobs = np.cumsum(probs)
        rn=random.random()
        for i, prob in enumerate(cumprobs):
            if rn<prob:
                centers.append(arr[i])
                break
    return centers

# Distance calculation for points to centers
def dist_calc(array_centers):
    points_array=np.array(array_points[:num_points])
    centers_array=np.array(array_centers)
    ans=np.linalg.norm(points_array[:, None] - centers_array, axis=2)
    return ans

# Assign points to clusters
def clustering(dists):
    return np.argmin(dists, axis=1)

# Update centroids or medoids
def update_centroids(clusters):
    new_centroids=[]
    for cluster in clusters:
        if cluster:
            new_centroids.append(np.mean(cluster, axis=0).tolist())
    return new_centroids

def update_medoids(clusters):
    new_medoids=[]
    for cluster in clusters:
        if cluster:
            cluster=np.array(cluster)       #konvert z listu na numpy
            total_dists=np.sum(np.linalg.norm(cluster - cluster[:, None], axis=2), axis=1)
            new_medoids.append(cluster[np.argmin(total_dists)])
    return new_medoids


# Calculate average distance in each cluster
def calculate_average_distance(clusters, centers):
    for cluster, center in zip(clusters, centers):
        avg_dist=np.mean([np.linalg.norm(np.array(point) - np.array(center)) for point in cluster])
        if avg_dist > 500:
            return False
    return True

# Draw clusters and centers
def draw_clusters(clusters, centers):
    for cluster_index, cluster in enumerate(clusters):
        color=cluster_colors[cluster_index % len(cluster_colors)]
        for point in cluster:
            coord_x = (point[0] // scaling_down)+250+50     #50 kvoli okraju
            coord_y = (point[1] // scaling_down)+250+50
            canvas.create_oval(coord_x - radius, coord_y - radius, coord_x + radius, coord_y + radius, fill=color, outline='')

    for center in centers:
        center_x = (center[0] // scaling_down)+250+50
        center_y = (center[1] // scaling_down)+250+50
        canvas.create_oval(center_x-3, center_y-3, center_x+3, center_y+3, outline="black", width=2)

# k-means clustering
def kcent_clustering():
    global centroids_id, choice
    centroids_id=kmeans_plus_plus_init(array_points, num_clus, choice)

    for _ in range(20):
        dists=dist_calc(centroids_id)
        clusters=[[] for _ in range(num_clus)]
        for point_index, cluster_index in enumerate(clustering(dists)):
            clusters[cluster_index].append(array_points[point_index])

        centroids_id=update_centroids(clusters)
        if calculate_average_distance(clusters, centroids_id):
            break

    draw_clusters(clusters, centroids_id)

# k-medoids clustering
def kmed_clustering():
    global medoids_id, choice
    medoids_id=kmeans_plus_plus_init(array_points, num_clus, choice)
    for _ in range(20):
        dists=dist_calc(medoids_id)
        clusters= [[] for _ in range(num_clus)]
        for point_index, cluster_index in enumerate(clustering(dists)):
            clusters[cluster_index].append(array_points[point_index])
        medoids_id=update_medoids(clusters)
        if calculate_average_distance(clusters, medoids_id):
            break
    draw_clusters(clusters, medoids_id)

# Divisive clustering
def divisive_clustering_main():
    clusters=[array_points]
    final_clusters=[]
    while clusters:
        cluster=clusters.pop(0)
        centroid=np.mean(cluster, axis=0)
        avg_dist=np.mean([np.linalg.norm(np.array(point) - centroid) for point in cluster])
        if avg_dist>500:
            clusters.extend(split_cluster(cluster))
        else:
            final_clusters.append(cluster)
    draw_clusters(final_clusters, [np.mean(cluster, axis=0) for cluster in final_clusters])

# Split clusters for divisive clustering
def split_cluster(cluster):
    next_centroids=np.array([cluster[random.randint(0, len(cluster)-1)] for _ in range(2)])
    prev_centroids=np.zeros(next_centroids.shape)
    clusters=[[], []]
    while not np.array_equal(next_centroids, prev_centroids):
        clusters=[[], []]
        for point in cluster:
            distances=[np.linalg.norm(np.array(point) - cent) for cent in next_centroids]
            clusters[np.argmin(distances)].append(point)
        prev_centroids=next_centroids
        next_centroids=np.array([np.mean(clus, axis=0) for clus in clusters])
    return clusters


# Colors for clusters
cluster_colors=[
    "red", "blue", "green", "purple", "orange", "yellow", "pink", "cyan", "magenta", "brown", "lime", 
    "indigo", "teal", "gold", "lightblue", "coral", "darkgreen", "salmon", "violet", "khaki"
]


# Initialize and generate points
init_20()
generate_more(num_points)
num_points+=20

# Select and run the chosen clustering method
start = time.time()
if choice==1:
    print("---Vybrane k-means, kde stred je centroid---")
    kcent_clustering()
elif choice==2:
    print("---Vybrane k-means, kde stred je medoid---")
    kmed_clustering()
elif choice==3:
    print("---Vybrane divizne zhlukovanie---")
    divisive_clustering_main()
else:
    print("Chybny vstup")

end=time.time()
print(f"cas bezania: {round(end-start,2)} sekund")
canvas.create_rectangle(50, 50, 550, 550)
root.mainloop()
