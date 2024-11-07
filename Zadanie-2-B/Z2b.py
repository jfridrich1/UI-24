import tkinter as tk
import numpy as np
import random
import time

#User input - vyber zhlukovaca
choice = int(input("Vyberte metodu zhlukovania (1/2/3)\n1: k-means-centroid\n2: k-means-medoid\n3: divizne zhlukovanie: "))

#Tkinter
root = tk.Tk()
max_x_can, max_y_can=600, 600
canvas=tk.Canvas(root, width=max_x_can, height=max_y_can)
canvas.pack()

#Suradnice
MAX_X, MAX_Y=5000, 5000
MIN_X, MIN_Y=-5000, -5000

#Parametre
num_points=20000
num_clus=20
radius=1
scaling_down=20
distances=np.zeros((num_points+20, num_clus))

#Polia
array_points=[]
array_worst_dist=[]
centroids_id=[]
medoids_id=[]

#Inicializacia prvych 20 bodov
def init_20():
    while len(array_points) < 20:
        new_point = (random.randint(MIN_X, MAX_X), random.randint(MIN_Y, MAX_Y))
        if new_point not in array_points:                       #zabezpecenie unikatnosti
            array_points.append(new_point)

#Generovanie zvysnych 40k
def generate_more(count):
    for i in range(count):
        point=random.choice(array_points)
        while True:
            new_x = point[0] + random.randint(-100, 100)
            new_y = point[1] + random.randint(-100, 100)
            if MIN_X<=new_x<=MAX_X and MIN_Y<=new_y<=MAX_Y:    #zabezpecenie rozsahu
                array_points.append((new_x, new_y))
                break

#k-means++, arr=zoznam bodov, k= pocet klastorov, choice=vyber zhlukovaca
def kmeans_pp(arr, k, choice):
    if choice==1:
        first_center=(random.uniform(MIN_X, MAX_X), random.uniform(MIN_Y, MAX_Y))
        centers=[first_center]
    elif choice==2:
        centers = [random.choice(arr)]

    for c in range(1, k):
        #vypocet vzdialenosti
        distances = np.array(
            [min([np.linalg.norm(np.array(point) - np.array(center)) ** 2 for center in centers]) 
                for point in arr]
            )
        #pravdepodobnost umerna k velkosti
        probs=distances/distances.sum()
        cumprobs = np.cumsum(probs)
        rn=random.random()
        for i, prob in enumerate(cumprobs):
            if rn<prob:
                centers.append(arr[i])
                break
    return centers

#Vypocet vzdialenosti
def dist_calc(array_centers):
    #premena na numpy polia
    points_array=np.array(array_points[:num_points])
    centers_array=np.array(array_centers)

    #euklidova vzdialenost, vytvori sa nova os a zapisuju sa vzdialenosti kazdeho bodu a centra
    ans=np.linalg.norm(points_array[:, None] - centers_array, axis=2)
    return ans

#Pridelovanie bodov
def clustering(dists):
    return np.argmin(dists, axis=1)     #najdi minimum zo stlpcov, vyber 1

#Update centroidov
def update_centroids(clusters):
    new_centroids=[]
    for cluster in clusters:
        if cluster:
            new_centroids.append(np.mean(cluster, axis=0).tolist())
    return new_centroids

#Update medoidov
def update_medoids(clusters):
    new_medoids=[]
    for cluster in clusters:
        if cluster:
            cluster=np.array(cluster)       #konvert z listu na numpy
            total_dists=np.sum(np.linalg.norm(cluster - cluster[:, None], axis=2), axis=1)
            new_medoids.append(cluster[np.argmin(total_dists)])
    return new_medoids


#Vypocet priemernej vzdialenosti bodov od centier, podmienka
def calculate_average_distance(clusters, centers):
    for cluster, center in zip(clusters, centers):
        avg_dist=np.mean([np.linalg.norm(np.array(point) - np.array(center)) for point in cluster])
        if avg_dist>500:
            return False
    return True

#Vykreslenie
def draw_clusters(clusters, centers):
    global radius

    #vykreslenie jednotlivych bodov v klastroch
    for cluster_index, cluster in enumerate(clusters):
        color=cluster_colors[cluster_index % len(cluster_colors)]
        for point in cluster:
            coord_x = (point[0] // scaling_down)+250+50     #50 kvoli okraju
            coord_y = (point[1] // scaling_down)+250+50
            canvas.create_oval(coord_x - radius, coord_y - radius, coord_x + radius, coord_y + radius, fill=color, outline='')

    #vykreslenie centrov klastrov
    for center in centers:
        center_x = (center[0] // scaling_down)+250+50
        center_y = (center[1] // scaling_down)+250+50
        canvas.create_oval(center_x-3, center_y-3, center_x+3, center_y+3, outline="black", width=2)

# k-cent clustering
def kcent_clustering():
    global centroids_id, choice
    check=False
    centroids_id=kmeans_pp(array_points, num_clus, choice)

    for c in range(20):
        dists=dist_calc(centroids_id)
        clusters=[[] for z in range(num_clus)]
        for point_index, cluster_index in enumerate(clustering(dists)):
            clusters[cluster_index].append(array_points[point_index])

        centroids_id=update_centroids(clusters)

        #kontrolovanie splnenia podmienky
        if calculate_average_distance(clusters, centroids_id):
            check=True
            break
    
    if check:
        print("Uspesne ukoncenie, podmienka splnena!")
    else:
        print("Neuspesne ukoncenie, dosiahnuty limit iteracii (20)!")

    #vykreslenie ako posledny krok
    draw_clusters(clusters, centroids_id)

#k-medoids clustering
def kmed_clustering():
    global medoids_id, choice
    check=False
    medoids_id=kmeans_pp(array_points, num_clus, choice)

    for c in range(20):
        dists=dist_calc(medoids_id)
        clusters= [[] for z in range(num_clus)]
        for point_index, cluster_index in enumerate(clustering(dists)):
            clusters[cluster_index].append(array_points[point_index])

        medoids_id=update_medoids(clusters)

        #kontrolovanie splnenia podmienky
        if calculate_average_distance(clusters, medoids_id):
            check=True
            break

    if check:
        print("Uspesne ukoncenie, podmienka splnena!")
    else:
        print("Neuspesne ukoncenie, dosiahnuty limit iteracii (20)!")

    #vykreslenie ako posledny krok
    draw_clusters(clusters, medoids_id)




#Divizne zhlukovanie
def divisive_clustering():
    clusters=[array_points]                 #pociatocny klaster, obsahuje vsetky body
    final_clusters=[]                       #ulozenie poslednych klastrov
    while clusters:
        cluster=clusters.pop(0)             #prechadzanie zoznamom pomocou pop
        centroid=np.mean(cluster, axis=0)   #vypocet stredu klastera

        #vypocet vzdialenosti kazdeho bodu od klastra
        avg_dist=np.mean([np.linalg.norm(np.array(point) - centroid) for point in cluster])
        if avg_dist>500:
            #delenie klastra na mensie
            clusters.extend(split_cluster(cluster))
        else:
            #dostatocny klaster, ponechany a ulozeny
            final_clusters.append(cluster)
    
    #vykreslenie ako posledny krok
    draw_clusters(final_clusters, [np.mean(cluster, axis=0) for cluster in final_clusters])


#Delenie klastera na 2 mensie klastre
def split_cluster(cluster):
    #vyber 2 nahodnych pociatocnych centier, pre nove klastre
    next_centroids=np.array([cluster[random.randint(0, len(cluster)-1)] for c in range(2)])

    #kontrola stabilizacie
    prev_centroids=np.zeros(next_centroids.shape)
    clusters=[[], []]                       #priprava na nove klastre

    while not np.array_equal(next_centroids, prev_centroids):
        clusters=[[], []]                   #reset

        #vypocet vzdialenosti bodu od centra pre kazdy bod
        for point in cluster:
            distances=[np.linalg.norm(np.array(point) - cent) for cent in next_centroids]
            #priradenie bodu ku klasteru
            clusters[np.argmin(distances)].append(point)

        prev_centroids=next_centroids
        next_centroids=np.array([np.mean(clus, axis=0) for clus in clusters])

    return clusters


#Dostupne farby na vykreslenie, 20
cluster_colors=[
    "red", "blue", "green", "purple", "orange", "yellow", "pink", "cyan", "magenta", "brown", "lime", 
    "indigo", "teal", "gold", "lightblue", "coral", "darkgreen", "salmon", "violet", "khaki"
]


#Inicializacia a generovanie bodov
init_20()
generate_more(num_points)
num_points+=20

#Vyber zhlukovaca
start = time.time()
if choice==1:
    print("---Vybrane k-means, kde stred je centroid---")
    kcent_clustering()
elif choice==2:
    print("---Vybrane k-means, kde stred je medoid---")
    kmed_clustering()
elif choice==3:
    print("---Vybrane divizne zhlukovanie---")
    divisive_clustering()
else:
    print("Chybny vstup")

end=time.time()
print(f"cas bezania: {round(end-start,2)} sekund")

#kvoli prekryvaniu s bodmi
canvas.create_rectangle(50, 50, 550, 550)
root.mainloop()
